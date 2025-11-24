import yaml
import argparse
import torch
import torch.distributed as dist
import os
import gc
import random
import logging
import pickle
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

from torch import optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import module của bạn
from utils import RecommendationMetrics, create_user_history_map, my_collate_fn
from dataset import EbnerdDataset
from model import PLMNR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Training Recommendation System")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--history_size", type=int, default=30)
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def setup_logger(log_dir="logs", rank=0):
    if rank != 0:
        return logging.getLogger(__name__)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# --- TRAINING FUNCTION ---
def train(model, train_loader, optimizer, global_step, epoch, total_epochs, device, logger, rank, scaler):
    model.train()
    running_loss = 0.0

    # Progress bar chỉ hiện ở Rank 0
    if rank == 0:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    else:
        progress_bar = train_loader

    for batch in progress_bar:
        global_step += 1
        his = batch['history'].to(device, non_blocking=True)
        cand = batch['candidate'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss, predict = model(his, cand, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if rank == 0:
            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(train_loader)
    if rank == 0:
        logger.info(f"End Epoch [{epoch + 1}/{total_epochs}] - Train Loss: {avg_loss:.4f}")

    return global_step, avg_loss


# --- VALIDATION FUNCTION (ĐÃ TỐI ƯU HÓA) ---
def validate(model, val_loader, device, logger):
    """
    Validate tối ưu: Tính metric theo từng batch, không gom tensor lớn.
    """
    model.eval()

    total_loss = 0.0

    # Các biến cộng dồn metric
    sum_map = 0.0
    sum_ndcg = 0.0
    sum_mrr = 0.0
    sum_hr = 0.0
    sum_auc = 0.0

    num_batches = len(val_loader)

    # Dùng inference_mode nhanh hơn no_grad
    with torch.inference_mode():
        # Không dùng tqdm ở đây để log đỡ rối, hoặc chỉ print kết quả cuối
        for batch in val_loader:
            his = batch['history'].to(device, non_blocking=True)
            cand = batch['candidate'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            loss, predict = model(his, cand, label)
            total_loss += loss.item()

            # --- TÍNH METRIC NGAY TẠI ĐÂY ---
            # Gọi static method từ class RecommendationMetrics (file utils.py mới)
            b_map, b_ndcg, b_mrr, b_hr, b_auc = RecommendationMetrics.calculate_metrics(
                predict, label, k=10
            )

            sum_map += b_map
            sum_ndcg += b_ndcg
            sum_mrr += b_mrr
            sum_hr += b_hr
            sum_auc += b_auc

    # Tính trung bình
    avg_loss = total_loss / num_batches
    avg_map = sum_map / num_batches
    avg_ndcg = sum_ndcg / num_batches
    avg_mrr = sum_mrr / num_batches
    avg_auc = sum_auc / num_batches

    # Logging
    logger.info(f"--- Validation Report ---")
    logger.info(f"Loss: {avg_loss:.4f} | AUC: {avg_auc:.4f} | MRR: {avg_mrr:.4f}")
    logger.info(f"NDCG@10: {avg_ndcg:.4f} | MAP: {avg_map:.4f}")

    return avg_loss, avg_auc  # Trả về AUC để check best model


# --- MAIN ---
if __name__ == "__main__":
    # 1. Setup DDP Env
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    args = parse_args()
    set_seed(args.seed + local_rank)  # Seed khác nhau cho mỗi GPU

    # 2. Logger (Chỉ Rank 0)
    logger = setup_logger(rank=global_rank)
    if global_rank == 0:
        logger.info(f"Training DDP on {world_size} GPUs.")

    # 3. Path & Config
    data_path = Path(args.dataset_path)
    embedding_path = Path(args.embedding_path)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    hparams = config['hparams']

    # 4. Load Data (Mọi Rank đều load embedding + train data)
    # Lưu ý: Load embedding 1 lần vào RAM mỗi process
    with open(embedding_path, "rb") as file:
        articles_dict = pickle.load(file)

    # Load DataFrame
    behaviors_train_df = pd.read_parquet(data_path / "train" / "behaviors.parquet")
    hist_train_df = pd.read_parquet(data_path / "train" / "history.parquet")
    train_user_history_map = create_user_history_map(hist_train_df)

    del hist_train_df
    gc.collect()

    # Chỉ Rank 0 load validation data
    if global_rank == 0:
        behaviors_val_df = pd.read_parquet(data_path / "validation" / "behaviors.parquet")
        hist_val_df = pd.read_parquet(data_path / "validation" / "history.parquet")
        val_user_history_map = create_user_history_map(hist_val_df)
        del hist_val_df
    else:
        behaviors_val_df = None
        val_user_history_map = None
    gc.collect()

    # 5. Dataset & Dataloader
    train_dataset = EbnerdDataset(
        behaviors_df=behaviors_train_df,
        user_history_map=train_user_history_map,
        articles_dict=articles_dict,
        history_size=args.history_size,
        negative_ratio=args.negative_ratio
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Sampler handles shuffling
        sampler=train_sampler,
        collate_fn=my_collate_fn,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )

    val_dataloader = None
    if global_rank == 0:
        val_dataset = EbnerdDataset(
            behaviors_df=behaviors_val_df,
            user_history_map=val_user_history_map,
            articles_dict=articles_dict,
            history_size=args.history_size,
            negative_ratio=args.negative_ratio
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=my_collate_fn,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True
        )

    # 6. Model & DDP
    model = PLMNR(hparams['model'], device)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 7. Train Loop
    best_val_auc = 0.0
    save_dir = "checkpoints"
    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    global_step = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        # Train
        global_step, train_loss = train(model, train_dataloader, optimizer, global_step, epoch, args.epochs, device,
                                        logger, global_rank, scaler)

        # Sync processes
        dist.barrier()

        # Validate (Rank 0 only)
        if global_rank == 0:
            logger.info("______Validating______")
            start_val = time.time()

            # Lưu ý gọi model.module khi validate
            val_loss, val_auc = validate(model.module, val_dataloader, device, logger)

            val_time = time.time() - start_val
            logger.info(f"⏱️ Validation Time: {str(timedelta(seconds=int(val_time)))}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_path = os.path.join(save_dir, f"best_model_auc_{best_val_auc:.4f}.pth")
                torch.save(model.module.state_dict(), best_path)
                logger.info(f"Saved new best model: {best_path}")

        dist.barrier()

    dist.destroy_process_group()