import os
import argparse
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv
from pathlib import Path

# --- IMPORT MODULES ĐÃ TẠO ---
# Giả sử bạn lưu các class vào các file tương ứng:
# model.py -> chứa NAMLConfig, OriginalNAML
# dataset.py -> chứa NAMLDataModule
# lightning_module.py -> chứa NAMLModule (class wrapper đã sửa ở bước trước)

from model import NAMLConfig
from dataset import NAMLDataModule
from lightning_module import NAMLModule

# Load biến môi trường
load_dotenv()

# Tối ưu Threading cho CPU
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ==========================================
# CẤU HÌNH MẶC ĐỊNH
# ==========================================
PROCESSED_DIR = "/processed_parquet"
EMBEDDING_DIR = "/embedding"



def parse_args():
    parser = argparse.ArgumentParser(description="Train NAML model")

    # Paths
    parser.add_argument("--root-dir", type=str, default=PROCESSED_DIR, help="Root path containing train/val folders")
    parser.add_argument("--embedding-dir", type=str, default=EMBEDDING_DIR,
                        help="Directory containing .npy embedding files")

    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")

    # Scheduler
    parser.add_argument("--lr-scheduler", type=str, choices=["onecycle", "cosine"], default="onecycle")
    parser.add_argument("--total-steps", type=int, default=10000, help="Explicit total steps for OneCycleLR")

    # Hardware
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    L.seed_everything(42)

    print("=" * 40)
    print("   NAML TRAINING PIPELINE   ")
    print("=" * 40)

    # 1. Init Config
    # Không cần TIME_FEATURE_NAMLConfig nữa vì ta đang dùng OriginalNAML chuẩn
    config = NAMLConfig()

    # Bạn có thể override config bằng args nếu muốn (ví dụ dropout)
    # config.dropout = 0.3 

    print(f"Model Config: EmbedDim={config.embedding_dim}, Filters={config.num_filters}")
    print(f"Data Dir: {args.root_dir}")
    print(f"Emb Dir:  {args.embedding_dir}")

    # 2. Init DataModule (Phiên bản tối ưu)
    dm = NAMLDataModule(
        root_path=args.root_dir,
        embedding_path=args.embedding_dir,  # DataModule dùng cái này để map ID
        batch_size=args.batch_size,
        history_len=30,  # Có thể đưa ra arg
        neg_ratio=4,
        num_workers=args.num_workers
    )

    # 3. Tính toán Total Steps cho Scheduler
    # Vì dùng IterableDataset, ta cần ước lượng số bước train
    if args.total_steps is None:
        # Giả sử số lượng mẫu train (bạn có thể check file info hoặc hardcode số liệu thật)
        # Ví dụ: EB-NeRD demo ~200k samples
        ESTIMATED_SAMPLES = 200000
        steps_per_epoch = ESTIMATED_SAMPLES // args.batch_size
        calculated_total_steps = steps_per_epoch * args.epochs
        print(f"ℹ️ Auto-calculated total_steps: {calculated_total_steps} (Est. Samples: {ESTIMATED_SAMPLES})")
    else:
        calculated_total_steps = args.total_steps
        print(f"ℹ️ Using provided total_steps: {calculated_total_steps}")

    # 4. Init Lightning Module
    model = NAMLModule(
        config=config,
        embedding_dir=args.embedding_dir,  # Model dùng cái này để load weight
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=calculated_total_steps,
        scheduler_type=args.lr_scheduler
    )

    # 5. Logger (Wandb)
    wandb_logger = WandbLogger(
        project="NAML-RecSys",
        name=f"naml-bs{args.batch_size}-lr{args.lr}",
        log_model=False,
        mode="offline"  # Đổi thành "online" khi chạy thật
    )

    # 6. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="naml-{epoch:02d}-{val/auc:.4f}",
        save_top_k=3,
        monitor="val/auc",
        mode="max",
        verbose=True
    )

    # 7. Trainer
    trainer = L.Trainer(
        accelerator="auto",  # Tự chọn GPU/MPS/CPU
        devices="auto",
        strategy="auto",
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=2),
            RichProgressBar()
        ],
        max_epochs=args.epochs,
        precision="16-mixed",  # Mixed Precision cho GPU (nhanh hơn, tốn ít VRAM)
        log_every_n_steps=50,
        gradient_clip_val=0.5  # Clip gradient để ổn định training
    )

    # 8. Start Training
    print("🚀 Starting training...")
    trainer.fit(model, datamodule=dm)

    print(f"✅ Training finished. Best model path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()