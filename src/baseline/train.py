import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar, ModelSummary

# Import các module đã viết trước đó
from dataset import NewsRecDataModule
from model import VariantNAMLConfig
from lightning_module import NAMLLightningModule

def parse_args():
    parser = argparse.ArgumentParser(description="Train VariantNAML Model with PyTorch Lightning")

    # --- Data Paths ---
    parser.add_argument('--root_data_dir', type=str, default='/home2/congnh/wm/processed_parquet', help='Thư mục chứa data parquet')
    parser.add_argument('--embedding_dir', type=str, default='/home2/congnh/wm/embedding', help='Thư mục chứa embeddings')
    # --- Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size cho training')
    parser.add_argument('--epochs', type=int, default=10, help='Số epoch tối đa')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--npratio', type=int, default=4, help='Số lượng negative samples per positive')

    # --- System ---
    parser.add_argument('--num_workers', type=int, default=4, help='Số CPU workers load data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Mixed precision (16-mixed) giúp train nhanh hơn trên GPU')

    # --- WandB ---
    parser.add_argument('--project_name', type=str, default='NewsRecSys', help='WandB Project Name')
    parser.add_argument('--run_name', type=str, default='baseline', help='Tên run cụ thể')

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Seed
    pl.seed_everything(args.seed)

    # 3. Init DataModule
    dm = NewsRecDataModule(
        root_data_dir=args.root_data_dir,
        batch_size=args.batch_size,
        npratio=args.npratio,
        num_workers=args.num_workers
    )

    # 4. Init Config & Model
    # Load config từ args
    config = VariantNAMLConfig()

    model = NAMLLightningModule(
        config=config,
        embedding_dir=args.embedding_dir,
        lr=args.lr
    )

    # 5. WandB Logger (Offline Mode)
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        offline=False,  # QUAN TRỌNG: Chế độ offline
        log_model=False
    )

    # Log hyperparameters để tiện so sánh sau này
    wandb_logger.log_hyperparams(args)

    # 6. Callbacks
    # Lưu model tốt nhất dựa trên NDCG@10 (metric quan trọng nhất của RecSys)
    checkpoint_callback = ModelCheckpoint(
        filename='Baseline-{epoch:02d}-{val/auc:.4f}',
        monitor='val/mrr',
        mode='max',
        save_top_k=1,
        verbose=True
    )

    # # Dừng sớm nếu NDCG@10 không tăng sau 3 epochs
    # early_stop_callback = EarlyStopping(
    #     monitor='val_ndcg@10',
    #     patience=3,
    #     mode='max',
    #     verbose=True
    # )

    # Theo dõi LR trong quá trình train
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 7. Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor,TQDMProgressBar(refresh_rate=20),ModelSummary()],
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,  # Tăng tốc GPU
        log_every_n_steps=50,
        val_check_interval=1.0,  # Check valid mỗi cuối epoch
        gradient_clip_val=1.0,
        # strategy='ddp' if args.devices > 1 else 'auto' # Tự động dùng DDP nếu nhiều GPU
    )

    # 8. Start Training
    print("🚀 Starting Training...")
    trainer.fit(model, datamodule=dm)

    print(f"✅ Training Done! Best model path: {checkpoint_callback.best_model_path}")
    print("ℹ️  To sync wandb logs later, run: wandb sync <path_to_wandb_dir>")


if __name__ == "__main__":
    main()