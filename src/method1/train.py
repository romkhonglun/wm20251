import os
import argparse
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv
from pathlib import Path

# Import các module của Method 1
from dataset import NAMLDataModule
from model import VariantNAMLConfig
from lightning_module import NAMLLightningModule

# Load biến môi trường (nếu có file .env)
load_dotenv()

# Tối ưu hóa hiệu năng
torch.set_float32_matmul_precision('medium')  # Tăng tốc trên GPU đời mới (A100, H100, 3090...)
torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ==========================================
# CẤU HÌNH MẶC ĐỊNH
# ==========================================
DEFAULT_ROOT_DIR = "/home2/congnh/wm/processed_parquet"
DEFAULT_EMB_DIR = "/home2/congnh/wm/embedding"


def parse_args():
    parser = argparse.ArgumentParser(description="Train NAML Method 1 (VariantNAML)")

    # --- Paths ---
    parser.add_argument("--root-dir", type=str, default=DEFAULT_ROOT_DIR,
                        help="Thư mục chứa dữ liệu đã tiền xử lý (chứa train/val folders và article_ids.npy)")
    parser.add_argument("--embedding-dir", type=str, default=DEFAULT_EMB_DIR,
                        help="Thư mục chứa các file embedding weight (.npy)")

    # --- Training Hyperparams ---
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Số lượng epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # --- Model Config (Optional override) ---
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # --- Data Config ---
    parser.add_argument("--history-len", type=int, default=30, help="Độ dài lịch sử đọc")
    parser.add_argument("--neg-ratio", type=int, default=4, help="Số lượng negative samples cho mỗi positive")
    parser.add_argument("--num-workers", type=int, default=4, help="Số worker cho DataLoader")

    # --- System ---
    parser.add_argument("--wandb-project", type=str, default="NAML-Method1", help="Tên project trên WandB")
    parser.add_argument("--wandb-name", type=str, default=None, help="Tên run cụ thể (optional)")
    parser.add_argument("--offline", action="store_true", help="Chạy chế độ offline (không sync wandb)")

    return parser.parse_args()


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()

    # Set seed để đảm bảo tái lập kết quả
    L.seed_everything(42)

    print("=" * 50)
    print(f"🚀 TRAINING METHOD 1: VariantNAML")
    print(f"   Data Dir:      {args.root_dir}")
    print(f"   Embedding Dir: {args.embedding_dir}")
    print(f"   Batch Size:    {args.batch_size}")
    print(f"   Learning Rate: {args.lr}")
    print("=" * 50)

    # 1. Khởi tạo Config & Model
    config = VariantNAMLConfig()
    config.dropout = args.dropout  # Override từ arguments

    model = NAMLLightningModule(
        config=config,
        embedding_dir=args.embedding_dir,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 2. Khởi tạo DataModule
    # Lưu ý: embedding_path của DataModule trỏ về root_dir vì cần file article_ids.npy ở đó
    dm = NAMLDataModule(
        root_path=args.root_dir,
        embedding_path=args.embedding_dir,
        batch_size=args.batch_size,
        history_len=args.history_len,
        neg_ratio=args.neg_ratio,
        num_workers=args.num_workers
    )

    # 3. Logger (WandB)
    run_name = args.wandb_name if args.wandb_name else f"M1-bs{args.batch_size}-lr{args.lr}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        log_model=False,
        mode="offline" if args.offline else "online"
    )

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/method1",
        filename="naml-m1-{epoch:02d}-{val/auc:.4f}",
        save_top_k=3,
        monitor="val/auc",
        mode="max",
        verbose=True,
        save_last=True
    )

    # 5. Trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",  # Tự động dùng DDP nếu có nhiều GPU
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=2),
            RichProgressBar(refresh_rate=1)
        ],
        max_epochs=args.epochs,
        precision="16-mixed",  # Mixed precision giúp train nhanh hơn và ít VRAM hơn
        gradient_clip_val=0.5,  # Chống bùng nổ gradient
        log_every_n_steps=50
    )

    # 6. Bắt đầu Train
    trainer.fit(model, datamodule=dm)

    print(f"\n✅ Training Completed!")
    print(f"🏆 Best Model: {checkpoint_callback.best_model_path}\n")


if __name__ == "__main__":
    main()