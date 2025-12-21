import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from pathlib import Path

# Import Model và Utils
from model import VariantNAML
from utils import MetricsMeter


class NAMLLightningModule(pl.LightningModule):
    def __init__(self, config, embedding_dir, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.embedding_dir = embedding_dir

        # --- 1. KHỞI TẠO MODEL ---
        self.model = VariantNAML(config)

        # --- 2. METRICS & LOSS ---
        # Sử dụng MetricsMeter để tính toán đồng bộ AUC, NDCG, MRR và Loss
        # Chỉ dùng BCE Loss cho bài toán Click Prediction
        self.loss_weights = {"bce_loss": 1.0}
        self.train_meter = MetricsMeter(self.loss_weights)
        self.val_meter = MetricsMeter(self.loss_weights)

        # --- 3. LOAD PRETRAINED EMBEDDINGS ---
        self._init_embeddings()

    def _init_embeddings(self):
        print(f"Loading embedding weights from {self.embedding_dir}...")
        try:
            # Load mmap (chế độ chỉ đọc, tiết kiệm RAM)
            title_w = np.load(f"{self.embedding_dir}/title_emb.npy", mmap_mode='r')
            body_w = np.load(f"{self.embedding_dir}/body_emb.npy", mmap_mode='r')
            cat_w = np.load(f"{self.embedding_dir}/cat_emb.npy", mmap_mode='r')

            # Convert sang Tensor
            title_tensor = torch.from_numpy(title_w).float()
            body_tensor = torch.from_numpy(body_w).float()
            cat_tensor = torch.from_numpy(cat_w).float()

            # Inject vào NewsEncoder
            self.model.news_encoder.title_emb = nn.Embedding.from_pretrained(title_tensor, freeze=True, padding_idx=0)
            self.model.news_encoder.body_emb = nn.Embedding.from_pretrained(body_tensor, freeze=True, padding_idx=0)
            self.model.news_encoder.cat_emb = nn.Embedding.from_pretrained(cat_tensor, freeze=True, padding_idx=0)

            print("✅ Embeddings injected successfully.")
            # Xóa biến tạm để giải phóng RAM
            del title_w, body_w, cat_w, title_tensor, body_tensor, cat_tensor

        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings ({e}). Using random init.")

    def forward(self, batch):
        """
        Forward pass: Truyền thẳng dict batch vào model.
        Model VariantNAML sẽ tự unpack các key: hist_indices, hist_scroll, hist_time...
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # 1. Forward
        preds = self(batch)  # Output: (Batch, Num_Candidates)

        # 2. Tính Loss & Metrics
        # Dataset trả về 'label_click', MetricsMeter cần key 'labels'
        meter_input = {
            "preds": preds,
            "labels": batch["label_click"]
        }

        losses = self.train_meter.update(meter_input)

        # 3. Log
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        meter_input = {
            "preds": preds,
            "labels": batch["label_click"]
        }

        # Update metrics tích lũy cho validation
        losses = self.val_meter.update(meter_input)

        # Log loss validation
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self):
        self.train_meter.reset()

    def on_validation_epoch_start(self):
        self.val_meter.reset()

    def on_validation_epoch_end(self):
        # Tính toán và Log AUC, NDCG, MRR cuối epoch
        metrics = self.val_meter.compute()

        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k == "auc"))

        print(f"\nEpoch Metrics: AUC={metrics.get('auc', 0):.4f} | NDCG@10={metrics.get('ndcg@10', 0):.4f}")
        self.val_meter.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Sử dụng CosineAnnealingLR
        # T_max: Quy định số epoch để LR giảm từ max xuống min.
        #        Ta lấy luôn self.trainer.max_epochs (được set từ train.py)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-4  # LR tối thiểu không bao giờ xuống thấp hơn mức này
        )
        # total_steps = getattr(self.hparams, "scheduler_total_steps", None)
        #
        # if total_steps is None or total_steps <= 0:
        #     total_steps = 10000
        # print(f"total_steps for OneCycleLR: {total_steps}")
        # max_lr = getattr(self.hparams, "scheduler_max_lr", 3e-3)
        #
        # if max_lr is None:
        #     max_lr = 3e-3
        #
        # print(f"total_steps for OneCycleLR: {total_steps}, max_lr: {max_lr}")
        # scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=max_lr,
        #     total_steps=total_steps,
        #     pct_start=0.1,
        #     anneal_strategy="cos"
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Quan trọng: Cập nhật mỗi epoch
                "frequency": 1,
            },
        }