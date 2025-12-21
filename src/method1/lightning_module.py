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
        self.save_hyperparameters(ignore=['config'])  # Không save config object vào hparams
        self.config = config
        self.embedding_dir = embedding_dir

        # --- 1. LOAD PRETRAINED EMBEDDINGS TRƯỚC ---
        embeddings_dict = self._load_embeddings_data()

        # --- 2. CẬP NHẬT CONFIG TỪ DỮ LIỆU THỰC TẾ ---
        if embeddings_dict is not None:
            # Lấy dimension thực tế từ file title_emb (ví dụ 768 thay vì 1024 mặc định)
            real_dim = embeddings_dict['title'].shape[1]
            if self.config.embedding_dim != real_dim:
                print(f"ℹ️ Updating embedding_dim from {self.config.embedding_dim} to {real_dim}")
                self.config.embedding_dim = real_dim

        # --- 3. KHỞI TẠO MODEL VỚI EMBEDDING ĐÃ LOAD ---
        # Truyền embeddings_dict vào thẳng model
        self.model = VariantNAML(self.config, pretrained_embeddings=embeddings_dict)

        # --- 4. METRICS & LOSS ---
        self.loss_weights = {"bce_loss": 1.0}
        self.train_meter = MetricsMeter(self.loss_weights)
        self.val_meter = MetricsMeter(self.loss_weights)

    def _load_embeddings_data(self):
        """Hàm helper chỉ để load data lên RAM/Tensor"""
        print(f"Loading embedding weights from {self.embedding_dir}...")
        try:
            # Load mmap_mode='r' để không tốn RAM nếu file lớn, nhưng nếu cần GPU thì nên copy
            # Ở đây ta load full vào RAM rồi chuyển sang Tensor
            title_w = np.load(f"{self.embedding_dir}/title_emb.npy")
            body_w = np.load(f"{self.embedding_dir}/body_emb.npy")
            cat_w = np.load(f"{self.embedding_dir}/cat_emb.npy")

            print(f"✅ Loaded embeddings: Title={title_w.shape}, Body={body_w.shape}")

            return {
                'title': torch.from_numpy(title_w).float(),
                'body': torch.from_numpy(body_w).float(),
                'cat': torch.from_numpy(cat_w).float()
            }
        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings ({e}). Model will use Random Init.")
            return None

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        meter_input = {"preds": preds, "labels": batch["label_click"]}
        losses = self.train_meter.update(meter_input)
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        meter_input = {"preds": preds, "labels": batch["label_click"]}
        self.val_meter.update(meter_input)

    def on_train_epoch_start(self):
        self.train_meter.reset()

    def on_validation_epoch_start(self):
        self.val_meter.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_meter.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k == "auc"))
        print(f"\nEpoch Metrics: AUC={metrics.get('auc', 0):.4f}")
        self.val_meter.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}