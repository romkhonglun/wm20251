import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from model import VariantNAML
from utils import MetricsMeter  # Import class MetricsMeter đã tối ưu ở câu trước


class NAMLLightningModule(pl.LightningModule):
    def __init__(self,
                 config,
                 embedding_dir: str,
                 lr=1e-4,
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_embeddings', 'config'])

        # 1. Model & Config
        self.config = config
        self.model = VariantNAML(config)
        self._init_embeddings(embedding_dir)

        # Hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay

        # 2. Metrics & Loss
        # Weights cho loss function (tùy chỉnh theo bài toán)
        self.loss_weights = {"bce_loss": 1.0}

        # [QUAN TRỌNG] Khởi tạo 2 meter riêng biệt.
        # MetricsMeter là nn.Module nên Lightning sẽ tự động move sang GPU.
        self.train_meter = MetricsMeter(self.loss_weights)
        self.val_meter = MetricsMeter(self.loss_weights)

    def _init_embeddings(self, embedding_dir):
        # ... (Giữ nguyên code load embedding của bạn) ...
        print(f"Loading embedding weights from {embedding_dir}...")
        try:
            title_w = np.load(f"{embedding_dir}/title_emb.npy", mmap_mode='r')
            body_w = np.load(f"{embedding_dir}/body_emb.npy", mmap_mode='r')
            cat_w = np.load(f"{embedding_dir}/cat_emb.npy", mmap_mode='r')

            title_tensor = torch.from_numpy(title_w).float()
            body_tensor = torch.from_numpy(body_w).float()
            cat_tensor = torch.from_numpy(cat_w).float()

            self.model.news_encoder.title_emb = nn.Embedding.from_pretrained(title_tensor, freeze=True)
            self.model.news_encoder.body_emb = nn.Embedding.from_pretrained(body_tensor, freeze=True)
            self.model.news_encoder.cat_emb = nn.Embedding.from_pretrained(cat_tensor, freeze=True)

            print("✅ Embeddings injected successfully.")
            del title_w, body_w, cat_w, title_tensor, body_tensor, cat_tensor
        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings ({e}). Using random init.")

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)

        # Lấy preds từ output dict
        preds = output["preds"] if isinstance(output, dict) else output

        # Chuẩn bị input cho meter
        meter_input = {"preds": preds, "labels": batch["labels"]}

        # Update train_meter -> Trả về loss dict (BCE, ListNet, Total Loss)
        # Lưu ý: Ở step này metrics (AUC/NDCG) chưa được compute để tiết kiệm thời gian
        losses = self.train_meter.update(meter_input)

        # Log losses
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return losses["loss"]

    def on_train_epoch_end(self):
        # Reset train meter sau mỗi epoch
        self.train_meter.reset()

    def validation_step(self, batch, batch_idx):
        # Forward pass
        output = self(batch)
        preds = output["preds"] if isinstance(output, dict) else output

        meter_input = {"preds": preds, "labels": batch["labels"]}

        # Update val_meter (Tích lũy preds/labels để tính metrics sau)
        # Hàm update trả về loss của batch hiện tại
        losses = self.val_meter.update(meter_input)

        # Chỉ log val/loss (để theo dõi loss loss curve)
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # [QUAN TRỌNG] Compute metrics trên toàn bộ tập validation
        # Lúc này MetricsMeter mới thực sự gọi các hàm vectorization của TorchMetrics
        metrics = self.val_meter.compute()

        # Log metrics với prefix 'val/' (ví dụ: val/ndcg@10, val/auc)
        # sync_dist=True là bắt buộc nếu chạy nhiều GPU (DDP) để lấy trung bình
        log_metrics = {f"val/{k}": v for k, v in metrics.items()}

        self.log_dict(log_metrics, on_epoch=True, prog_bar=True, sync_dist=True)

        # Reset sau khi đã log xong
        self.val_meter.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-4)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            },
        }