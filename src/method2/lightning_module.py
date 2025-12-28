import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import numpy as np
import gc
from pathlib import Path

# Import Models
from model import TIME_FEATURE_NAML, TIME_FEATURE_NAMLConfig
# Import Utils
from utils import MetricsMeter


class NAMLLightningModule(L.LightningModule):

    def __init__(
            self,
            config=None,
            embedding_path=None,
            lr=1e-4,
            weight_decay=1e-5,
            scheduler="onecycle",
            scheduler_total_steps=None,
            scheduler_max_lr=None,
            scheduler_t_max=None,
            use_compile=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config if config is not None else TIME_FEATURE_NAMLConfig()

        # --- 1. Init Embeddings (Tối ưu bộ nhớ) ---
        vectors_tensor = self._load_embeddings_safely(embedding_path)

        # --- 2. Init Model ---
        # Truyền tensor đã xử lý vào model
        raw_model = TIME_FEATURE_NAML(self.config, vectors_tensor)

        # Sau khi model đã init xong (đã lưu weights vào nn.Embedding),
        # ta xóa biến tạm vectors_tensor để giải phóng RAM triệt để
        del vectors_tensor
        gc.collect()

        # --- 3. Compile Model (Optional) ---
        if use_compile and hasattr(torch, "compile"):
            print("🚀 Compiling model with torch.compile...")
            self.model = torch.compile(raw_model)
        else:
            self.model = raw_model

        # --- 4. Metrics ---
        self.loss_weights = {"bce_loss": 1.0}
        self.train_meter = MetricsMeter(self.loss_weights)
        self.val_meter = MetricsMeter(self.loss_weights)

    def _load_embeddings_safely(self, embedding_path):
        """Hàm load embedding tối ưu bộ nhớ, tránh lỗi read-only numpy"""
        if embedding_path and Path(embedding_path).exists():
            print(f"Loading vectors from {embedding_path}...")

            # 1. Load Numpy
            # Không dùng mmap_mode='r' để tránh lỗi pytorch conflict
            vectors_np = np.load(embedding_path)

            # 2. Tạo Tensor Copy (Explicit Copy)
            # Dùng torch.tensor() thay vì as_tensor/from_numpy để đảm bảo sở hữu memory riêng
            vectors_tensor = torch.tensor(vectors_np, dtype=torch.float32)

            # 3. Xóa ngay Numpy Array khỏi RAM
            del vectors_np
            gc.collect()

            # 4. Update config dim nếu cần
            real_dim = vectors_tensor.shape[1]
            if self.config.pretrained_dim != real_dim:
                print(f"⚠️ Updating config dim from {self.config.pretrained_dim} to {real_dim}")
                self.config.pretrained_dim = real_dim

            # 5. Normalize (In-place để tiết kiệm RAM)
            print(" -> Normalizing vectors...")
            norm = torch.norm(vectors_tensor, p=2, dim=1, keepdim=True)
            vectors_tensor.div_(norm + 1e-4)  # In-place division

            print("✅ Embeddings loaded & optimized.")
            return vectors_tensor
        else:
            print("⚠️ Embedding path not found. Using Random Embeddings.")
            return torch.randn(100000, self.config.pretrained_dim)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # Tự động ép kiểu float16/float32 tùy theo precision của model
        try:
            target_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        # Cast floating tensors
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=target_dtype)
                if torch.isnan(batch[k]).any():
                    print(f"⚠️ Warning: NaN found in {k}")

        output = self(batch)
        meter_input = {"preds": output["preds"], "labels": batch["labels"]}

        losses = self.train_meter.update(meter_input)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True
        )
        return losses["loss"]

    def on_train_epoch_start(self):
        self.train_meter.reset()

    def on_validation_epoch_start(self):
        self.val_meter.reset()

    def validation_step(self, batch, batch_idx):
        try:
            target_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=target_dtype)

        output = self(batch)
        meter_input = {"preds": output["preds"], "labels": batch["labels"]}

        losses = self.val_meter.update(meter_input)
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True)
        return losses["loss"]

    def on_validation_epoch_end(self):
        metrics = self.val_meter.compute()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.val_meter.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler_choice = getattr(self.hparams, "scheduler", "onecycle")

        if scheduler_choice == "onecycle":
            total_steps = getattr(self.hparams, "scheduler_total_steps", 10000) or 10000
            max_lr = getattr(self.hparams, "scheduler_max_lr", 3e-3) or 3e-3

            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos"
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        if scheduler_choice == "cosine":
            t_max = getattr(self.hparams, "scheduler_t_max", 10000) or 10000
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer