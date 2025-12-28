# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pytorch_lightning as pl
# import numpy as np
# from pathlib import Path
#
# # Import Model và Utils
# from model import FullNewsRecModel as VariantNAML
# from utils import MetricsMeter
#
# class NAMLLightningModule(pl.LightningModule):
#     def __init__(self, config, embedding_dir, lr=1e-3, weight_decay=1e-4):
#         super().__init__()
#         self.save_hyperparameters(ignore=['config'])  # Không save config object vào hparams
#         self.config = config
#         self.embedding_dir = embedding_dir
#
#
#         # --- 3. KHỞI TẠO MODEL VỚI EMBEDDING ĐÃ LOAD ---
#         # Truyền embeddings_dict vào thẳng model
#         self.model = VariantNAML(self.config)
#
#         # --- 4. METRICS & LOSS ---
#         self.loss_weights = {"bce_loss": 1.0}
#         self.train_meter = MetricsMeter(self.loss_weights)
#         self.val_meter = MetricsMeter(self.loss_weights)
#
#         self._init_embeddings()
#
#     def _init_embeddings(self):
#         print(f"Loading embedding weights from {self.embedding_dir}...")
#         try:
#             # Load mmap (chế độ chỉ đọc, tiết kiệm RAM)
#             title_w = np.load(f"{self.embedding_dir}/title_emb.npy", mmap_mode='r')
#             body_w = np.load(f"{self.embedding_dir}/body_emb.npy", mmap_mode='r')
#             cat_w = np.load(f"{self.embedding_dir}/cat_emb.npy", mmap_mode='r')
#
#             # Convert sang Tensor
#             title_tensor = torch.from_numpy(title_w).float()
#             body_tensor = torch.from_numpy(body_w).float()
#             cat_tensor = torch.from_numpy(cat_w).float()
#
#             # Inject vào NewsEncoder
#             self.model.news_encoder.title_emb = nn.Embedding.from_pretrained(title_tensor, freeze=True, padding_idx=0)
#             self.model.news_encoder.body_emb = nn.Embedding.from_pretrained(body_tensor, freeze=True, padding_idx=0)
#             self.model.news_encoder.cat_emb = nn.Embedding.from_pretrained(cat_tensor, freeze=True, padding_idx=0)
#
#             print("✅ Embeddings injected successfully.")
#             # Xóa biến tạm
#             del title_w, body_w, cat_w, title_tensor, body_tensor, cat_tensor
#
#         except Exception as e:
#             print(f"⚠️ Warning: Could not load embeddings ({e}). Using random init.")
#
#     def forward(self, batch):
#         return self.model(batch)
#
#     def training_step(self, batch, batch_idx):
#         output = self(batch)
#         meter_input = {"preds": output["preds"], "labels": batch["labels"]}
#         # Sử dụng train_meter
#         losses = self.train_meter.update(meter_input)
#
#         self.log_dict(
#             {f"train/{k}": v for k, v in losses.items()},
#             on_step=True, on_epoch=True, prog_bar=True
#         )
#         return losses["loss"]
#
#     def validation_step(self, batch, batch_idx):
#         output = self(batch)
#         meter_input = {"preds": output["preds"], "labels": batch["labels"]}
#
#         # Hàm update của MetricsMeter trả về dict chứa "loss"
#         # Sử dụng val_meter
#         losses = self.val_meter.update(meter_input)
#
#         # Log loss theo batch mà không sợ lẫn với train step
#         self.log("val/loss", losses["loss"], on_step=False, on_epoch=True)
#
#     def on_train_epoch_start(self):
#         self.train_meter.reset()
#
#     def on_validation_epoch_start(self):
#         self.val_meter.reset()
#
#     def on_validation_epoch_end(self):
#         metrics = self.val_meter.compute()
#         self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)
#
#         # Reset sau khi đã log xong kết quả epoch
#         self.val_meter.reset()
#
#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-4)
#         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import gc
from pathlib import Path

# Import Model và Utils
from model import FullNewsRecModel as VariantNAML
from utils import MetricsMeter


class NAMLLightningModule(pl.LightningModule):
    def __init__(self, config, embedding_dir, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])  # Không save config object vào hparams
        self.config = config
        self.embedding_dir = embedding_dir

        # --- 3. KHỞI TẠO MODEL ---
        self.model = VariantNAML(self.config)

        # --- 4. LOAD EMBEDDINGS (FIX MEMORY ISSUE) ---
        self._init_embeddings()

        # --- 5. METRICS & LOSS ---
        self.loss_weights = {"bce_loss": 1.0}
        self.train_meter = MetricsMeter(self.loss_weights)
        self.val_meter = MetricsMeter(self.loss_weights)

    def _init_embeddings(self):
        print(f"Loading embedding weights from {self.embedding_dir}...")
        try:
            # --- TITLE ---
            print(" -> Loading Title Embedding...")
            # Load bình thường (bỏ mmap_mode) để kiểm soát RAM
            val = np.load(f"{self.embedding_dir}/title_emb.npy")
            tensor = torch.tensor(val, dtype=torch.float32)
            del val  # Xóa numpy array khỏi RAM ngay

            self.model.news_encoder.title_emb = nn.Embedding.from_pretrained(
                tensor, freeze=True, padding_idx=0
            )
            del tensor  # Xóa tensor tạm
            gc.collect()  # Dọn dẹp RAM

            # --- BODY ---
            print(" -> Loading Body Embedding...")
            val = np.load(f"{self.embedding_dir}/body_emb.npy")
            tensor = torch.tensor(val, dtype=torch.float32)
            del val

            self.model.news_encoder.body_emb = nn.Embedding.from_pretrained(
                tensor, freeze=True, padding_idx=0
            )
            del tensor
            gc.collect()

            # --- CAT ---
            print(" -> Loading Category Embedding...")
            val = np.load(f"{self.embedding_dir}/cat_emb.npy")
            tensor = torch.tensor(val, dtype=torch.float32)
            del val

            self.model.news_encoder.cat_emb = nn.Embedding.from_pretrained(
                tensor, freeze=True, padding_idx=0
            )
            del tensor
            gc.collect()

            print("✅ Embeddings injected successfully (Method 1).")

        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings ({e}). Using random init.")

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        meter_input = {"preds": output["preds"], "labels": batch["labels"]}
        # Sử dụng train_meter
        losses = self.train_meter.update(meter_input)

        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True, on_epoch=True, prog_bar=True
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        meter_input = {"preds": output["preds"], "labels": batch["labels"]}

        # Hàm update của MetricsMeter trả về dict chứa "loss"
        losses = self.val_meter.update(meter_input)

        # Log loss theo batch
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        self.train_meter.reset()

    def on_validation_epoch_start(self):
        self.val_meter.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_meter.compute()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.val_meter.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-4)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}