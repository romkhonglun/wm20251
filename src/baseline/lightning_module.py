import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import numpy as np
from pathlib import Path

# Import class model và metrics của bạn
from model import NAMLConfig, OriginalNAML
from utils import MetricsMeter


class NAMLModule(L.LightningModule):
    def __init__(
            self,
            config: NAMLConfig,
            embedding_dir: str,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            total_steps: int = 10000,  # Quan trọng cho OneCycleLR
            scheduler_type: str = "onecycle"
    ):
        super().__init__()
        # 1. Lưu hyperparameters tự động (truy cập qua self.hparams.lr, v.v.)
        self.save_hyperparameters()

        self.config = config
        self.embedding_dir = embedding_dir

        # 2. Khởi tạo Model
        self.model = OriginalNAML(config)

        # 3. Khởi tạo Metrics Meter
        # Kết hợp BCE (cho classification) và ListNet (cho ranking)
        self.loss_weights = {"bce_loss": 1.0}
        self.train_meter = MetricsMeter(self.loss_weights)
        self.val_meter = MetricsMeter(self.loss_weights)

        # 4. Load Pre-trained Embeddings
        self._init_embeddings()

    def forward(self, batch):
        # [QUAN TRỌNG] Model OriginalNAML yêu cầu 2 tham số đầu vào riêng biệt
        # Dataset trả về dict, ta cần unpack nó ra
        hist_indices = batch["hist_indices"]
        cand_indices = batch["cand_indices"]

        # Gọi model
        scores = self.model(hist_indices, cand_indices)
        return scores

    def _init_embeddings(self):
        print(f"Loading embedding weights from {self.embedding_dir}...")
        try:
            emb_path = Path(self.embedding_dir)
            # Load numpy mmap mode để tiết kiệm RAM lúc init
            title_w = np.load(emb_path / "title_emb.npy", mmap_mode="r")
            body_w = np.load(emb_path / "body_emb.npy", mmap_mode="r")
            cat_w = np.load(emb_path / "cat_emb.npy", mmap_mode="r")

            # Convert sang Tensor (Float)
            title_tensor = torch.from_numpy(np.array(title_w)).float()
            body_tensor = torch.from_numpy(np.array(body_w)).float()
            cat_tensor = torch.from_numpy(np.array(cat_w)).float()

            # Inject vào Model
            # freeze=True: Không train lại Embedding này (đỡ tốn VRAM)
            # padding_idx=0: Vector tại index 0 sẽ luôn là vector 0
            self.model.news_encoder.title_emb = nn.Embedding.from_pretrained(
                title_tensor, freeze=True, padding_idx=0
            )
            self.model.news_encoder.body_emb = nn.Embedding.from_pretrained(
                body_tensor, freeze=True, padding_idx=0
            )
            self.model.news_encoder.cat_emb = nn.Embedding.from_pretrained(
                cat_tensor, freeze=True, padding_idx=0
            )

            print("✅ Embeddings injected successfully.")
            # Xóa biến tạm
            del title_w, body_w, cat_w, title_tensor, body_tensor, cat_tensor

        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings ({e}). Using random init.")

    def training_step(self, batch, batch_idx):
        # 1. Forward Pass
        preds = self(batch)  # shape: (Batch, Cand_Len)

        # 2. Tính Loss & Metrics
        # MetricsMeter nhận dict input
        meter_input = {"preds": preds, "labels": batch["labels"]}
        losses = self.train_meter.update(meter_input)

        # 3. Log
        # Log loss tổng
        self.log("train/loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        # Log loss thành phần (để debug xem cái nào đang chiếm ưu thế)
        if "bce_loss" in losses:
            self.log("train/bce", losses["bce_loss"], on_step=False, on_epoch=True)
        if "listnet_loss" in losses:
            self.log("train/listnet", losses["listnet_loss"], on_step=False, on_epoch=True)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        meter_input = {"preds": preds, "labels": batch["labels"]}

        # Update metrics tích lũy
        self.val_meter.update(meter_input)

        # Tính loss cho batch hiện tại (chỉ để log, không backprop)
        # Lưu ý: val_meter.update trả về loss dict của batch đó
        # (cần sửa MetricsMeter một chút để nó return loss dict ở update - code trước đã có return)
        # Nếu MetricsMeter của bạn không return loss ở update, bạn cần gọi compute loss riêng.
        # Nhưng theo code MetricsMeter bạn đưa thì nó có return metrics dict (chứa loss).

        # Chúng ta không log loss ở step validation để tránh log quá nhiều,
        # chỉ log aggregated metric ở cuối epoch.

    def on_validation_epoch_end(self):
        # 1. Compute Metrics tích lũy cả epoch
        metrics = self.val_meter.compute()

        # 2. Log Metrics (AUC, MRR, NDCG...)
        # prog_bar=True để hiện AUC lên thanh tiến trình
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k == "auc"))

        print(f"\nExample Val Metrics: AUC={metrics.get('auc', 0):.4f} | NDCG@10={metrics.get('ndcg@10', 0):.4f}")

        # 3. Reset cho epoch sau
        self.val_meter.reset()

    def on_train_epoch_start(self):
        self.train_meter.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        if self.hparams.scheduler_type == "onecycle":
            # Với IterableDataset, Trainer đôi khi khó ước lượng total steps chính xác
            # Nên truyền thủ công total_steps vào __init__ là an toàn nhất.
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.hparams.total_steps,
                pct_start=0.2,  # 20% warm-up
                anneal_strategy="cos"
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"  # Update mỗi batch
                }
            }
        if self.hparams.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=5
            )


        return optimizer