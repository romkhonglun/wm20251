import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import AUROC
import numpy as np

# Chỉ import VariantNAML
from model import VariantNAML, VariantNAMLConfig


class NAMLLightningModule(pl.LightningModule):
    def __init__(self, config, embedding_dir, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.embedding_dir = embedding_dir

        # --- 1. KHỞI TẠO MODEL MULTI-TASK ---
        self.model = VariantNAML(config)

        # --- 2. METRICS ---
        # AUC cho bài toán Click (Binary)
        self.val_auc = AUROC(task="binary")

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
            # Xóa biến tạm
            del title_w, body_w, cat_w, title_tensor, body_tensor, cat_tensor

        except Exception as e:
            print(f"⚠️ Warning: Could not load embeddings ({e}). Using random init.")

    def forward(self, batch):
        """
        Forward pass xử lý input từ batch dict
        """
        hist_indices = batch['hist_indices']
        cand_indices = batch['cand_indices']
        hist_scrolls = batch['hist_scrolls']
        hist_times = batch['hist_times']

        # hist_clicks là optional, nếu dataset không trả về thì model tự xử lý (coi là toàn 1)
        hist_clicks = batch.get('label_click_hist', None)

        # Gọi model VariantNAML
        return self.model(hist_indices, hist_scrolls, hist_times, cand_indices, hist_clicks)

    def _compute_loss(self, batch, outputs):
        """
        Tính toán Multi-task Loss
        """
        pred_click, pred_scroll, pred_time = outputs

        label_click = batch['label_click']  # (B, Num_Cand) - 0/1
        label_scroll = batch['label_scroll']  # (B, Num_Cand) - 0-100
        label_time = batch['label_time']  # (B, Num_Cand) - Log Time

        # 1. Click Loss: Binary Cross Entropy with Logits
        # pred_click là logits (chưa qua sigmoid)
        loss_click = F.binary_cross_entropy_with_logits(pred_click, label_click)

        # 2. Scroll Loss: MSE
        # Vì Scroll range 0-100 nên MSE rất lớn. Cần giảm weight.
        loss_scroll = F.mse_loss(pred_scroll, label_scroll)

        # 3. Time Loss: MSE
        loss_time = F.mse_loss(pred_time, label_time)

        # --- TỔNG HỢP LOSS (Weighted Sum) ---
        # Click (quan trọng nhất): weight = 1.0
        # Scroll (MSE to ~1000):   weight = 0.002 (để loss về tầm ~2.0)
        # Time (MSE nhỏ ~1.0):     weight = 0.1
        total_loss = loss_click + (0.1 * loss_scroll) + (0.1 * loss_time)

        # Log metrics thành phần để debug
        self.log("L_click", loss_click, on_epoch=True, prog_bar=False)
        self.log("L_scroll", loss_scroll, on_epoch=True, prog_bar=False)
        self.log("L_time", loss_time, on_epoch=True, prog_bar=False)

        return total_loss, pred_click

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, _ = self._compute_loss(batch, outputs)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, logits_click = self._compute_loss(batch, outputs)

        # Tính AUC cho Click Task
        # logits_click -> sigmoid -> probability
        probs = torch.sigmoid(logits_click)
        labels = batch['label_click']

        # Flatten batch để tính AUC trên toàn bộ các cặp mẫu
        self.val_auc(probs.view(-1), labels.view(-1).long())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_auc", self.val_auc, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }