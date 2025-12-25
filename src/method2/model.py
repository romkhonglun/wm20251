import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================
# 1. CONFIGURATION
# ==========================================
class TIME_FEATURE_NAMLConfig:
    def __init__(self):
        # --- Dimensions ---
        # Kích thước vector BERT/RoBERTa từ file body_emb.npy
        self.pretrained_dim = 1024

        # Kích thước vector chuẩn (d_model) cho toàn bộ mạng
        self.window_size = 128

        # --- Feature Engineering Specs (Khớp với dataset.py) ---
        # [log_views, log_inviews, sentiment, log_read_time, freshness]
        self.num_numerical_features = 6

        # Kích thước embedding cho mỗi feature số sau khi qua hàm Sinusoidal
        self.sinusoidal_dim = 16

        # [log_hist_len, time_of_day, norm_age]
        self.num_impression_features = 3

        # Category ID
        self.num_categories = 50  # Tùy chỉnh theo dataset thực tế
        self.category_emb_dim = 32

        self.num_interests = 5
        self.dropout = 0.2

        # --- Ranker Specs ---
        self.rankformer_layers = 3
        self.rankformer_heads = 4
        self.rankformer_ffn_dim = 512


# ======================================
# ==========================================
# 2. BUILDING BLOCKS (UTILITIES)====
class SinusoidalEmbedding(nn.Module):
    """
    Biến đổi một giá trị số thực (scalar) thành vector (vectorization).
    Giúp mạng nơ-ron học tốt hơn các tính chất thứ tự và độ lớn.
    """

    def __init__(self, embedding_dim: int, M: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.M = M

    def forward(self, x: torch.Tensor):
        # x shape đầu vào: [Batch, Seq] hoặc [Batch]
        # Nếu x có dimension cuối là 1 [Batch, Seq, 1], cần squeeze
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        device = x.device
        half_dim = self.embedding_dim // 2

        # Công thức Positional Encoding chuẩn
        emb = np.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))

        # [Batch, Seq, 1] * [1, 1, Half_Dim] -> [Batch, Seq, Half_Dim]
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)

        # Concat sin và cos -> [Batch, Seq, Dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ==========================================
# 3. ENCODERS
# ==========================================

class CandidateFeatureEncoder(nn.Module):
    """
    Mã hóa bài báo ứng viên (Candidate).
    Kết hợp: Body Vector + 5 Feature Số + 1 Feature Category.
    """

    def __init__(self, config, pretrained_vectors):
        super().__init__()
        self.config = config

        # 1. Pretrained Body Embedding (Đóng băng để tiết kiệm VRAM)
        self.body_emb = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.body_proj = nn.Sequential(
            nn.Linear(config.pretrained_dim, config.window_size),
            nn.LayerNorm(config.window_size),
            nn.GELU()
        )

        # 2. Numerical Features (5 features -> 5 Sinusoidal Layers)
        self.num_sinusoidal = nn.ModuleList([
            SinusoidalEmbedding(config.sinusoidal_dim)
            for _ in range(config.num_numerical_features)
        ])

        # 3. Categorical Feature
        self.cat_emb = nn.Embedding(config.num_categories + 1, config.category_emb_dim, padding_idx=0)

        # 4. Fusion Layer
        # Tính tổng kích thước input sau khi concat tất cả
        total_input_dim = (
                config.window_size +
                (config.num_numerical_features * config.sinusoidal_dim) +
                config.category_emb_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, config.window_size),
            nn.LayerNorm(config.window_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

    def forward(self, indices, num_feats, cat_feats):
        """
        indices:   [Batch, Seq]
        num_feats: [Batch, Seq, 5]
        cat_feats: [Batch, Seq, 1]
        """
        # 1. Lấy dtype hiện tại của model (sẽ là float16 nếu dùng precision=16)
        # Vì self.fusion là Sequential, ta lấy từ tham số đầu tiên của module
        target_dtype = next(self.parameters()).dtype

        # 2. Ép kiểu đầu vào (Chỉ ép kiểu các float tensor, indices/cat thường là Long)
        num_feats = num_feats.to(target_dtype)

        # A. Xử lý Body
        body_vec = self.body_emb(indices)  # [B, S, 768]
        body_vec = self.body_proj(body_vec)  # [B, S, Window_Size]

        # B. Xử lý Numerical (Sinusoidal)
        num_embs = []
        for i in range(self.config.num_numerical_features):
            col = num_feats[:, :, i]  # [B, S]
            # Đảm bảo đầu ra của sinusoidal khớp dtype
            emb = self.num_sinusoidal[i](col).to(target_dtype)
            num_embs.append(emb)
        num_vec = torch.cat(num_embs, dim=-1)  # [B, S, 5 * Sin_Dim]

        # C. Xử lý Categorical
        cat_vec = self.cat_emb(cat_feats.squeeze(-1))  # [B, S, Cat_Dim]

        # D. Fusion
        # Đảm bảo tất cả các thành phần trong list concat đều cùng dtype
        concat = torch.cat([
            body_vec.to(target_dtype),
            num_vec,
            cat_vec.to(target_dtype)
        ], dim=-1)

        out = self.fusion(concat)  # [B, S, Window_Size]

        return out


class ImpressionEncoder(nn.Module):
    """
    Mã hóa ngữ cảnh hiển thị (Impression).
    Input: [log_hist_len, time_of_day, norm_age]
    """

    def __init__(self, config):
        super().__init__()
        self.sinusoidal = nn.ModuleList([
            SinusoidalEmbedding(config.sinusoidal_dim) for _ in range(config.num_impression_features)
        ])

        input_dim = config.num_impression_features * config.sinusoidal_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.window_size),
            nn.GELU()
        )

    def forward(self, imp_feats):
        # Lấy dtype hiện tại của mô hình (Half nếu dùng precision=16)
        target_dtype = next(self.parameters()).dtype

        # Ép kiểu đầu vào
        imp_feats = imp_feats.to(target_dtype)

        embs = []
        for i in range(len(self.sinusoidal)):
            col = imp_feats[:, i]
            # Đảm bảo đầu ra của sinusoidal cũng đồng nhất dtype
            emb = self.sinusoidal[i](col.unsqueeze(1)).squeeze(1)
            embs.append(emb.to(target_dtype))

        concat = torch.cat(embs, dim=-1)

        # Lúc này concat đã cùng dtype với self.proj
        return self.proj(concat)


class MultiInterestUserEncoder(nn.Module):
    """
    Mã hóa lịch sử User thành K vectors sở thích.
    """

    def __init__(self, config, pretrained_vectors):
        super().__init__()
        self.dim = config.window_size
        self.num_interests = config.num_interests

        # Embedding layer (Share weight với Candidate Encoder để đồng bộ không gian vector)
        self.body_emb = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)
        self.body_proj = nn.Sequential(
            nn.Linear(config.pretrained_dim, self.dim),
            nn.GELU()
        )

        # Interaction Projection (Scroll, Time, Recency)
        # Input: Scroll(1) + Time(1) + Recency(1) = 3
        self.inter_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, self.dim)
        )
        self.layer_norm = nn.LayerNorm(self.dim)

        # Transformer Encoder để học chuỗi lịch sử
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=4,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Multi-Interest Attention Mechanism
        # Learnable Queries (Seed Vectors)
        self.interest_queries = nn.Parameter(torch.randn(self.num_interests, self.dim))
        self.interest_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, hist_indices, hist_scroll, hist_time, hist_diff, mask=None):
        # 1. Embed Content
        x = self.body_emb(hist_indices)  # [B, S, 768]
        x = self.body_proj(x)  # [B, S, Window_Size]

        # 2. Embed Interaction
        inter = torch.stack([hist_scroll, hist_time, hist_diff], dim=-1)  # [B, S, 3]
        inter = self.inter_proj(inter)

        # 3. Combine & Normalize
        x = self.layer_norm(x + inter)

        # 4. Sequence Encoding
        # mask=True là padding
        seq_rep = self.transformer(x, src_key_padding_mask=mask)

        # 5. Extract Interests
        batch_size = x.size(0)
        # Expand queries cho từng user trong batch -> [B, K, Dim]
        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Query: Interests, Key/Value: History Sequence
        interests, _ = self.interest_attn(
            query=queries,
            key=seq_rep,
            value=seq_rep,
            key_padding_mask=mask
        )
        return interests  # [Batch, K, Window_Size]


# ==========================================
# 4. MAIN MODEL (VARIANT NAML)
# ==========================================
class TIME_FEATURE_NAML(nn.Module):
    def __init__(self, config, pretrained_vectors_tensor):
        super().__init__()
        self.config = config

        # --- Init Encoders ---
        self.cand_encoder = CandidateFeatureEncoder(config, pretrained_vectors_tensor)
        self.user_encoder = MultiInterestUserEncoder(config, pretrained_vectors_tensor)
        self.imp_encoder = ImpressionEncoder(config)

        # --- Ranker Transformer ---
        # Input sequence: [Context, Interest_1...K, Cand_1...N]
        ranker_layer = nn.TransformerEncoderLayer(
            d_model=config.window_size,
            nhead=config.rankformer_heads,
            dim_feedforward=config.rankformer_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.ranker = nn.TransformerEncoder(ranker_layer, num_layers=config.rankformer_layers)

        # --- Final Prediction Head ---
        self.head = nn.Sequential(
            nn.Linear(config.window_size, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        """
        Sửa lỗi dtype và tối ưu cho torch.compile
        """
        try:
            target_dtype = next(self.ranker.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        # Cast only floating tensors to the model dtype
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=target_dtype)
        # --- 0. Chuẩn hóa Dtype ngay từ đầu ---
        # Ép tất cả các trường dữ liệu số thực về float32 để khớp với trọng số Model
        for k in ["hist_scroll", "hist_time", "hist_diff", "imp_feats", "cand_num"]:
            if k in batch and batch[k].dtype != torch.float32:
                batch[k] = batch[k].to(torch.float32)
        hist_emb = self.user_encoder.body_emb(batch["hist_indices"])  # [B, H, 1024]
        cand_emb = self.cand_encoder.body_emb(batch["cand_indices"])  # [B, C, 1024]

        # B. Tính Mean History Vector (Cẩn thận padding = 0)
        # Tạo mask: 1 nếu là bài đọc, 0 nếu là padding
        hist_mask = (batch["hist_indices"] != 0).float().unsqueeze(-1)  # [B, H, 1]

        # Tổng các vector lịch sử
        sum_hist = torch.sum(hist_emb * hist_mask, dim=1)  # [B, 1024]

        # Đếm số lượng bài (tránh chia 0 bằng cách kẹp min=1e-9)
        count_hist = torch.sum(hist_mask, dim=1).clamp(min=1e-4)  # [B, 1]

        # Mean Vector
        mean_hist = sum_hist / count_hist  # [B, 1024]

        # C. Tính Dot Product (Similarity)
        # [B, C, 1024] x [B, 1024, 1] -> [B, C, 1]
        # Dùng bmm (Batch Matrix Multiplication)
        cand_sim_gpu = torch.bmm(cand_emb, mean_hist.unsqueeze(-1))  # [B, C, 1]

        # =========================================================
        # 2. GHÉP VÀO FEATURE ĐỂ MODEL HỌC
        # =========================================================
        # batch["cand_num"] gốc có shape [B, C, 5]
        # Ghép thêm cand_sim_gpu vào cuối -> [B, C, 6]

        # Đảm bảo cùng kiểu dữ liệu
        cand_sim_gpu = cand_sim_gpu.to(dtype=batch["cand_num"].dtype)

        # Cập nhật lại batch["cand_num"]
        batch["cand_num"] = torch.cat([batch["cand_num"], cand_sim_gpu], dim=-1)
        # ==========================
        # 1. ENCODE HISTORY (USER)
        # ==========================
        hist_idx = batch["hist_indices"]
        hist_mask = (hist_idx == 0)

        user_interests = self.user_encoder(
            hist_indices=hist_idx,
            hist_scroll=batch["hist_scroll"],
            hist_time=batch["hist_time"],
            hist_diff=batch["hist_diff"],
            mask=hist_mask
        )

        # ==========================
        # 2. ENCODE CONTEXT (IMPRESSION)
        # ==========================
        # imp_encoder sẽ trả về Float32 vì batch["imp_feats"] đã được cast
        imp_ctx = self.imp_encoder(batch["imp_feats"]).unsqueeze(1)

        # ==========================
        # 3. ENCODE CANDIDATES (ITEMS)
        # ==========================
        cand_idx = batch["cand_indices"]
        cand_vecs = self.cand_encoder(
            indices=cand_idx,
            num_feats=batch["cand_num"],
            cat_feats=batch["cand_cat"]
        )

        # ==========================
        # 4. RANKING
        # ==========================
        seq = torch.cat([imp_ctx, user_interests, cand_vecs], dim=1)

        # Đảm bảo seq là float32 (đôi khi autocast làm thay đổi nó)
        seq = seq.to(torch.float32)

        batch_size = seq.size(0)
        fixed_len = 1 + self.config.num_interests

        # Chỉ định rõ device và dtype khi tạo mask mới
        fixed_mask = torch.zeros(
            (batch_size, fixed_len),
            device=seq.device,
            dtype=torch.bool
        )

        cand_pad_mask = (cand_idx == 0)
        full_mask = torch.cat([fixed_mask, cand_pad_mask], dim=1)

        # Quan trọng: Transformer trong torch.compile rất khắt khe với mask dtype
        # Đảm bảo ranker nhận seq và mask khớp với dtype của trọng số
        out = self.ranker(seq, src_key_padding_mask=full_mask)

        cand_out = out[:, fixed_len:, :]
        scores = self.head(cand_out).squeeze(-1)

        # Dùng giá trị âm cực lớn phù hợp với Float32
        scores = scores.masked_fill(cand_pad_mask, -1e4)

        return {
            "preds": scores,
            "labels": batch.get("labels", None)
        }