import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ==========================================
# 1. CONFIGURATION
# ==========================================
class UnifiedConfig:
    def __init__(self):
        # --- Data Specs ---
        self.embedding_dim = 1024

        # --- Model Specs ---
        self.d_model = 128
        self.query_vector_dim = 200
        self.nhead = 4
        self.dropout = 0.2
        self.num_res_blocks = 3

        # --- User Behavior Specs ---
        # [CHANGE] Đổi WPM thành Time buckets
        self.time_buckets = 20  # Chia khoảng thời gian logarit thành 20 mức
        self.scroll_buckets = 11

        # [NEW] Giá trị trần cho log1p(time).
        # log1p(400s) ≈ 6.0. Các bài đọc > 400s sẽ được coi là max.
        self.max_hist_time_log = 6.0

    # ==========================================


# 2. UTILS (ResBlock, Attention...)
# ==========================================
# ... (Giữ nguyên ResBlock, DeepProjector, AdditiveAttention, MaskedMultiHeadAttention như cũ) ...
class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class DeepProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_res_blocks=1, dropout=0.1):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_stack = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)])

    def forward(self, x):
        return self.res_stack(self.compress(x))


class SafeAdditiveAttention(nn.Module):
    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        scores = torch.matmul(self.tanh(self.linear(x)), self.query)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        weights = F.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            key_padding_mask = (mask == 0)
            is_all_masked = key_padding_mask.all(dim=1)
            if is_all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[is_all_masked, 0] = False
        else:
            key_padding_mask = None
        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        return attn_out


class MultiViewAttentionFusion(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        # 1. Feature Transformation
        # Giữ lại các lớp này để biến đổi đặc trưng trước khi fuse.
        # Đặc biệt quan trọng với v_low để model có thể "xoay" vector này
        # sang hướng có thể triệt tiêu nhiễu thay vì cộng dồn.
        self.trans_high = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.trans_low = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.trans_glo = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

        # 2. Multi-Head Attention Fusion
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Learnable Query ("User Fusion Token")
        # Vector này đóng vai trò như [CLS] token, học cách đặt câu hỏi:
        # "Tôi nên tin tưởng bao nhiêu % vào High, Low và Global?"
        self.fusion_query = nn.Parameter(torch.randn(1, 1, dim))

        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_high, v_low, v_glo):
        """
        v_high, v_low, v_glo: [Batch, Dim]
        """
        B = v_high.size(0)

        # A. Transform Input Views
        t_high = self.trans_high(v_high)  # [B, D]
        t_low = self.trans_low(v_low)  # [B, D]
        t_glo = self.trans_glo(v_glo)  # [B, D]

        # B. Stack thành Sequence (Key & Value)
        # Sequence Length = 3 (tương ứng 3 views)
        # Shape: [Batch, 3, Dim]
        kv_seq = torch.stack([t_high, t_low, t_glo], dim=1)

        # C. Prepare Query
        # Expand query cho khớp batch size: [Batch, 1, Dim]
        query = self.fusion_query.expand(B, -1, -1)

        # D. MHA Forward
        # Query: Learnable Token
        # Key/Value: 3 Views (High, Low, Glo)
        # Output: [Batch, 1, Dim]
        # Weights: [Batch, 1, 3] -> Cho biết view nào được chú ý nhiều nhất
        attn_out, attn_weights = self.mha(query, kv_seq, kv_seq)

        user_vec = attn_out.squeeze(1)  # [B, D]

        # E. Residual Connection & Norm
        # Cộng residual với v_glo (luồng thông tin ổn định nhất) để luồng gradient mượt hơn
        user_vec = user_vec + v_glo

        user_vec = self.layer_norm(user_vec)
        user_vec = self.dropout(user_vec)

        return user_vec, attn_weights

# ==========================================
# 3. NAML NEWS ENCODER
# ==========================================
# ==========================================
# 3. NAML NEWS ENCODER (UPDATED WITH MHA)
# ==========================================
class NAMLNewsEncoder(nn.Module):
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        # 1. Embeddings
        self.title_emb = nn.Embedding(20000, config.embedding_dim)
        self.body_emb = nn.Embedding(20000, config.embedding_dim)
        self.cat_emb = nn.Embedding(20000, config.embedding_dim)

        # Init weights (như cũ)
        with torch.no_grad():
            self.title_emb.weight[0].fill_(1e-4)
            self.body_emb.weight[0].fill_(1e-4)
            self.cat_emb.weight[0].fill_(1e-4)

        # 2. Projectors (như cũ)
        self.title_proj = DeepProjector(config.embedding_dim, config.d_model, config.num_res_blocks, config.dropout)
        self.body_proj = DeepProjector(config.embedding_dim, config.d_model, config.num_res_blocks, config.dropout)
        self.cat_proj = DeepProjector(config.embedding_dim, config.d_model, num_res_blocks=1, dropout=config.dropout)

        # 3. [CHANGE] Thay Additive thành MultiHeadAttention
        # Chúng ta dùng MHA để gộp 3 views (Title, Body, Cat) thành 1 vector
        self.news_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )

        # "Learnable Query": Một vector tham số hóa đóng vai trò "câu hỏi"
        # để model học cách lấy thông tin từ Title, Body, Cat.
        # Shape: [1, 1, d_model]
        self.news_query = nn.Parameter(torch.randn(1, 1, config.d_model))

        # LayerNorm đầu ra cho ổn định (Optional nhưng khuyến nghị với MHA)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, indices):
        # indices shape: [Batch, News_Count]
        valid_mask = (indices != 0).float().unsqueeze(-1)  # [B, N, 1]

        # 1. Embed & Project
        t = self.title_proj(self.title_emb(indices))
        b = self.body_proj(self.body_emb(indices))
        c = self.cat_proj(self.cat_emb(indices))

        # 2. Stack Views: [B, N, 3, D]
        stacked = torch.stack([t, b, c], dim=2)

        B, N, V, D = stacked.shape

        # Flatten Batch và News dimension để đưa vào Attention
        # Input shape cho MHA: [Batch_Size * News_Count, Seq_Len=3, Dim]
        flat_input = stacked.view(-1, V, D)
        batch_size_prime = flat_input.size(0)  # = B * N

        # 3. Prepare Query
        # Expand query cho khớp batch size: [B*N, 1, D]
        query = self.news_query.expand(batch_size_prime, -1, -1)

        # 4. Multi-Head Attention
        # Query: Vector đại diện (1 token)
        # Key/Value: 3 views (Title, Body, Cat)
        # Output: [B*N, 1, D]
        attn_out, _ = self.news_attn(query, flat_input, flat_input)

        # Squeeze sequence dim: [B*N, 1, D] -> [B*N, D]
        news_vec_flat = attn_out.squeeze(1)

        # Add & Norm (Residual connection với Query gốc - tùy chọn, ở đây mình dùng Norm output)
        news_vec_flat = self.norm(news_vec_flat)

        # 5. Reshape lại về [Batch, News_Count, Dim]
        vecs = news_vec_flat.view(B, N, D)

        # 6. Apply Mask (Những bài báo là padding (index 0) thì vector về 0)
        vecs = vecs * valid_mask

        if torch.isnan(vecs).any():
            vecs = torch.nan_to_num(vecs, nan=0.0)

        return vecs


# ==========================================
# 4. QUALITY-AWARE USER ENCODER (UPDATED)
# ==========================================
class QualityAwareUserEncoder(nn.Module):
    def __init__(self, config: UnifiedConfig, news_encoder: nn.Module):
        super().__init__()
        self.news_encoder = news_encoder
        self.dim = config.d_model

        # [CHANGE] Update tên config
        self.time_buckets = config.time_buckets
        self.scroll_buckets = config.scroll_buckets
        self.max_hist_time_log = config.max_hist_time_log

        # --- Context Embeddings ---
        # [CHANGE] WPM Emb -> Time Emb
        self.time_emb = nn.Embedding(config.time_buckets, self.dim, padding_idx=0)
        self.scroll_emb = nn.Embedding(config.scroll_buckets, self.dim, padding_idx=0)
        self.pos_emb = nn.Embedding(100, self.dim)

        self.base_attn = MaskedMultiHeadAttention(self.dim, n_heads=config.nhead)

        self.pool_high = SafeAdditiveAttention(self.dim, config.query_vector_dim)
        self.pool_low = SafeAdditiveAttention(self.dim, config.query_vector_dim)
        self.pool_global = SafeAdditiveAttention(self.dim, config.query_vector_dim)

        self.fusion_module = MultiViewAttentionFusion(dim=self.dim, n_heads=config.nhead,dropout=config.dropout)
        self.fusion_norm = nn.LayerNorm(self.dim)
        self.denoise_factor = nn.Parameter(torch.tensor(0.5))
    def bucketize(self, float_tensor, num_buckets):
        # Clamp giá trị về [0, 1] trước khi chia bucket
        val = torch.clamp(float_tensor, 0.0, 1.0)
        indices = (val * (num_buckets - 1)).long()
        return indices

    def generate_quality_masks(self, time_log_float, scroll_float):
        """
        Logic phân loại High/Low quality dựa trên Time (log1p) và Scroll.
        - High Quality: Đọc lâu (Time > X) VÀ có cuộn trang (Scroll > Y)
        - Low Quality: Đọc quá nhanh (Time < A) HOẶC không cuộn (Scroll < B)
        """
        # Ngưỡng (Thresholds)
        # log1p(20s) ~ 3.0  => Coi là đọc kỹ
        # log1p(3s)  ~ 1.3  => Coi là clickbait/lướt
        high_time_thresh = 4.0
        low_time_thresh = 1.3

        scroll_thresh = 0.2  # 20%

        mask_high = ((time_log_float > high_time_thresh) & (scroll_float > scroll_thresh)).float()

        # Low quality: Time quá thấp HOẶC Scroll quá ít (mà time không đủ cao để bù lại)
        mask_low = ((time_log_float < low_time_thresh) | (scroll_float < 0.1)).float()

        # Đảm bảo không chồng lấn: Nếu đã là High thì không là Low
        mask_low = mask_low * (1.0 - mask_high)

        return mask_high, mask_low

    def forward(self, hist_indices, hist_scroll_float, hist_time_log_float, padding_mask):
        """
        hist_time_log_float: Giá trị log1p(time), dải giá trị khoảng 0 -> 6+
        """
        B, S = hist_indices.shape

        # A. Encode News
        news_vecs = self.news_encoder(hist_indices)  # [B, S, D]

        # B. Context Embedding
        # [CHANGE] Normalize time log về [0, 1] trước khi bucketize
        norm_time = hist_time_log_float / self.max_hist_time_log
        time_ids = self.bucketize(norm_time, self.time_buckets)

        scroll_ids = self.bucketize(hist_scroll_float, self.scroll_buckets)
        pos_ids = torch.arange(S, device=hist_indices.device).expand(B, S)

        context = (self.time_emb(time_ids) +
                   self.scroll_emb(scroll_ids) +
                   self.pos_emb(pos_ids))

        context = context * padding_mask.unsqueeze(-1)
        x = news_vecs + context

        # C. Base Attention
        x = self.base_attn(x, x, x, mask=padding_mask)
        x = torch.nan_to_num(x, nan=0.0)

        # D. Disentanglement (Quality-based)
        # [CHANGE] Dùng Time Log thay cho WPM
        m_high, m_low = self.generate_quality_masks(hist_time_log_float, hist_scroll_float)

        m_high = m_high * padding_mask
        m_low = m_low * padding_mask

        # E. Pooling
        v_high = self.pool_high(x, mask=m_high)
        v_low = self.pool_low(x, mask=m_low)
        v_glo = self.pool_global(x, mask=padding_mask)

        # F. Fusion
        user_vec = v_glo + v_high - (torch.sigmoid(self.denoise_factor) * v_low)
        user_vec = self.fusion_norm(user_vec)
        # user_vec, weights = self.fusion_module(v_high, v_low, v_glo)
        return user_vec


# ==========================================
# 5. FULL MODEL WRAPPER
# ==========================================
class FullNewsRecModel(nn.Module):
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.news_encoder = NAMLNewsEncoder(config)
        self.user_encoder = QualityAwareUserEncoder(config, self.news_encoder)

    def forward(self, batch: Dict[str, torch.Tensor]):
        hist_idx = batch['hist_indices']
        hist_scroll = batch['hist_scroll']
        # [CHANGE] Lấy key log1p thay vì wpm score
        hist_time_log = batch['hist_time_log1p']
        cand_idx = batch['cand_indices']

        padding_mask = (hist_idx != 0).float()

        # 1. Get User Vector
        user_vec = self.user_encoder(
            hist_indices=hist_idx,
            hist_scroll_float=hist_scroll,
            hist_time_log_float=hist_time_log,  # [CHANGE] Pass time log
            padding_mask=padding_mask
        )

        # 2. Get Candidate Vectors
        cand_vecs = self.news_encoder(cand_idx)

        # 3. Scoring
        raw_scores = torch.bmm(user_vec.unsqueeze(1), cand_vecs.transpose(1, 2)).squeeze(1)

        d_model = user_vec.shape[-1]
        scaled_scores = raw_scores / (d_model ** 0.5)
        # final_scores = torch.clamp(scaled_scores, min=-10.0, max=10.0)

        return {
            "preds": scaled_scores,
            "labels": batch.get("labels", None)
        }


# ==========================================
# MAIN TESTING (Updated for Time Log)
# ==========================================
if __name__ == "__main__":
    import os
    import numpy as np

    # CONFIG
    REAL_EMB_DIR = "/home2/congnh/wm/embedding"
    HISTORY_LEN = 10

    print("=" * 60)
    print("🚀 TESTING NEWS RECOMMENDER (Time Log Version)")
    print("=" * 60)

    # 1. SETUP
    if not os.path.exists(REAL_EMB_DIR):
        print(f"❌ Error: Folder not found at {REAL_EMB_DIR}. Using random weights for test.")
        real_emb_dim = 1024  # Fake dim
        config = UnifiedConfig()
        config.embedding_dim = real_emb_dim
        model = FullNewsRecModel(config)
    else:
        # Load Real Logic (Giữ nguyên logic load NPY của bạn)
        test_load = np.load(f"{REAL_EMB_DIR}/title_emb.npy", mmap_mode='r')
        real_vocab_size, real_emb_dim = test_load.shape
        config = UnifiedConfig()
        config.embedding_dim = real_emb_dim
        model = FullNewsRecModel(config)

        print("\n💉 Injecting Real Embeddings...")
        # ... (Code inject giữ nguyên) ...

    model.eval()

    # =========================================================
    # CASE: ACTIVE USER with TIME LOG data
    # =========================================================
    print("\n" + "-" * 60)
    print("👤 CASE: ACTIVE USER (With Time Log1p)")
    print("-" * 60)

    # Giả lập input
    cand_tensor = torch.tensor([[10, 100, 500, 1000]], dtype=torch.long)

    # Random Log Time: Giá trị từ 0 đến 6.0
    # Ví dụ: 0.0 (0s), 2.3 (10s), 4.6 (100s), ...
    random_time_log = torch.rand((1, HISTORY_LEN)) * 6.0

    batch_active = {
        'hist_indices': torch.randint(1, 1000, (1, HISTORY_LEN), dtype=torch.long),
        'hist_scroll': torch.rand((1, HISTORY_LEN), dtype=torch.float),
        'hist_time_log1p': random_time_log,  # [CHANGE] Key mới
        'cand_indices': cand_tensor
    }

    print(f"🕒 Sample Time Logs: {batch_active['hist_time_log1p'][0, -5:].tolist()}")

    with torch.no_grad():
        out = model(batch_active)
        scores = out['preds'].view(-1).tolist()

    print("\n📊 Scores:")
    for sc in scores:
        print(f"  Score: {sc:.4f}")