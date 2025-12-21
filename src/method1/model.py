import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CONFIGURATION
# ==========================================
class VariantNAMLConfig:
    def __init__(self):
        # --- Dimensions ---
        self.embedding_dim = 768
        self.window_size = 256  # d_model

        # --- Multi-Interest Specs ---
        self.num_interests = 5  # K = 5 vector sở thích

        # --- Internal Specs ---
        self.query_vector_dim = 200
        self.dropout = 0.2
        self.num_res_blocks = 2
        self.nhead = 4  # Số head cho Transformer


# ==========================================
# 2. UTILS & BLOCKS
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x): return x + self.block(x)


class DeepProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_res_blocks=2, dropout=0.1):
        super().__init__()
        self.compress_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.deep_stack = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)])

    def forward(self, x): return self.deep_stack(self.compress_layer(x))


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        proj = self.tanh(self.linear(x))
        scores = torch.matmul(proj, self.query)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.softmax(scores)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


# ==========================================
# 3. ENCODERS
# ==========================================

class InteractionMultiInterestUserEncoder(nn.Module):
    """
    Sự kết hợp hoàn hảo:
    1. Interaction Aware: Dùng Transformer + Scroll/Time để hiểu ngữ cảnh đọc.
    2. Multi-Interest: Dùng K Query Seeds để trích xuất đa sở thích từ ngữ cảnh đó.
    """

    def __init__(self, config):
        super().__init__()
        self.dim = config.window_size
        self.num_interests = config.num_interests

        # --- A. INTERACTION MODULE ---
        # Project scroll/time về cùng không gian vector
        self.inter_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.dim)
        )
        self.layer_norm = nn.LayerNorm(self.dim)

        # Transformer Encoder: Học sự phụ thuộc giữa các bài đã đọc
        # (Bài đọc lâu ảnh hưởng thế nào đến bài đọc nhanh?)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=config.nhead,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # --- B. MULTI-INTEREST MODULE ---
        # K vector hạt giống (Learnable Queries)
        self.interest_queries = nn.Parameter(torch.randn(self.num_interests, self.dim))
        nn.init.xavier_uniform_(self.interest_queries)

        # Attention cơ chế Query-Key-Value
        # Query: Interest Seeds
        # Key/Value: Output của Transformer (Interaction Context)
        self.interest_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=config.nhead,
            batch_first=True
        )

    def forward(self, news_vecs, scrolls, times, mask=None):
        # 1. Nhúng thông tin hành vi (Fusion)
        inter_feats = torch.stack([scrolls, times], dim=-1)  # [B, S, 2]
        inter_emb = self.inter_proj(inter_feats)  # [B, S, Dim]

        # Cộng residual: Content + Behavior
        combined = self.layer_norm(news_vecs + inter_emb)

        # 2. Interaction Modeling (Transformer)
        # seq_rep: [Batch, Seq, Dim] - Đây là lịch sử đã được "hiểu" theo ngữ cảnh
        seq_rep = self.transformer(combined, src_key_padding_mask=mask)

        # 3. Multi-Interest Extraction
        batch_size = seq_rep.size(0)

        # Expand Queries cho batch: [B, K, Dim]
        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Queries tìm kiếm thông tin trong seq_rep
        # Masking: Queries không được nhìn vào padding của history
        user_interests, _ = self.interest_attn(
            query=queries,
            key=seq_rep,
            value=seq_rep,
            key_padding_mask=mask
        )

        return user_interests  # [Batch, K, Dim]


class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.title_emb = nn.Embedding(1, config.embedding_dim)
        self.body_emb = nn.Embedding(1, config.embedding_dim)
        self.cat_emb = nn.Embedding(1, config.embedding_dim)

        self.title_proj = DeepProjector(config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout)
        self.body_proj = DeepProjector(config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout)
        self.cat_proj = DeepProjector(config.embedding_dim, config.window_size, num_res_blocks=0,
                                      dropout=config.dropout)
        self.final_attention = AdditiveAttention(config.window_size, config.query_vector_dim)

    def forward(self, indices):
        t_vec = self.title_proj(self.title_emb(indices))
        b_vec = self.body_proj(self.body_emb(indices))
        c_vec = self.cat_proj(self.cat_emb(indices))

        batch_size, num_news, dim = t_vec.shape
        stacked = torch.stack([t_vec, b_vec, c_vec], dim=2).view(-1, 3, dim)
        return self.final_attention(stacked).view(batch_size, num_news, dim)


# ==========================================
# 4. MAIN MODEL
# ==========================================
class VariantNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config)

        # ==> Dùng Class mới kết hợp cả 2 cơ chế
        self.user_encoder = InteractionMultiInterestUserEncoder(config)

    def forward(self, batch):
        # 1. Encode History
        hist_idx = batch['hist_indices']
        hist_vecs = self.news_encoder(hist_idx)
        hist_mask = (hist_idx == 0)

        # Output: [Batch, K, Dim] (K vector sở thích đã thấm nhuần hành vi scroll/time)
        user_interests = self.user_encoder(
            hist_vecs,
            batch['hist_scroll'],
            batch['hist_time'],
            mask=hist_mask
        )

        # 2. Encode Candidates
        cand_vecs = self.news_encoder(batch['cand_indices'])  # [B, C, D]

        # 3. Matching Strategy: Max(DotProduct)
        # Transpose candidate để nhân ma trận
        cand_vecs_T = cand_vecs.transpose(1, 2)  # [B, D, C]

        # Tính độ khớp của TẤT CẢ interests với candidates
        # [B, K, D] x [B, D, C] = [B, K, C]
        scores_all = torch.matmul(user_interests, cand_vecs_T)

        # Lấy Max qua trục K (User click vì 1 lý do/sở thích lớn nhất)
        final_scores, _ = torch.max(scores_all, dim=1)  # [B, C]

        return final_scores
