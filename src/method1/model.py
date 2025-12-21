import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CONFIGURATION
# ==========================================
class VariantNAMLConfig:
    def __init__(self):
        # --- Dimensions ---
        self.embedding_dim = 1024
        self.window_size = 256  # d_model

        # --- Multi-Interest Specs ---
        self.num_interests = 5  # K = 5 vector sở thích

        # --- Internal Specs ---
        self.query_vector_dim = 200
        self.dropout = 0.2
        self.num_res_blocks = 2
        self.nhead = 4  # Số head cho Transformer
        self.num_interaction_features = 2

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
    def __init__(self, config):
        super().__init__()
        self.dim = config.window_size
        self.num_interests = config.num_interests

        # --- CẢI TIẾN: GATING NETWORK ---
        # Thay vì projection thông thường, ta xây dựng mạng Gate
        # Input: 2 (Scroll, Time) -> Output: 1 (Hệ số alpha từ 0-1)
        self.gate_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Quan trọng: Ép giá trị về [0, 1]
        )

        # Vẫn giữ lại projection để cộng thêm thông tin ngữ cảnh thời gian (nếu muốn)
        # Hoặc có thể bỏ đi để tiết kiệm. Ở đây ta giữ lại để model vừa "lọc" vừa "hiểu".
        self.context_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.dim)
        )
        self.layer_norm = nn.LayerNorm(self.dim)

        # Transformer Encoder (Giữ nguyên)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=config.nhead,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Multi-Interest Module (Giữ nguyên)
        self.interest_queries = nn.Parameter(torch.randn(self.num_interests, self.dim))
        nn.init.xavier_uniform_(self.interest_queries)
        self.interest_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=config.nhead,
            batch_first=True
        )

    def forward(self, news_vecs, scrolls, times, mask=None):
        # 1. Tạo features hành vi
        # behaviors: [Batch, Seq, 2]
        behaviors = torch.stack([scrolls, times], dim=-1)

        # ---------------------------------------------------------
        # CÁCH MỚI: GATING MECHANISM (Lọc nhiễu clickbait)
        # ---------------------------------------------------------

        # Tính hệ số quan trọng (alpha) cho từng bài đã đọc
        # alpha: [Batch, Seq, 1] (Giá trị từ 0.0 đến 1.0)
        # Bài nào đọc < 5s hoặc scroll 0% -> alpha sẽ tự học về gần 0
        alpha = self.gate_net(behaviors)

        # Nhân bản alpha để khớp dimension với news_vecs
        # [B, S, 1] * [B, S, Dim] -> [B, S, Dim]
        # Phép nhân này sẽ "làm mờ" những bài có alpha thấp (click nhầm)
        news_vecs_gated = news_vecs * alpha

        # (Tùy chọn) Vẫn cộng thêm thông tin ngữ cảnh thời gian
        # để model biết "bài này đọc vào lúc nào/bao lâu" dù nó bị làm mờ
        context_emb = self.context_proj(behaviors)

        # Residual Connection
        # Input cho Transformer giờ là vector ĐÃ ĐƯỢC LỌC
        combined = self.layer_norm(news_vecs_gated + context_emb)

        # ---------------------------------------------------------

        # FIX NaN (Giữ nguyên từ bước trước)
        if mask is not None:
            is_all_masked = mask.all(dim=1)
            if is_all_masked.any():
                mask[is_all_masked, 0] = False

        # 2. Interaction Modeling (Transformer)
        seq_rep = self.transformer(combined, src_key_padding_mask=mask)

        # 3. Multi-Interest Extraction
        batch_size = seq_rep.size(0)
        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)

        user_interests, _ = self.interest_attn(
            query=queries,
            key=seq_rep,
            value=seq_rep,
            key_padding_mask=mask
        )

        return user_interests

class MultiInterestUserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.window_size
        self.num_interests = config.num_interests

        # --- CẢI TIẾN: GATING NETWORK ---
        # Input: 2 (Scroll, Time) -> Output: 1 (Hệ số alpha từ 0-1)
        self.gate_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Ép giá trị về [0, 1]
        )

        # Projection ngữ cảnh (giữ lại để model hiểu ngữ cảnh thời gian)
        self.context_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.dim)
        )
        self.layer_norm = nn.LayerNorm(self.dim)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=config.nhead,
            dim_feedforward=self.dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Multi-Interest Module
        self.interest_queries = nn.Parameter(torch.randn(self.num_interests, self.dim))
        nn.init.xavier_uniform_(self.interest_queries)
        self.interest_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=config.nhead,
            batch_first=True
        )

    def forward(self, news_vecs, scrolls, times, mask=None):
        # 1. Tạo features hành vi
        behaviors = torch.stack([scrolls, times], dim=-1)

        # --- GATING MECHANISM ---
        # Tính alpha (độ quan trọng) dựa trên hành vi
        alpha = self.gate_net(behaviors)

        # "Làm mờ" các bài clickbait/đọc lướt
        news_vecs_gated = news_vecs * alpha

        # Cộng thông tin ngữ cảnh thời gian
        context_emb = self.context_proj(behaviors)

        # Residual Connection
        combined = self.layer_norm(news_vecs_gated + context_emb)

        # FIX NaN: Đảm bảo không mask toàn bộ chuỗi
        if mask is not None:
            is_all_masked = mask.all(dim=1)
            if is_all_masked.any():
                mask[is_all_masked, 0] = False

        # 2. Interaction Modeling (Transformer)
        seq_rep = self.transformer(combined, src_key_padding_mask=mask)

        # 3. Multi-Interest Extraction
        batch_size = seq_rep.size(0)
        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)

        user_interests, _ = self.interest_attn(
            query=queries,
            key=seq_rep,
            value=seq_rep,
            key_padding_mask=mask
        )

        return user_interests

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
        # self.user_encoder = InteractionMultiInterestUserEncoder(config)
        self.user_encoder = MultiInterestUserEncoder(config)

    def forward(self, batch):
        # 1. Encode History
        hist_idx = batch['hist_indices']
        hist_vecs = self.news_encoder(hist_idx)

        hist_mask = (hist_idx == 0)

        # Fix NaN (dự phòng ở cấp này)
        if hist_mask.all(dim=1).any():
            hist_mask[hist_mask.all(dim=1), 0] = False

        user_interests = self.user_encoder(
            hist_vecs,
            batch['hist_scroll'],
            batch['hist_time'],
            mask=hist_mask
        )

        # 2. Encode Candidates
        cand_vecs = self.news_encoder(batch['cand_indices'])  # [B, C, D]

        # 3. Matching Strategy
        cand_vecs_T = cand_vecs.transpose(1, 2)  # [B, D, C]

        # Scaling Dot Product
        dim_scale = self.config.window_size ** 0.5
        scores_all = torch.matmul(user_interests, cand_vecs_T) / dim_scale

        # Max Pooling over Interests
        final_scores, _ = torch.max(scores_all, dim=1)  # [B, C]

        return final_scores
