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
        self.window_size = 128  # d_model

        # --- Internal Specs ---
        self.query_vector_dim = 200
        self.dropout = 0.2
        self.num_res_blocks = 1

# ==========================================
# 2. UTILS & BLOCKS
# ==========================================
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
    def __init__(self, input_dim, hidden_dim, num_res_blocks=2, dropout=0.1):
        super().__init__()
        self.compress_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.deep_stack = nn.Sequential(*[
            ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])

    def forward(self, x):
        x = self.compress_layer(x)
        x = self.deep_stack(x)
        return x


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        proj = self.tanh(self.linear(x))
        scores = torch.matmul(proj, self.query)
        weights = self.softmax(scores)
        output = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out




class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        print("⚡ NewsEncoder: Initializing with Pretrained Embeddings")
        self.title_emb = nn.Embedding(1,config.embedding_dim)
        self.body_emb = nn.Embedding(1,config.embedding_dim)
        self.cat_emb = nn.Embedding(1,config.embedding_dim)

        self.title_proj = DeepProjector(
            config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout
        )
        self.body_proj = DeepProjector(
            config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout
        )

        # Category đơn giản hơn, có thể dùng ít block hơn hoặc bằng (ở đây để bằng cho đồng bộ)
        self.cat_proj = DeepProjector(
            config.embedding_dim, config.window_size, num_res_blocks=1, dropout=config.dropout
        )

        # 3. Attention để gộp 3 views (Title, Body, Category)
        self.final_attention = AdditiveAttention(config.window_size, config.query_vector_dim)
        self.config = config

    def forward(self, indices):
        # ... (Code forward giữ nguyên) ...
        """
        Input: (Batch, Num_News) - Index của bài báo
        Output: (Batch, Num_News, Hidden_Dim) - Vector đại diện bài báo
        """
        # Lookup Embeddings: (Batch, Num_News, 1024)
        t_vec = self.title_emb(indices)
        b_vec = self.body_emb(indices)
        c_vec = self.cat_emb(indices)

        # Deep Projection: (Batch, Num_News, 400)
        t_vec = self.title_proj(t_vec)
        b_vec = self.body_proj(b_vec)
        c_vec = self.cat_proj(c_vec)

        # Stack views: (Batch * Num_News, 3, 400)
        # Gom Batch và Num_News lại để tính attention một thể cho nhanh
        batch_size, num_news, _ = t_vec.shape

        # Stack dọc theo dimension mới
        stacked_views = torch.stack([t_vec, b_vec, c_vec], dim=2)

        # Flatten batch dimension: (Batch * Num_News, 3_Views, 400)
        stacked_views = stacked_views.view(-1, 3, self.config.window_size)

        # Apply Attention
        # (Batch * Num_News, 400)
        news_vectors = self.final_attention(stacked_views)

        # Reshape lại về batch ban đầu
        return news_vectors.view(batch_size, num_news, -1)

class SingleInterestUserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.self_attn = MultiHeadSelfAttention(config.window_size, num_heads=4)
        self.attention = AdditiveAttention(config.window_size, config.query_vector_dim)

    def forward(self, news_vecs):
        # x = self.self_attn(news_vecs)
        x = self.attention(news_vecs)
        return x

class VariantNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Truyền embeddings xuống NewsEncoder
        self.news_encoder = NewsEncoder(config)
        self.config = config
        # User Encoder (Giữ nguyên class bạn đang dùng)
        self.user_encoder = SingleInterestUserEncoder(config)

    def forward(self, batch):
        """
        hist_indices: (Batch, History_Len)
        cand_indices: (Batch, Num_Candidates)
        """
        # 1. Encode History -> User Vector
        # Lấy vector các bài đã đọc: (Batch, Hist_Len, 400)
        hist_vecs = self.news_encoder(batch["hist_ids"])

        # Gộp thành 1 User Vector duy nhất: (Batch, 400)
        user_vec = self.user_encoder(hist_vecs)

        # 2. Encode Candidates
        # Lấy vector các bài ứng viên: (Batch, Num_Cand, 400)
        cand_vecs = self.news_encoder(batch["candidate_ids"])

        # 3. Dot Product (Tính điểm)
        # User: (Batch, 1, 400)
        # Cand: (Batch, 400, Num_Cand) [Transpose]
        # Result: (Batch, 1, Num_Cand) -> Squeeze -> (Batch, Num_Cand)
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(2)).squeeze(2)
        return {
            "preds": scores,
            "labels": batch.get("labels", None)
        }
