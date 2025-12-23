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
        self.window_size = 128  # d_model / input_dim cho attention

        # --- Internal Specs ---
        self.query_vector_dim = 256
        self.dropout = 0.2
        self.num_res_blocks = 3

        # [NEW] Thêm số đầu (Heads) cho Multi-Head Attention
        self.num_heads = 4

    # ==========================================


# 2. UTILS & BLOCKS (Giữ nguyên)
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

    def forward(self, x):
        return x + self.block(x)


class DeepProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_res_blocks=2, dropout=0.1):
        super().__init__()
        self.compress_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
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
        # batch_first=True để input/output là (Batch, Seq, Dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        # Self-attention: Query=Key=Value=x
        attn_output, _ = self.mha(x, x, x)
        # Residual connection + LayerNorm (Chuẩn Transformer)
        return self.layer_norm(x + attn_output)


# ==========================================
# 3. MODIFIED NEWS ENCODER
# ==========================================
class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        print("⚡ NewsEncoder: Initializing with Multi-Head Attention Interaction")
        self.title_emb = nn.Embedding(1, config.embedding_dim)
        self.body_emb = nn.Embedding(1, config.embedding_dim)
        self.cat_emb = nn.Embedding(1, config.embedding_dim)

        self.title_proj = DeepProjector(
            config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout
        )
        self.body_proj = DeepProjector(
            config.embedding_dim, config.window_size, config.num_res_blocks, config.dropout
        )
        self.cat_proj = DeepProjector(
            config.embedding_dim, config.window_size, num_res_blocks=1, dropout=config.dropout
        )

        # --- [CHANGE START] ---
        # 1. Multi-Head Self-Attention: Cho phép Title, Body, Category "nhìn thấy" nhau
        self.view_interaction = MultiHeadSelfAttention(config.window_size, config.num_heads)

        # 2. Pooling: Sau khi MHA, output vẫn là (Batch, 3, Dim).
        # Ta vẫn cần Additive Attention để chọn lọc và gộp thành (Batch, Dim).
        self.final_pooling = AdditiveAttention(config.window_size, config.query_vector_dim)
        # --- [CHANGE END] ---

        self.config = config

    def forward(self, indices):
        """
        Input: (Batch, Num_News)
        Output: (Batch, Num_News, Hidden_Dim)
        """
        # Lookup Embeddings: (Batch, Num_News, 1024)
        t_vec = self.title_emb(indices)
        b_vec = self.body_emb(indices)
        c_vec = self.cat_emb(indices)

        # Project: (Batch, Num_News, 128)
        t_vec = self.title_proj(t_vec)
        b_vec = self.body_proj(b_vec)
        c_vec = self.cat_proj(c_vec)

        # Gom Batch và Num_News để xử lý song song
        batch_size, num_news, _ = t_vec.shape

        # Stack views: (Batch * Num_News, 3, 128)
        # Sequence Length = 3 (Title, Body, Category)
        stacked_views = torch.stack([t_vec, b_vec, c_vec], dim=2)
        stacked_views = stacked_views.view(-1, 3, self.config.window_size)

        # --- [NEW LOGIC] ---
        # 1. Apply Multi-Head Attention
        # Output: (Batch * Num_News, 3, 128)
        contextualized_views = self.view_interaction(stacked_views)

        # 2. Apply Additive Attention Pooling
        # Output: (Batch * Num_News, 128)
        news_vectors = self.final_pooling(contextualized_views)

        # Reshape lại về batch ban đầu
        return news_vectors.view(batch_size, num_news, -1)


# ==========================================
# 4. REMAINING MODEL
# ==========================================
class SingleInterestUserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AdditiveAttention(config.window_size, config.query_vector_dim)

    def forward(self, news_vecs):
        x = self.attention(news_vecs)
        return x


class VariantNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_encoder = NewsEncoder(config)
        self.config = config
        self.user_encoder = SingleInterestUserEncoder(config)

    def forward(self, batch):
        hist_vecs = self.news_encoder(batch["hist_ids"])
        user_vec = self.user_encoder(hist_vecs)
        cand_vecs = self.news_encoder(batch["candidate_ids"])

        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(2)).squeeze(2)
        return {
            "preds": scores,
            "labels": batch.get("labels", None)
        }