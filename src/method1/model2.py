import torch
import torch.nn as nn
import torch.nn.functional as F


class VariantNAMLConfig:
    def __init__(self):
        self.embedding_dim = 1024
        self.window_size = 128
        self.query_vector_dim = 200
        self.dropout = 0.2
        self.num_res_blocks = 3
        self.nhead = 8
        self.num_user_layers = 4


# --- 1. CÔNG CỤ CƠ BẢN ---

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
        res_blocks = [ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        self.deep_stack = nn.Sequential(*res_blocks)

    def forward(self, x):
        return self.deep_stack(self.compress_layer(x))


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, query_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        proj = torch.tanh(self.linear(x))
        scores = torch.matmul(proj, self.query)
        weights = self.softmax(scores)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


# --- 2. IDEA 3: ENGAGEMENT-BASED COMPONENTS ---

class EngagementEncoder(nn.Module):
    """Tính toán scalar g đại diện cho độ mạnh của bộ nhớ (Memory Strength)"""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, scrolls, times):
        feats = torch.stack([scrolls, times], dim=-1)  # [B, T, 2]
        return self.mlp(feats).squeeze(-1)  # [B, T]


class EngagementBiasedMHA(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, engagement, mask=None):
        B, T, D = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 🔥 Engagement bias: Dùng log(g) để bias vào attention scores
        # Các tin tức có engagement cao (g gần 1) sẽ ít bị trừ điểm hơn
        log_g = torch.log(engagement.clamp(min=1e-6))
        attn = attn + log_g.unsqueeze(1).unsqueeze(2)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn = F.softmax(attn, dim=-1)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, D))


class EngagementTransformerBlock(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.attn = EngagementBiasedMHA(dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, engagement, mask=None):
        x = x + self.attn(self.norm1(x), engagement, mask)
        x = x + self.ffn(self.norm2(x))
        return x


# --- 3. MAIN ENCODERS ---

class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.title_emb = nn.Embedding(1, config.embedding_dim)
        self.body_emb = nn.Embedding(1, config.embedding_dim)
        self.cat_emb = nn.Embedding(1, config.embedding_dim)

        self.title_proj = DeepProjector(config.embedding_dim, config.window_size, config.num_res_blocks)
        self.body_proj = DeepProjector(config.embedding_dim, config.window_size, config.num_res_blocks)
        self.cat_proj = DeepProjector(config.embedding_dim, config.window_size, 0)

        self.final_attention = AdditiveAttention(config.window_size, config.query_vector_dim)

    def forward(self, indices):
        t = self.title_proj(self.title_emb(indices))
        b = self.body_proj(self.body_emb(indices))
        c = self.cat_proj(self.cat_emb(indices))

        batch_size, num_news, dim = t.shape
        stacked = torch.stack([t, b, c], dim=2).view(-1, 3, dim)
        return self.final_attention(stacked).view(batch_size, num_news, dim)


class InteractionAwareUserEncoder(nn.Module):
    """Triển khai IDEA 3: Semantic Transformer điều hướng bởi Engagement"""

    def __init__(self, config):
        super().__init__()
        self.engagement_encoder = EngagementEncoder()
        self.layers = nn.ModuleList([
            EngagementTransformerBlock(config.window_size, config.nhead, config.dropout)
            for _ in range(config.num_user_layers)
        ])
        self.attn_pool = AdditiveAttention(config.window_size, config.query_vector_dim)

    def forward(self, news_vecs, scrolls, times, mask=None):
        g = self.engagement_encoder(scrolls, times)
        x = news_vecs
        for layer in self.layers:
            x = layer(x, g, mask)
        return self.attn_pool(x)


# --- 4. TOP MODEL ---

class VariantNAML(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = InteractionAwareUserEncoder(config)

    def forward(self, batch, hist_mask=None):
        # news_vecs: [B, T, D]
        hist_vecs = self.news_encoder(batch["hist_indices"])
        # user_vec: [B, D]
        user_vec = self.user_encoder(hist_vecs, batch["hist_scroll"], batch["hist_wpm_score"], mask=hist_mask)
        # cand_vecs: [B, C, D]
        cand_vecs = self.news_encoder(batch["cand_indices"])

        # Scoring
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(2)).squeeze(2)
        return {"preds":scores,"labels":batch.get("labels", None)}
