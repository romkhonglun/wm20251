import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg:
        d_h: the last dimension of input
    '''

    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class NewsEncoder(nn.Module):
    def __init__(self, hparams) -> None:
        super(NewsEncoder, self).__init__()
        self.hparams = hparams
        self.mha = nn.MultiheadAttention(hparams['embed_size'], num_heads=hparams['nhead'], dropout=0.1)
        self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size'])
        # self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size'])

    def forward(self, news_embed):
        '''
        Args:
            news_embed (tensor): Clicked/candidate news title and abstract [num_user, num_click_docs, seq_len].

        Returns:
            output: (tensor): Clicked/candidate news representation.
        '''
        news_embed = F.dropout(news_embed, 0.2)
        news_embed = news_embed.permute(1, 0, 2)
        output, _ = self.mha(news_embed, news_embed, news_embed)
        output = F.dropout(output.permute(1, 0, 2))
        output = self.proj(output)
        # output = self.additive_attn(output)
        return output


class UserEncoder(nn.Module):
    """
    Encodes user behavior based on clicked documents using multi-head attention and additive attention.
    """

    def __init__(self, encoder_size, nhead, dropout, v_size):
        super(UserEncoder, self).__init__()
        self.mha = nn.MultiheadAttention(encoder_size, nhead, dropout=dropout)
        self.projection = nn.Linear(encoder_size, encoder_size)
        self.additive_attention = AdditiveAttention(encoder_size, v_size)

    def forward(self, click_embeddings):
        """
        Encodes user interactions from clicked document embeddings.

        Args:
            click_embeddings (tensor): [num_click_docs, num_user, encoder_size]

        Returns:
            tensor: User-level representations [num_user, encoder_size]
        """
        # Multi-head attention over clicked document embeddings
        attention_output, _ = self.mha(click_embeddings, click_embeddings, click_embeddings)
        attention_output = F.dropout(attention_output.permute(1, 0, 2), 0.2)

        # Project and apply additive attention
        projected_output = self.projection(attention_output)
        user_representation = self.additive_attention(projected_output)
        return user_representation


class PLMNR(nn.Module):
    """
    Neural News Recommendation Model (NRMS).
    """

    def __init__(self, hparams, device, weight=None):
        super(PLMNR, self).__init__()
        self.hparams = hparams
        self.device = device

        # Document encoder to process individual documents
        self.news_encoder = NewsEncoder(hparams)
        self.news_encoder.to(device)
        # User encoder to aggregate clicked document information
        self.user_encoder = UserEncoder(
            encoder_size=hparams['encoder_size'],
            nhead=hparams['nhead'],
            dropout=0.1,
            v_size=hparams['v_size']
        )
        self.user_encoder.to(device)

        # Cross-entropy loss for training
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, browsed_texts, candidate_texts, labels=None):
        """
        Forward pass for NRMS.

        Args:
            browsed_news_ids (tensor): Clicked news title and abstract [num_user, num_click_docs, seq_len].
            candidate_news_ids (tensor): Candidate news title and abstract [num_user, num_candidate_docs,seq_len].
            labels (tensor, optional): Ground-truth labels for the candidates.

        Returns:
            If labels are provided: tuple (loss, score).
            Otherwise: tensor (probabilities for each candidate document).
        """
        num_user, num_click_docs, embed_dim = browsed_texts.shape
        num_cand_docs = candidate_texts.shape[1]

        # Encode documents
        click_embeddings = self.news_encoder(browsed_texts)
        cand_embeddings = self.news_encoder(candidate_texts)

        # User representation from clicked documents
        click_embeddings = click_embeddings.permute(1, 0, 2)  # [num_click_docs, num_user, encoder_size]
        user_representations = self.user_encoder(click_embeddings)  # [num_user, user_dim]

        # Compute logits for candidate ranking
        score = torch.bmm(user_representations.unsqueeze(1), cand_embeddings.permute(0, 2, 1)).squeeze(1)

        if labels is not None:
            loss = self.criterion(score, labels.to(self.device))
            return loss, score
        return score