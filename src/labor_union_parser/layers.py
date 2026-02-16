"""Generic transformer building blocks: rotary embeddings, self-attention, cross-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=64, base=10000):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x):
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPESelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, self.embed_dim)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class SelfAttentionLayer(nn.Module):
    """Standard transformer layer: self-attention + FFN, no positional encoding."""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
        for mod in self.ff:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, x, padding_mask=None, return_attn_weights=False):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).reshape(B, S, self.embed_dim)
        out = self.out_proj(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))

        if return_attn_weights:
            return x, attn_weights
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, attn_temperature=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_temperature = attn_temperature

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_emb,
        field_emb,
        query_mask=None,
        field_mask=None,
        return_attn_weights=False,
    ):
        B, Q, _ = query_emb.shape
        _, K, _ = (
            field_emb.shape
        )  # K = number of fields (don't shadow F = torch.nn.functional)

        # Zero out padded query tokens so they don't participate in attention.
        if query_mask is not None:
            query_emb = query_emb.masked_fill(query_mask.unsqueeze(-1), 0.0)

        q = (
            self.q_proj(query_emb)
            .view(B, Q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(field_emb)
            .view(B, K, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(field_emb)
            .view(B, K, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Build attention mask
        attn_mask = None
        if field_mask is not None:
            attn_mask = (
                field_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, Q, K)
            )
            attn_mask = torch.where(attn_mask, float("-inf"), 0.0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).reshape(B, Q, self.embed_dim)
        out = self.out_proj(out)

        out = self.norm(query_emb + out)

        # Keep padded positions zeroed to avoid leaking noise downstream.
        if query_mask is not None:
            out = out.masked_fill(query_mask.unsqueeze(-1), 0.0)

        if return_attn_weights:
            return out, attn_weights
        return out
