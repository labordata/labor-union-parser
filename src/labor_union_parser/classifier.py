"""Structured multi-head classifier for labor union field prediction.

Encoder: token-charcnn (CharCNN per token, 20 tokens x 20 chars)
Transformer with RoPE, per-field classification heads and pointer heads.

Fields: union_name, desig_name (classification)
        desig_num, prefix, suffix (pointer)
        f_num (classification)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .char_cnn import CharacterCNN

MAX_TOKENS = 20
MAX_CHARS_PER_TOKEN = 20

FIELDS = ["union_name", "desig_name", "desig_num", "prefix", "suffix", "f_num"]
POINTER_FIELDS = {"desig_num", "prefix", "suffix"}


def precompute_freqs(dim, max_len, theta=10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos_freqs, sin_freqs):
    """Apply rotary position embedding. x: (batch, seq, heads, head_dim)"""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    seq_len = x.shape[1]
    cos_f = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(2)
    sin_f = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(2)
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x2 * cos_f + x1 * sin_f
    return torch.cat([out1, out2], dim=-1)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=MAX_TOKENS):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        cos_freqs, sin_freqs = precompute_freqs(self.head_dim, max_seq_len)
        self.register_buffer("cos_freqs", cos_freqs)
        self.register_buffer("sin_freqs", sin_freqs)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = apply_rope(q, self.cos_freqs, self.sin_freqs)
        k = apply_rope(k, self.cos_freqs, self.sin_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0, max_seq_len=MAX_TOKENS):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.ff(self.norm2(x))
        return x


class StructuredClassifier(nn.Module):
    def __init__(
        self,
        field_sizes,
        d_model=256,
        n_heads=4,
        n_layers=2,
        ff_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder_type = "token-charcnn"
        self.char_cnn = CharacterCNN(embed_dim=d_model, char_embed_dim=16)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, dropout, MAX_TOKENS)
                for _ in range(n_layers)
            ]
        )

        # Classification heads (non-pointer fields)
        self.heads = nn.ModuleDict(
            {
                f: nn.Linear(d_model, n_classes)
                for f, n_classes in field_sizes.items()
                if f not in POINTER_FIELDS
            }
        )

        # Pointer heads
        self.pointer_heads = nn.ModuleDict(
            {f: nn.Linear(d_model, 1) for f in POINTER_FIELDS}
        )
        self.pointer_none = nn.ParameterDict(
            {f: nn.Parameter(torch.zeros(1)) for f in POINTER_FIELDS}
        )

    def forward(self, char_ids, mask=None):
        """
        Args:
            char_ids: (B, MAX_TOKENS, MAX_CHARS_PER_TOKEN)
            mask: (B, MAX_TOKENS) bool

        Returns:
            dict of field -> logits
        """
        x = self.char_cnn(char_ids)

        for layer in self.layers:
            x = layer(x, mask)

        # Mean pool (masked) for classification heads
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).float()
            pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(
                min=1
            )
        else:
            pooled = x.mean(dim=1)

        logits = {f: head(pooled) for f, head in self.heads.items()}

        # Pointer head logits: (B, MAX_TOKENS+1)
        for f in POINTER_FIELDS:
            pos_scores = self.pointer_heads[f](x).squeeze(-1)
            if mask is not None:
                pos_scores = pos_scores.masked_fill(~mask, float("-inf"))
            none_score = self.pointer_none[f].expand(pos_scores.shape[0], 1)
            logits[f] = torch.cat([pos_scores, none_score], dim=1)

        return logits
