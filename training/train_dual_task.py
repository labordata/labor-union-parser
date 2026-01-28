"""
Dual-Task Training: Retrieval (Dual Tower) + Re-ranking (Cross-Attention Bridge)

This combines two complementary objectives:
1. Contrastive loss (dual tower): Learn globally discriminative embeddings for retrieval
2. Classification loss (cross-attention): Learn fine-grained matching for re-ranking

Shared components:
- CharCNN for text encoding
- Frozen number embeddings
- Record field embeddings (union, desig, prefix, suffix, magnitude)
- Query self-attention layers

Task-specific components:
- Dual tower: attention pooling, projection, low-rank f_num embeddings
- Cross-attention: cross-attention layers, unit_id embedding, classifier
"""

import json
import random
import sys
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.char_cnn import NUMBER_VOCAB, CharacterCNN, tokenize_to_chars

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"

# Shared dimensions
EMBED_DIM = 64
RETRIEVAL_DIM = 128  # Output dim for dual tower
MAX_QUERY_LEN = 20  # Non-space tokens only


def smart_truncate_nonspace(text, max_nonspace_tokens):
    """
    Tokenize text, keep first N non-space tokens, and recover lost numbers.

    This combines two strategies:
    1. Remove space tokens to reduce sequence length
    2. Replace trailing non-numeric tokens with numbers that were truncated

    Args:
        text: Input text to tokenize
        max_nonspace_tokens: Maximum number of non-space tokens to keep

    Returns:
        Same format as tokenize_to_chars: (chars, tokens, is_num, token_types, num_ids)
    """
    # Tokenize full text
    full_chars, full_tokens, full_is_num, full_token_types, full_num_ids = (
        tokenize_to_chars(text, 999)
    )

    # Extract all non-space tokens
    nonspace_data = []
    for i, tt in enumerate(full_token_types):
        if full_tokens[i] and tt != 2:  # token_type 2 = space
            nonspace_data.append(
                {
                    "chars": full_chars[i],
                    "token": full_tokens[i],
                    "is_num": full_is_num[i],
                    "token_type": tt,
                    "num_id": full_num_ids[i],
                }
            )

    # Take first N non-space tokens
    trunc_data = nonspace_data[:max_nonspace_tokens]

    # Find lost numbers and recover them
    trunc_numbers = {d["token"] for d in trunc_data if d["is_num"]}
    all_numbers = [d for d in nonspace_data if d["is_num"]]
    lost_numbers = [d for d in all_numbers if d["token"] not in trunc_numbers]

    if lost_numbers:
        # Replace trailing non-numeric tokens with lost numbers
        replace_positions = []
        for i in range(len(trunc_data) - 1, -1, -1):
            if not trunc_data[i]["is_num"] and trunc_data[i]["token"]:
                replace_positions.append(i)
                if len(replace_positions) >= len(lost_numbers):
                    break

        replace_positions.reverse()

        for pos, lost_num_data in zip(replace_positions, lost_numbers):
            trunc_data[pos] = lost_num_data

    # Pad to max_nonspace_tokens
    while len(trunc_data) < max_nonspace_tokens:
        trunc_data.append(
            {
                "chars": [0] * 20,
                "token": "",
                "is_num": 0,
                "token_type": 4,  # padding
                "num_id": 0,
            }
        )

    # Build output arrays (same format as tokenize_to_chars)
    trunc_chars = [d["chars"] for d in trunc_data]
    trunc_tokens = [d["token"] for d in trunc_data]
    trunc_is_num = [d["is_num"] for d in trunc_data]
    trunc_token_types = [d["token_type"] for d in trunc_data]
    trunc_num_ids = [d["num_id"] for d in trunc_data]

    return trunc_chars, trunc_tokens, trunc_is_num, trunc_token_types, trunc_num_ids


def normalize_designation(s: str) -> str:
    if not s:
        return s
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


# =============================================================================
# Shared Encoder Components
# =============================================================================


class SharedQueryEncoder(nn.Module):
    """
    Shared query encoder used by both tasks.

    Produces token-level embeddings that can be:
    - Pooled for dual tower (retrieval)
    - Used directly for cross-attention (re-ranking)
    """

    def __init__(self, char_cnn, embed_dim=EMBED_DIM):
        super().__init__()
        self.char_cnn = char_cnn
        self.embed_dim = embed_dim

        # Shared frozen number embedding
        self.frozen_num_embed = nn.Embedding(len(NUMBER_VOCAB), 32)
        self.frozen_num_embed.weight.requires_grad = False

        # Project to embed_dim (char_cnn + frozen_num, no is_num feature)
        input_dim = char_cnn.embed_dim + 32
        self.proj = nn.Linear(input_dim, embed_dim)

        # Self-attention with RoPE (shared between tasks)
        self.self_attn1 = RoPESelfAttentionLayer(
            embed_dim, num_heads=4, max_seq_len=MAX_QUERY_LEN
        )
        self.self_attn2 = RoPESelfAttentionLayer(
            embed_dim, num_heads=4, max_seq_len=MAX_QUERY_LEN
        )

    def forward(self, char_ids, is_number, numeric_ids):
        """
        Returns token-level embeddings and padding mask.

        Args:
            char_ids: [batch, seq_len, char_len]
            is_number: [batch, seq_len]
            numeric_ids: [batch, seq_len]

        Returns:
            token_emb: [batch, seq_len, embed_dim]
            padding_mask: [batch, seq_len] - True for padded positions
        """
        # CharCNN embeddings (zeroed for numbers)
        char_emb = self.char_cnn(char_ids)
        is_num_mask = is_number.unsqueeze(-1).float()
        char_emb = char_emb * (1 - is_num_mask)

        # Frozen number embeddings (zeroed for non-numbers)
        frozen_emb = self.frozen_num_embed(numeric_ids) * is_num_mask

        # Combine and project (no explicit is_num feature - implicit in which embedding is active)
        combined = torch.cat([char_emb, frozen_emb], dim=-1)
        token_emb = self.proj(combined)

        # Padding mask
        padding_mask = char_ids.sum(-1) == 0

        # Self-attention
        token_emb = self.self_attn1(token_emb, padding_mask)
        token_emb = self.self_attn2(token_emb, padding_mask)

        return token_emb, padding_mask


class SharedRecordEncoder(nn.Module):
    """
    Shared record field embeddings used by both tasks.

    Encodes 6 structured fields: union_name, desig_name, prefix, desig_num, suffix, unit_id
    """

    def __init__(
        self,
        num_union_names,
        num_desig_names,
        num_prefixes,
        num_suffixes,
        num_unit_ids,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Field embeddings (matching original dimensions)
        self.union_embed = nn.Embedding(num_union_names, 64)
        self.desig_embed = nn.Embedding(num_desig_names + 1, 16)
        self.prefix_embed = nn.Embedding(num_prefixes, 16, padding_idx=0)
        self.suffix_embed = nn.Embedding(num_suffixes, 16, padding_idx=0)
        self.unit_id_embed = nn.Embedding(num_unit_ids, embed_dim)

        # Number: frozen embedding only (no magnitude - numbers are IDs, not quantities)
        self.frozen_num_embed = nn.Embedding(len(NUMBER_VOCAB), 32)
        self.frozen_num_embed.weight.requires_grad = False

        # Projections to embed_dim
        self.union_proj = nn.Linear(64, embed_dim)
        self.desig_proj = nn.Linear(16, embed_dim)
        self.prefix_proj = nn.Linear(16, embed_dim)
        self.suffix_proj = nn.Linear(16, embed_dim)
        self.num_proj = nn.Linear(32, embed_dim)

        # Field type embeddings: union=0, desig=1, prefix=2, number=3, suffix=4, unit_id=5
        self.field_type_embed = nn.Embedding(6, embed_dim)

    def forward(
        self,
        union_idx,
        desig_idx,
        prefix_idx,
        num_hash,
        num_val,
        suffix_idx,
        unit_id_idx,
    ):
        """
        Returns field embeddings with field type information.

        Returns:
            field_emb: [batch, 6, embed_dim] - (union, desig, prefix, number, suffix, unit_id)
            field_mask: [batch, 6] - True for empty prefix/suffix
        """
        batch_size = union_idx.shape[0]
        device = union_idx.device

        # Embed each field
        u_e = self.union_proj(self.union_embed(union_idx))
        d_e = self.desig_proj(self.desig_embed(desig_idx))
        p_e = self.prefix_proj(self.prefix_embed(prefix_idx))
        s_e = self.suffix_proj(self.suffix_embed(suffix_idx))
        uid_e = self.unit_id_embed(unit_id_idx)

        # Number: frozen embedding only (treated as UUID)
        n_e = self.num_proj(self.frozen_num_embed(num_hash))

        # Stack: [batch, 6, embed_dim]
        field_emb = torch.stack([u_e, d_e, p_e, n_e, s_e, uid_e], dim=1)

        # Add field type embeddings
        field_type_ids = (
            torch.arange(6, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        field_emb = field_emb + self.field_type_embed(field_type_ids)

        # Mask empty prefix/suffix
        field_mask = torch.zeros(batch_size, 6, device=device, dtype=torch.bool)
        field_mask[:, 2] = prefix_idx == 0
        field_mask[:, 4] = suffix_idx == 0

        return field_emb, field_mask


# =============================================================================
# Task-Specific Heads
# =============================================================================


class DualTowerHead(nn.Module):
    """
    Dual tower head for retrieval.

    Takes shared query embeddings and produces a pooled embedding.
    Uses shared record encoder (including unit_id).
    """

    def __init__(
        self, shared_record_encoder, embed_dim=EMBED_DIM, output_dim=RETRIEVAL_DIM
    ):
        super().__init__()
        self.shared_record = shared_record_encoder

        # Attention pooling for query
        self.attn_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Query projection to output_dim
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Record projection (from shared fields to output_dim)
        # Input: 6 fields * embed_dim, flattened and projected
        self.record_proj = nn.Sequential(
            nn.Linear(6 * embed_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def encode_query(self, token_emb, padding_mask):
        """Pool token embeddings to single vector."""
        batch_size = token_emb.shape[0]
        query = self.attn_query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(
            query, token_emb, token_emb, key_padding_mask=padding_mask
        )
        pooled = pooled.squeeze(1)
        emb = self.query_proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    def encode_record(self, field_emb):
        """Encode record to single vector from pre-computed field embeddings."""
        # Flatten and project
        flat = field_emb.view(field_emb.shape[0], -1)
        emb = self.record_proj(flat)
        return F.normalize(emb, p=2, dim=-1)


class CrossAttentionHead(nn.Module):
    """
    Cross-attention head for re-ranking.

    Takes shared query token embeddings and applies cross-attention to
    shared record field embeddings (which include unit_id and field types).
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-attention layers
        self.cross_attn1 = CrossAttentionLayer(embed_dim, num_heads=4)
        self.cross_attn2 = CrossAttentionLayer(embed_dim, num_heads=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

    def score_pair(self, token_emb, token_mask, field_emb, field_mask):
        """
        Score a single query-record pair.

        Args:
            token_emb: [batch, seq_len, embed_dim]
            token_mask: [batch, seq_len]
            field_emb: [batch, 6, embed_dim]
            field_mask: [batch, 6]

        Returns:
            scores: [batch]
        """
        # Cross-attention
        enhanced = self.cross_attn1(token_emb, field_emb, token_mask, field_mask)
        enhanced = self.cross_attn2(enhanced, field_emb, token_mask, field_mask)

        # Pool and classify
        mask_expanded = (~token_mask).unsqueeze(-1).float()
        pooled = (enhanced * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)

        return self.classifier(pooled).squeeze(-1)

    def score_all_pairs(self, token_emb, token_mask, field_emb, field_mask):
        """
        Score all N×N query-record pairs using in-batch negatives.

        Args:
            token_emb: [N, seq_len, embed_dim] - N query token embeddings
            token_mask: [N, seq_len] - N query masks
            field_emb: [N, 6, embed_dim] - N record field embeddings
            field_mask: [N, 6] - N record masks

        Returns:
            scores: [N, N] - score[i,j] = score(query_i, record_j)
        """
        N = token_emb.shape[0]
        seq_len = token_emb.shape[1]

        # Expand queries: [N, N, seq_len, embed_dim]
        # Each query repeated N times (once for each record)
        q_exp = token_emb.unsqueeze(1).expand(N, N, seq_len, self.embed_dim)
        q_mask_exp = token_mask.unsqueeze(1).expand(N, N, seq_len)

        # Expand records: [N, N, 6, embed_dim]
        # Each record repeated N times (once for each query)
        r_exp = field_emb.unsqueeze(0).expand(N, N, 6, self.embed_dim)
        r_mask_exp = field_mask.unsqueeze(0).expand(N, N, 6)

        # Flatten to [N*N, ...] for batch processing
        q_flat = q_exp.reshape(N * N, seq_len, self.embed_dim)
        q_mask_flat = q_mask_exp.reshape(N * N, seq_len)
        r_flat = r_exp.reshape(N * N, 6, self.embed_dim)
        r_mask_flat = r_mask_exp.reshape(N * N, 6)

        # Score all pairs
        scores_flat = self.score_pair(q_flat, q_mask_flat, r_flat, r_mask_flat)

        # Reshape to [N, N]
        return scores_flat.view(N, N)


# =============================================================================
# Attention Layers (from cross-attention bridge)
# =============================================================================


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

    def forward(self, query_emb, field_emb, query_mask=None, field_mask=None):
        B, Q, _ = query_emb.shape
        _, K, _ = (
            field_emb.shape
        )  # K = number of fields (don't shadow F = torch.nn.functional)

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

        # Build attention mask for SDPA
        attn_mask = None
        if field_mask is not None:
            # field_mask: [B, K] where True = masked
            # SDPA expects [B, num_heads, Q, K] with -inf for masked positions
            attn_mask = (
                field_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, Q, K)
            )
            attn_mask = torch.where(attn_mask, float("-inf"), 0.0)

        # Use SDPA for numerical stability
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
        )

        out = out.transpose(1, 2).reshape(B, Q, self.embed_dim)
        out = self.out_proj(out)

        return self.norm(query_emb + out)


# =============================================================================
# Combined Model
# =============================================================================


class DualTaskModel(nn.Module):
    """
    Combined model for dual-task training.

    Shared:
    - CharCNN
    - Query encoder (self-attention)
    - Record field embeddings (including unit_id and field types)

    Task-specific:
    - Dual tower head (attention pooling + projection for retrieval)
    - Cross-attention head (cross-attention + classifier for re-ranking)
    """

    def __init__(
        self,
        num_union_names,
        num_desig_names,
        num_prefixes,
        num_suffixes,
        num_unit_ids,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        # Shared components
        self.char_cnn = CharacterCNN(embed_dim=embed_dim, char_embed_dim=16)
        self.query_encoder = SharedQueryEncoder(self.char_cnn, embed_dim)
        self.record_encoder = SharedRecordEncoder(
            num_union_names,
            num_desig_names,
            num_prefixes,
            num_suffixes,
            num_unit_ids,
            embed_dim,
        )

        # Sync frozen number embeddings
        self.record_encoder.frozen_num_embed = self.query_encoder.frozen_num_embed

        # Task-specific heads
        self.dual_tower = DualTowerHead(self.record_encoder, embed_dim)
        self.cross_attention = CrossAttentionHead(embed_dim)

    def forward_dual_task(
        self,
        q_char_ids,
        q_is_number,
        q_numeric_ids,
        r_union_idx,
        r_desig_idx,
        r_prefix_idx,
        r_num_hash,
        r_num_val,
        r_suffix_idx,
        r_unit_id_idx,
    ):
        """
        Forward pass for both tasks using in-batch negatives.

        Args:
            q_*: Query tensors [N, ...]
            r_*: Record tensors [N, ...]

        Returns:
            retrieval_sim: [N, N] - similarity matrix for retrieval
            rerank_scores: [N, N] - score matrix for re-ranking
        """
        # Shared query encoding (done once, used by both tasks)
        token_emb, padding_mask = self.query_encoder(
            q_char_ids, q_is_number, q_numeric_ids
        )

        # Shared record encoding (done once, used by both tasks)
        field_emb, field_mask = self.record_encoder(
            r_union_idx,
            r_desig_idx,
            r_prefix_idx,
            r_num_hash,
            r_num_val,
            r_suffix_idx,
            r_unit_id_idx,
        )

        # === Retrieval task ===
        # Pool query embeddings
        query_emb = self.dual_tower.encode_query(token_emb, padding_mask)

        # Encode records for retrieval (flatten fields to single vector)
        record_emb = self.dual_tower.encode_record(field_emb)

        # Similarity matrix: [N, N]
        retrieval_sim = torch.matmul(query_emb, record_emb.T)

        # === Re-ranking task ===
        # Score all N×N pairs using cross-attention
        rerank_scores = self.cross_attention.score_all_pairs(
            token_emb, padding_mask, field_emb, field_mask
        )

        return retrieval_sim, rerank_scores

    def forward_retrieval(
        self,
        q_char_ids,
        q_is_number,
        q_numeric_ids,
        r_union_idx,
        r_desig_idx,
        r_prefix_idx,
        r_num_hash,
        r_num_val,
        r_suffix_idx,
        r_unit_id_idx,
    ):
        """
        Forward pass for retrieval task only (for evaluation).
        """
        token_emb, padding_mask = self.query_encoder(
            q_char_ids, q_is_number, q_numeric_ids
        )
        field_emb, _ = self.record_encoder(
            r_union_idx,
            r_desig_idx,
            r_prefix_idx,
            r_num_hash,
            r_num_val,
            r_suffix_idx,
            r_unit_id_idx,
        )
        query_emb = self.dual_tower.encode_query(token_emb, padding_mask)
        record_emb = self.dual_tower.encode_record(field_emb)
        return query_emb, record_emb

    def forward_reranking(
        self,
        q_char_ids,
        q_is_number,
        q_numeric_ids,
        r_union_idx,
        r_desig_idx,
        r_prefix_idx,
        r_num_hash,
        r_num_val,
        r_suffix_idx,
        r_unit_id_idx,
    ):
        """
        Forward pass for re-ranking task only (for evaluation).
        """
        token_emb, padding_mask = self.query_encoder(
            q_char_ids, q_is_number, q_numeric_ids
        )
        field_emb, field_mask = self.record_encoder(
            r_union_idx,
            r_desig_idx,
            r_prefix_idx,
            r_num_hash,
            r_num_val,
            r_suffix_idx,
            r_unit_id_idx,
        )
        return self.cross_attention.score_pair(
            token_emb, padding_mask, field_emb, field_mask
        )


# =============================================================================
# Batch Encoding Helpers (for mining efficiency)
# =============================================================================


def encode_query_batch(queries, max_len=MAX_QUERY_LEN):
    """Encode a batch of query strings into tensors."""
    char_ids_list = []
    is_number_list = []
    numeric_ids_list = []

    for query in queries:
        char_ids, _, is_number, _, numeric_ids = smart_truncate_nonspace(query, max_len)
        char_ids_list.append(char_ids)
        is_number_list.append(is_number)
        numeric_ids_list.append(numeric_ids)

    return {
        "char_ids": torch.tensor(char_ids_list, dtype=torch.long),
        "is_number": torch.tensor(is_number_list, dtype=torch.long),
        "numeric_ids": torch.tensor(numeric_ids_list, dtype=torch.long),
    }


def encode_record_batch(records, vocab):
    """Encode a batch of record dicts into tensors."""
    u_map = vocab["union_name_to_idx"]
    d_map = vocab["desig_name_to_idx"]
    p_map = vocab["prefix_to_idx"]
    s_map = vocab["suffix_to_idx"]
    uid_map = vocab["unit_id_to_idx"]

    union_idx = []
    desig_idx = []
    prefix_idx = []
    num_hash = []
    num_val = []
    suffix_idx = []
    unit_id_idx = []

    for rec in records:
        union_idx.append(u_map.get(rec["union_name"], 0))
        desig_idx.append(d_map.get(rec.get("desig_name", ""), 0))
        prefix_norm = normalize_designation(rec.get("prefix", "") or "")
        prefix_idx.append(p_map.get(prefix_norm, 0))
        num_hash.append(
            NUMBER_VOCAB.get(str(rec.get("desig_num", 0)), NUMBER_VOCAB["<UNK>"])
        )
        num_val.append(rec.get("desig_num", 0))
        suffix_norm = normalize_designation(rec.get("suffix", "") or "")
        suffix_idx.append(s_map.get(suffix_norm, 0))
        unit_id_idx.append(uid_map.get(rec.get("unit_id", ""), 0))

    return {
        "union_idx": torch.tensor(union_idx, dtype=torch.long),
        "desig_idx": torch.tensor(desig_idx, dtype=torch.long),
        "prefix_idx": torch.tensor(prefix_idx, dtype=torch.long),
        "num_hash": torch.tensor(num_hash, dtype=torch.long),
        "num_val": torch.tensor(num_val, dtype=torch.long),
        "suffix_idx": torch.tensor(suffix_idx, dtype=torch.long),
        "unit_id_idx": torch.tensor(unit_id_idx, dtype=torch.long),
    }


# =============================================================================
# Hard Negative Mining (ANCE-style)
# =============================================================================


def encode_all_records(model, fnum_to_records, vocab, batch_size=512):
    """
    Encode ALL records once at start of mining epoch.
    No filtering - include every record variant.
    Returns embeddings (on GPU), records list, and f_num tensor.
    """
    model.eval()
    device = next(model.parameters()).device

    all_records = []
    record_fnums = []

    for fnum, records in fnum_to_records.items():
        for rec in records:
            all_records.append(rec)
            record_fnums.append(fnum)

    all_record_embs = []
    for i in range(0, len(all_records), batch_size):
        batch_recs = all_records[i : i + batch_size]
        rec_batch = encode_record_batch(batch_recs, vocab)
        rec_batch = {k: v.to(device) for k, v in rec_batch.items()}

        with torch.no_grad():
            field_emb, _ = model.record_encoder(
                rec_batch["union_idx"],
                rec_batch["desig_idx"],
                rec_batch["prefix_idx"],
                rec_batch["num_hash"],
                rec_batch["num_val"],
                rec_batch["suffix_idx"],
                rec_batch["unit_id_idx"],
            )
            emb = model.dual_tower.encode_record(field_emb)
        all_record_embs.append(emb)  # Keep on GPU

    all_record_embs = torch.cat(all_record_embs, dim=0)  # [M, 128] on GPU
    record_fnums = torch.tensor(record_fnums)  # CPU is fine for indexing

    return all_record_embs, all_records, record_fnums


def mine_hard_candidates(
    model,
    train_examples,
    all_record_embs,
    all_records,
    record_fnums,
    vocab,
    k=10,
    query_batch_size=256,
):
    """
    Mine top-k most similar records for all training examples.

    Uses batched matrix multiplication for efficiency - encodes all queries
    and computes all similarities in batches on GPU.
    """
    model.eval()
    device = next(model.parameters()).device

    # Filter to train examples only
    train_indices = [
        i for i, ex in enumerate(train_examples) if ex.get("split") == "train"
    ]

    # Move record embeddings to GPU if not already
    all_record_embs = all_record_embs.to(device)  # [M, 128]

    # Process queries in batches
    for batch_start in tqdm(
        range(0, len(train_indices), query_batch_size), desc="Mining"
    ):
        batch_indices = train_indices[batch_start : batch_start + query_batch_size]
        batch_queries = [train_examples[i]["query"] for i in batch_indices]

        # Encode batch of queries
        query_batch = encode_query_batch(batch_queries)
        query_batch = {k: v.to(device) for k, v in query_batch.items()}

        with torch.no_grad():
            token_emb, mask = model.query_encoder(
                query_batch["char_ids"],
                query_batch["is_number"],
                query_batch["numeric_ids"],
            )
            query_embs = model.dual_tower.encode_query(token_emb, mask)  # [B, 128]

            # Compute similarities: [B, M]
            sims = torch.matmul(query_embs, all_record_embs.T)

            # Get top-k for all queries at once: [B, k]
            topk_indices = sims.topk(k, dim=1).indices  # [B, k]

        # Store results
        topk_indices = topk_indices.cpu().tolist()
        for i, ex_idx in enumerate(batch_indices):
            train_examples[ex_idx]["mined_candidates"] = [
                {"f_num": record_fnums[j].item(), "record": all_records[j]}
                for j in topk_indices[i]
            ]

    return train_examples


# =============================================================================
# Dataset
# =============================================================================


class DualTaskDataset(Dataset):
    """
    Dataset for dual-task training with in-batch negatives.

    Each example provides:
    - query tokens
    - positive record fields (including unit_id)

    Both tasks use in-batch negatives, so no explicit negatives needed.
    """

    def __init__(
        self,
        examples,
        union_name_to_idx,
        desig_name_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        unit_id_to_idx,
    ):
        self.examples = examples
        self.u_map = union_name_to_idx
        self.d_map = desig_name_to_idx
        self.p_map = prefix_to_idx
        self.s_map = suffix_to_idx
        self.uid_map = unit_id_to_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"]
        records = ex["records"]

        # Sample a positive record variant
        pos_record = random.choice(records)

        # Tokenize query (smart: no spaces, recover lost numbers)
        char_ids, _, is_number, token_type, numeric_ids = smart_truncate_nonspace(
            query, MAX_QUERY_LEN
        )

        # Encode record fields
        union_idx = self.u_map.get(pos_record["union_name"], 0)
        desig_idx = self.d_map.get(pos_record["desig_name"], 0)
        prefix_norm = (
            normalize_designation(pos_record["prefix"]) if pos_record["prefix"] else ""
        )
        suffix_norm = (
            normalize_designation(pos_record["suffix"]) if pos_record["suffix"] else ""
        )
        prefix_idx = self.p_map.get(prefix_norm, 0)
        suffix_idx = self.s_map.get(suffix_norm, 0)
        num_hash = NUMBER_VOCAB.get(str(pos_record["desig_num"]), NUMBER_VOCAB["<UNK>"])
        unit_id_idx = self.uid_map.get(pos_record.get("unit_id", ""), 0)

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "numeric_ids": torch.tensor(numeric_ids, dtype=torch.long),
            "union_idx": torch.tensor(union_idx, dtype=torch.long),
            "desig_idx": torch.tensor(desig_idx, dtype=torch.long),
            "prefix_idx": torch.tensor(prefix_idx, dtype=torch.long),
            "num_hash": torch.tensor(num_hash, dtype=torch.long),
            "num_val": torch.tensor(pos_record["desig_num"], dtype=torch.long),
            "suffix_idx": torch.tensor(suffix_idx, dtype=torch.long),
            "unit_id_idx": torch.tensor(unit_id_idx, dtype=torch.long),
            "f_num": torch.tensor(ex["f_num"], dtype=torch.long),
        }


class DualTaskDatasetWithCandidates(Dataset):
    """
    Training dataset that returns K+1 candidates per query (for ANCE training).

    Each example provides:
    - query tokens
    - K+1 candidate records (1 positive + K mined candidates)
    - f_num for each candidate (to identify positives in the loss)
    """

    def __init__(self, examples, vocab):
        self.examples = examples
        self.u_map = vocab["union_name_to_idx"]
        self.d_map = vocab["desig_name_to_idx"]
        self.p_map = vocab["prefix_to_idx"]
        self.s_map = vocab["suffix_to_idx"]
        self.uid_map = vocab["unit_id_to_idx"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"]
        query_fnum = ex["f_num"]

        # Positive record (always first in candidate list)
        pos_record = random.choice(ex["records"])

        # Build candidate list with f_nums: [(fnum, record), ...]
        candidates = [(query_fnum, pos_record)]  # First is always a positive

        if "mined_candidates" in ex and ex["mined_candidates"]:
            for cand_data in ex["mined_candidates"]:
                candidates.append((cand_data["f_num"], cand_data["record"]))

        # Encode query
        char_ids, _, is_number, _, numeric_ids = smart_truncate_nonspace(
            query, MAX_QUERY_LEN
        )

        # Encode all candidates [K+1, ...]
        cand_fnums = []  # Track f_num for supervised contrastive loss
        cand_union_idx = []
        cand_desig_idx = []
        cand_prefix_idx = []
        cand_num_hash = []
        cand_num_val = []
        cand_suffix_idx = []
        cand_unit_id_idx = []

        for fnum, rec in candidates:
            cand_fnums.append(fnum)
            cand_union_idx.append(self.u_map.get(rec["union_name"], 0))
            cand_desig_idx.append(self.d_map.get(rec.get("desig_name", ""), 0))
            prefix_norm = normalize_designation(rec.get("prefix", "") or "")
            cand_prefix_idx.append(self.p_map.get(prefix_norm, 0))
            cand_num_hash.append(
                NUMBER_VOCAB.get(str(rec.get("desig_num", 0)), NUMBER_VOCAB["<UNK>"])
            )
            cand_num_val.append(rec.get("desig_num", 0))
            suffix_norm = normalize_designation(rec.get("suffix", "") or "")
            cand_suffix_idx.append(self.s_map.get(suffix_norm, 0))
            cand_unit_id_idx.append(self.uid_map.get(rec.get("unit_id", ""), 0))

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "numeric_ids": torch.tensor(numeric_ids, dtype=torch.long),
            "f_num": torch.tensor(query_fnum, dtype=torch.long),  # Query's f_num
            # Candidate records [K+1, ...]
            "cand_fnums": torch.tensor(cand_fnums, dtype=torch.long),  # For loss
            "cand_union_idx": torch.tensor(cand_union_idx, dtype=torch.long),
            "cand_desig_idx": torch.tensor(cand_desig_idx, dtype=torch.long),
            "cand_prefix_idx": torch.tensor(cand_prefix_idx, dtype=torch.long),
            "cand_num_hash": torch.tensor(cand_num_hash, dtype=torch.long),
            "cand_num_val": torch.tensor(cand_num_val, dtype=torch.long),
            "cand_suffix_idx": torch.tensor(cand_suffix_idx, dtype=torch.long),
            "cand_unit_id_idx": torch.tensor(cand_unit_id_idx, dtype=torch.long),
            "num_candidates": torch.tensor(len(candidates), dtype=torch.long),
        }


# =============================================================================
# Loss Functions
# =============================================================================


def supervised_contrastive_loss(similarities, fnum_ids, temperature=0.07):
    """
    Supervised contrastive loss that treats all same-f_num pairs as positives.

    Args:
        similarities: [N, N] similarity/score matrix
        fnum_ids: [N] f_num identifiers for each example
        temperature: temperature scaling for similarities

    Returns:
        Scalar loss
    """
    # positive_mask[i,j] = True if same f_num (includes diagonal)
    positive_mask = fnum_ids.unsqueeze(0) == fnum_ids.unsqueeze(1)

    # Scale and stabilize
    sim_scaled = similarities / temperature
    sim_scaled = sim_scaled - sim_scaled.max(dim=1, keepdim=True).values

    # Softmax denominator
    exp_sim = torch.exp(sim_scaled)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Log probabilities
    log_prob = sim_scaled - torch.log(denom)

    # Average log prob over all positives for each query
    num_positives = positive_mask.sum(dim=1).float().clamp(min=1)
    loss = -(log_prob * positive_mask.float()).sum(dim=1) / num_positives

    return loss.mean()


def supervised_contrastive_loss_asymmetric(
    similarities, query_fnums, cand_fnums, temperature=0.07
):
    """
    Supervised contrastive loss for asymmetric N queries × M candidates.

    Args:
        similarities: [N, M] similarity/score matrix
        query_fnums: [N] f_num for each query
        cand_fnums: [M] f_num for each candidate
        temperature: temperature scaling

    Returns:
        Scalar loss
    """
    # positive_mask[i,j] = True if query i and candidate j have same f_num
    positive_mask = query_fnums.unsqueeze(1) == cand_fnums.unsqueeze(0)  # [N, M]

    # Scale and stabilize
    sim_scaled = similarities / temperature
    sim_scaled = sim_scaled - sim_scaled.max(dim=1, keepdim=True).values

    # Softmax denominator over all candidates
    exp_sim = torch.exp(sim_scaled)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Log probabilities
    log_prob = sim_scaled - torch.log(denom)

    # Average log prob over all positives for each query
    num_positives = positive_mask.sum(dim=1).float().clamp(min=1)
    loss = -(log_prob * positive_mask.float()).sum(dim=1) / num_positives

    return loss.mean()


def contrastive_loss_with_mask(similarities, positive_mask, temperature=0.07):
    """
    Contrastive loss where each query compares against its own candidates.

    Args:
        similarities: [N, K+1] - each query vs its own K+1 candidates
        positive_mask: [N, K+1] - True where candidate is a positive (same f_num)
        temperature: temperature scaling

    Returns:
        Scalar loss
    """
    # Scale and stabilize
    sim_scaled = similarities / temperature
    sim_scaled = sim_scaled - sim_scaled.max(dim=1, keepdim=True).values

    # Softmax denominator over candidates
    exp_sim = torch.exp(sim_scaled)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Log probabilities
    log_prob = sim_scaled - torch.log(denom)

    # Average log prob over all positives for each query
    num_positives = positive_mask.sum(dim=1).float().clamp(min=1)
    loss = -(log_prob * positive_mask.float()).sum(dim=1) / num_positives

    return loss.mean()


# =============================================================================
# Training Loop
# =============================================================================


def collate_fn(batch):
    """Collate batch items into tensors."""
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}


def collate_fn_with_candidates(batch):
    """
    Collate batch items with variable-length candidates.

    Pads candidate tensors to max number of candidates in the batch.
    Uses -1 for cand_fnums padding (will never match a real f_num).
    """
    max_cands = max(item["num_candidates"].item() for item in batch)

    # Candidate field keys that need padding
    cand_keys = [
        "cand_fnums",
        "cand_union_idx",
        "cand_desig_idx",
        "cand_prefix_idx",
        "cand_num_hash",
        "cand_num_val",
        "cand_suffix_idx",
        "cand_unit_id_idx",
    ]

    padded_batch = []
    for item in batch:
        num_cands = item["num_candidates"].item()
        new_item = {}

        for key, val in item.items():
            if key in cand_keys:
                if num_cands < max_cands:
                    pad_size = max_cands - num_cands
                    # Use -1 for cand_fnums (won't match any f_num), 0 for others
                    pad_value = -1 if key == "cand_fnums" else 0
                    new_item[key] = F.pad(val, (0, pad_size), value=pad_value)
                else:
                    new_item[key] = val
            else:
                new_item[key] = val

        padded_batch.append(new_item)

    return {
        k: torch.stack([item[k] for item in padded_batch])
        for k in padded_batch[0].keys()
    }


def evaluate(model, dataloader, device, temperature=0.07):
    """Evaluate on both tasks. Counts any same-f_num prediction as correct."""
    model.eval()
    total_retrieval_correct = 0
    total_rerank_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            N = batch["char_ids"].shape[0]
            fnum_ids = batch["f_num"]

            retrieval_sim, rerank_scores = model.forward_dual_task(
                batch["char_ids"],
                batch["is_number"],
                batch["numeric_ids"],
                batch["union_idx"],
                batch["desig_idx"],
                batch["prefix_idx"],
                batch["num_hash"],
                batch["num_val"],
                batch["suffix_idx"],
                batch["unit_id_idx"],
            )

            # Retrieval accuracy: is predicted record's f_num same as query's f_num?
            retrieval_preds = retrieval_sim.argmax(dim=1)
            pred_fnums = fnum_ids[retrieval_preds]
            retrieval_correct = (pred_fnums == fnum_ids).sum().item()

            # Re-ranking accuracy: same logic
            rerank_preds = rerank_scores.argmax(dim=1)
            pred_fnums = fnum_ids[rerank_preds]
            rerank_correct = (pred_fnums == fnum_ids).sum().item()

            total_retrieval_correct += retrieval_correct
            total_rerank_correct += rerank_correct
            total_samples += N

    return {
        "retrieval_acc": total_retrieval_correct / total_samples,
        "rerank_acc": total_rerank_correct / total_samples,
    }


@click.command()
@click.option("--epochs", default=40, help="Number of epochs to train")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option(
    "--checkpoint", type=click.Path(exists=True), help="Resume from checkpoint"
)
@click.option(
    "--mine-every", default=5, help="Mine candidates every N epochs (0 to disable)"
)
@click.option("--mine-k", default=50, help="Number of candidates to mine per example")
@click.option(
    "--fresh-optimizer", is_flag=True, help="Don't load optimizer state from checkpoint"
)
def train_dual_task(
    epochs=40,
    batch_size=128,
    lr=1e-4,
    checkpoint=None,
    mine_every=5,
    mine_k=10,
    fresh_optimizer=False,
    retrieval_weight=1.0,
    reranking_weight=1.0,
    temperature=0.07,
):
    """
    Dual-task training loop with ANCE-style hard negative mining.

    Both tasks share:
    - CharCNN text encoder
    - Query self-attention layers
    - Record field embeddings (including unit_id and field types)

    Task-specific:
    - Retrieval: attention pooling + projection
    - Re-ranking: cross-attention + classifier

    ANCE mining:
    - Periodically (every mine_every epochs), encode all records and find
      top-k most similar candidates for each training query
    - Training uses N×(K+1) candidates per batch (mined + positive)
    - Supervised contrastive loss identifies positives via f_num matching
    """
    print(f"Using device: {DEVICE}")
    if mine_every > 0:
        print(f"ANCE mining: every {mine_every} epochs, k={mine_k}")

    # Track starting epoch for resumption
    start_epoch = 0
    ckpt = None

    # Load checkpoint first if resuming
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Will resume from epoch {start_epoch}")
        if "val_metrics" in ckpt:
            print(f"  Previous val metrics: {ckpt['val_metrics']}")

    # Always load fresh vocabularies
    print("Loading vocabularies...")
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    union_name_to_idx = vocab["union_name_to_idx"]
    desig_name_to_idx = vocab["desig_name_to_idx"]
    prefix_to_idx = vocab["prefix_to_idx"]
    suffix_to_idx = vocab["suffix_to_idx"]
    unit_id_to_idx = vocab["unit_id_to_idx"]

    # Load training examples
    print("Loading training examples...")
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    print(f"  {len(union_name_to_idx)} union names")
    print(f"  {len(desig_name_to_idx)} desig names")
    print(f"  {len(unit_id_to_idx)} unit_ids")

    # Split by pre-assigned split
    train_ex = [ex for ex in all_examples if ex["split"] == "train"]
    val_ex = [ex for ex in all_examples if ex["split"] == "val"]
    test_ex = [ex for ex in all_examples if ex["split"] == "test"]

    print(f"  Train: {len(train_ex)}, Val: {len(val_ex)}, Test: {len(test_ex)}")

    # Build fnum_to_records mapping for mining (use all examples, deduplicated)
    fnum_to_records = {}
    seen_records = {}  # (fnum, record_key) -> record dict
    for ex in all_examples:
        fnum = ex["f_num"]
        for rec in ex["records"]:
            # Create hashable key from record fields
            key = (
                fnum,
                rec.get("union_name", ""),
                rec.get("desig_name", ""),
                rec.get("prefix", ""),
                rec.get("desig_num", 0),
                rec.get("suffix", ""),
                rec.get("unit_id", ""),
            )
            if key not in seen_records:
                seen_records[key] = rec
                if fnum not in fnum_to_records:
                    fnum_to_records[fnum] = []
                fnum_to_records[fnum].append(rec)
    total_records = sum(len(recs) for recs in fnum_to_records.values())
    print(
        f"  {len(fnum_to_records)} unique f_nums, {total_records} unique records for mining"
    )

    # Create datasets
    train_ds = DualTaskDataset(
        train_ex,
        union_name_to_idx,
        desig_name_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        unit_id_to_idx,
    )
    val_ds = DualTaskDataset(
        val_ex,
        union_name_to_idx,
        desig_name_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        unit_id_to_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_ds = DualTaskDataset(
        test_ex,
        union_name_to_idx,
        desig_name_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        unit_id_to_idx,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Initialize model
    model = DualTaskModel(
        num_union_names=len(union_name_to_idx),
        num_desig_names=len(desig_name_to_idx),
        num_prefixes=len(prefix_to_idx),
        num_suffixes=len(suffix_to_idx),
        num_unit_ids=len(unit_id_to_idx),
    )
    model.to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load model weights if resuming
    if ckpt:
        ckpt_state = ckpt["model_state_dict"]
        model_state = model.state_dict()

        # Copy weights, handling size mismatches for embedding layers
        for name, param in ckpt_state.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    model_state[name].copy_(param)
                elif (
                    param.shape[0] < model_state[name].shape[0]
                    and len(param.shape) == 2
                ):
                    # Embedding grew - copy old weights to beginning
                    model_state[name][: param.shape[0]].copy_(param)
                    print(
                        f"  Expanded {name}: {param.shape[0]} -> {model_state[name].shape[0]}"
                    )
                else:
                    print(
                        f"  Skipped {name}: shape mismatch {param.shape} vs {model_state[name].shape}"
                    )

        model.load_state_dict(model_state)
        print("  Loaded model weights from checkpoint")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler (accounts for total epochs from start)
    total_epochs_from_start = start_epoch + epochs
    total_steps = len(train_loader) * total_epochs_from_start
    warmup_steps = len(train_loader)  # 1 epoch warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # If resuming, restore optimizer/scheduler state
    # Skip optimizer state when using ANCE (mine_every > 0) or --fresh-optimizer
    # The training dynamics change significantly with ANCE, causing NaN gradients
    skip_optimizer = fresh_optimizer or mine_every > 0
    if ckpt and "optimizer_state_dict" in ckpt and not skip_optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print("  Restored optimizer and scheduler state")
    elif ckpt and skip_optimizer:
        print("  Using fresh optimizer (ANCE mode or --fresh-optimizer)")

    best_val_acc = ckpt.get("best_val_acc", 0.0) if ckpt else 0.0
    step = start_epoch * len(train_loader)

    # Track whether we have mined candidates
    has_mined_candidates = False

    end_epoch = start_epoch + epochs
    for epoch in range(start_epoch, end_epoch):
        # === Mining step (periodically, starting after warmup) ===
        if mine_every > 0 and epoch > 0 and epoch % mine_every == 0:
            print(f"\nMining hard candidates at epoch {epoch}...")

            # Encode all records ONCE
            all_record_embs, all_records, record_fnums = encode_all_records(
                model, fnum_to_records, vocab, batch_size=512
            )
            print(f"  Encoded {len(all_records)} records")

            # Mine for each training example
            train_ex = mine_hard_candidates(
                model,
                train_ex,
                all_record_embs,
                all_records,
                record_fnums,
                vocab,
                k=mine_k,
            )
            has_mined_candidates = True

            # Recreate dataset and dataloader with candidates
            train_ds = DualTaskDatasetWithCandidates(train_ex, vocab)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn_with_candidates,
                drop_last=True,
            )
            print("  Recreated dataloader with mined candidates")

        model.train()
        epoch_retrieval_loss = 0.0
        epoch_rerank_loss = 0.0
        epoch_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch}")
        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad()

            # Check if batch has mined candidates
            if has_mined_candidates and "cand_fnums" in batch:
                # === ANCE training: each query vs its own K+1 candidates ===
                N = batch["char_ids"].shape[0]
                K_plus_1 = batch["cand_union_idx"].shape[1]  # candidates per query

                # Query f_nums [N] and candidate f_nums [N, K+1]
                query_fnums = batch["f_num"]  # [N]
                cand_fnums = batch["cand_fnums"]  # [N, K+1]

                # Encode queries [N, 128]
                token_emb, padding_mask = model.query_encoder(
                    batch["char_ids"], batch["is_number"], batch["numeric_ids"]
                )
                query_emb = model.dual_tower.encode_query(token_emb, padding_mask)

                # Encode candidates [N*(K+1), ...] then reshape to [N, K+1, 128]
                cand_union_idx = batch["cand_union_idx"].view(-1)
                cand_desig_idx = batch["cand_desig_idx"].view(-1)
                cand_prefix_idx = batch["cand_prefix_idx"].view(-1)
                cand_num_hash = batch["cand_num_hash"].view(-1)
                cand_num_val = batch["cand_num_val"].view(-1)
                cand_suffix_idx = batch["cand_suffix_idx"].view(-1)
                cand_unit_id_idx = batch["cand_unit_id_idx"].view(-1)

                field_emb_flat, field_mask_flat = model.record_encoder(
                    cand_union_idx,
                    cand_desig_idx,
                    cand_prefix_idx,
                    cand_num_hash,
                    cand_num_val,
                    cand_suffix_idx,
                    cand_unit_id_idx,
                )
                cand_emb_flat = model.dual_tower.encode_record(
                    field_emb_flat
                )  # [N*(K+1), 128]
                cand_emb = cand_emb_flat.view(N, K_plus_1, -1)  # [N, K+1, 128]

                # Compute similarities: each query vs its own candidates [N, K+1]
                retrieval_sim = torch.einsum("nd,nkd->nk", query_emb, cand_emb)

                # Contrastive loss: positive_mask[i,j] = (query_fnums[i] == cand_fnums[i,j])
                positive_mask = query_fnums.unsqueeze(1) == cand_fnums  # [N, K+1]
                retrieval_loss = contrastive_loss_with_mask(
                    retrieval_sim, positive_mask, temperature
                )

                # Re-ranking: each query vs its own K+1 candidates
                seq_len = token_emb.shape[1]
                field_emb = field_emb_flat.view(
                    N, K_plus_1, 6, -1
                )  # [N, K+1, 6, embed_dim]
                field_mask = field_mask_flat.view(N, K_plus_1, 6)  # [N, K+1, 6]

                # Expand queries: [N, K+1, seq_len, embed_dim]
                q_exp = token_emb.unsqueeze(1).expand(N, K_plus_1, seq_len, -1)
                q_mask_exp = padding_mask.unsqueeze(1).expand(N, K_plus_1, seq_len)

                # Flatten to [N*K+1, ...] for batch scoring
                q_flat = q_exp.reshape(N * K_plus_1, seq_len, -1)
                q_mask_flat = q_mask_exp.reshape(N * K_plus_1, seq_len)
                r_flat = field_emb.reshape(N * K_plus_1, 6, -1)
                r_mask_flat = field_mask.reshape(N * K_plus_1, 6)

                # Score all pairs and reshape to [N, K+1]
                scores_flat = model.cross_attention.score_pair(
                    q_flat, q_mask_flat, r_flat, r_mask_flat
                )
                rerank_scores = scores_flat.view(N, K_plus_1)

                rerank_loss = contrastive_loss_with_mask(
                    rerank_scores, positive_mask, temperature=1.0
                )
            else:
                # === Standard training with in-batch negatives ===
                retrieval_sim, rerank_scores = model.forward_dual_task(
                    batch["char_ids"],
                    batch["is_number"],
                    batch["numeric_ids"],
                    batch["union_idx"],
                    batch["desig_idx"],
                    batch["prefix_idx"],
                    batch["num_hash"],
                    batch["num_val"],
                    batch["suffix_idx"],
                    batch["unit_id_idx"],
                )

                # Supervised contrastive loss: all same-f_num pairs are positives
                retrieval_loss = supervised_contrastive_loss(
                    retrieval_sim, batch["f_num"], temperature
                )

                # Re-ranking loss (rerank_scores are raw logits, so use temperature=1.0)
                rerank_loss = supervised_contrastive_loss(
                    rerank_scores, batch["f_num"], temperature=1.0
                )

            # Combined loss
            loss = retrieval_weight * retrieval_loss + reranking_weight * rerank_loss

            # Check for NaN and skip batch if detected
            if torch.isnan(loss):
                print(f"\nWARNING: NaN in loss at epoch {epoch}, skipping batch")
                print(
                    f"  retrieval_loss={retrieval_loss.item()}, rerank_loss={rerank_loss.item()}"
                )
                optimizer.zero_grad()
                continue

            loss.backward()

            # Check for NaN gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"\nWARNING: NaN gradient in {name}, skipping batch")
                    has_nan_grad = True
                    break
            if has_nan_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_retrieval_loss += retrieval_loss.item()
            epoch_rerank_loss += rerank_loss.item()
            epoch_batches += 1
            step += 1

            pbar.set_postfix(
                {
                    "ret": f"{retrieval_loss.item():.3f}",
                    "rer": f"{rerank_loss.item():.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # Epoch summary
        if epoch_batches > 0:
            avg_ret = epoch_retrieval_loss / epoch_batches
            avg_rer = epoch_rerank_loss / epoch_batches
            print(f"Epoch {epoch+1}: retrieval={avg_ret:.4f}, rerank={avg_rer:.4f}")
        else:
            print(f"Epoch {epoch+1}: All batches skipped due to NaN!")

        # Validation
        val_metrics = evaluate(model, val_loader, DEVICE, temperature)
        print(
            f"  Val: retrieval_acc={val_metrics['retrieval_acc']:.4f}, rerank_acc={val_metrics['rerank_acc']:.4f}"
        )

        # Save best model (based on average of both metrics)
        avg_acc = (val_metrics["retrieval_acc"] + val_metrics["rerank_acc"]) / 2
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "union_name_to_idx": union_name_to_idx,
                    "desig_name_to_idx": desig_name_to_idx,
                    "prefix_to_idx": prefix_to_idx,
                    "suffix_to_idx": suffix_to_idx,
                    "unit_id_to_idx": unit_id_to_idx,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_val_acc": best_val_acc,
                },
                "training/dual_task_model.pt",
            )
            print(f"  Saved best model (avg_acc={avg_acc:.4f})")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    best_ckpt = torch.load("training/dual_task_model.pt", map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, DEVICE, temperature)
    print(
        f"Test: retrieval_acc={test_metrics['retrieval_acc']:.4f}, rerank_acc={test_metrics['rerank_acc']:.4f}"
    )


if __name__ == "__main__":
    train_dual_task()
