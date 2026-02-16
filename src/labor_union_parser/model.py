"""Model architecture for dual-task retrieval and re-ranking."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from labor_union_parser.char_cnn import NUMBER_VOCAB, CharacterCNN
from labor_union_parser.layers import RoPESelfAttentionLayer, SelfAttentionLayer

EMBED_DIM = 64
RETRIEVAL_DIM = 128
MAX_QUERY_LEN = 20  # Total sequence will be 21 with [CLS]


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

        # Learned [CLS] token for global summary
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Shared frozen number embedding
        self.frozen_num_embed = nn.Embedding(len(NUMBER_VOCAB), 32)
        self.frozen_num_embed.weight.requires_grad = False

        # Project to embed_dim (char_cnn + frozen_num, no is_num feature)
        input_dim = char_cnn.embed_dim + 32
        self.proj = nn.Linear(input_dim, embed_dim)

        # Self-attention with RoPE (shared between tasks)
        self.self_attn1 = RoPESelfAttentionLayer(
            embed_dim, num_heads=4, max_seq_len=MAX_QUERY_LEN + 1
        )
        self.self_attn2 = RoPESelfAttentionLayer(
            embed_dim, num_heads=4, max_seq_len=MAX_QUERY_LEN + 1
        )

    def forward(self, char_ids, is_number, numeric_ids):
        """
        Returns token-level embeddings and padding mask.

        Args:
            char_ids: [batch, seq_len, char_len]
            is_number: [batch, seq_len]
            numeric_ids: [batch, seq_len]

        Returns:
            token_emb: [batch, seq_len + 1, embed_dim]
            padding_mask: [batch, seq_len + 1] - True for padded positions
        """
        batch_size = char_ids.shape[0]

        # CharCNN embeddings (zeroed for numbers)
        char_emb = self.char_cnn(char_ids)
        is_num_mask = is_number.unsqueeze(-1).float()
        char_emb = char_emb * (1 - is_num_mask)

        # Frozen number embeddings (zeroed for non-numbers)
        frozen_emb = self.frozen_num_embed(numeric_ids) * is_num_mask

        # Combine and project
        combined = torch.cat([char_emb, frozen_emb], dim=-1)
        token_emb = self.proj(combined)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        token_emb = torch.cat([cls_tokens, token_emb], dim=1)

        # Padding mask (including [CLS] which is never masked)
        q_padding_mask = char_ids.sum(-1) == 0
        cls_mask = torch.zeros(batch_size, 1, device=token_emb.device, dtype=torch.bool)
        padding_mask = torch.cat([cls_mask, q_padding_mask], dim=1)

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
        num_suffixes,
        num_unit_ids,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Field embeddings (matching original dimensions)
        self.union_embed = nn.Embedding(num_union_names, 64)
        self.desig_embed = nn.Embedding(num_desig_names + 1, 16)
        self.suffix_embed = nn.Embedding(num_suffixes, 16)
        self.unit_id_embed = nn.Embedding(num_unit_ids, embed_dim)

        # Number: frozen embedding only (no magnitude - numbers are IDs, not quantities)
        self.frozen_num_embed = nn.Embedding(len(NUMBER_VOCAB), 32)
        self.frozen_num_embed.weight.requires_grad = False

        # Projections to embed_dim
        self.union_proj = nn.Linear(64, embed_dim)
        self.desig_proj = nn.Linear(16, embed_dim)
        self.prefix_num_proj = nn.Linear(32, embed_dim)
        self.suffix_proj = nn.Linear(16, embed_dim)
        self.num_proj = nn.Linear(32, embed_dim)

        # Field type embeddings: union=0, desig=1, prefix=2, number=3, suffix=4, unit_id=5
        self.field_type_embed = nn.Embedding(6, embed_dim)

    def forward(
        self,
        union_idx,
        desig_idx,
        prefix_hash,
        num_hash,
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
        p_e = self.prefix_num_proj(self.frozen_num_embed(prefix_hash))
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

        # No masking: index-0 embeddings serve as learned "not applicable" sentinels
        # so cross-attention produces active mismatch signals instead of no signal
        field_mask = torch.zeros(batch_size, 6, device=device, dtype=torch.bool)

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


class CrossEncoderHead(nn.Module):
    """
    Cross-encoder head for re-ranking.

    Concatenates query token embeddings and record field embeddings into a
    single sequence, then runs full self-attention over it. This lets query
    tokens and record fields attend to each other bidirectionally.

    Segment embeddings distinguish query tokens (segment 0) from record
    fields (segment 1). No positional encoding is added — query tokens
    already carry positional info from the shared encoder's RoPE layers,
    and record fields are distinguished by their field type embeddings.
    """

    NUM_FIELDS = 6

    def __init__(self, embed_dim=EMBED_DIM, pool_mode="cls", num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_mode = pool_mode  # "cls" or "mean"

        # Segment embeddings: 0 = query, 1 = record
        self.segment_embed = nn.Embedding(2, embed_dim)

        # Full self-attention layers (no positional encoding)
        self.self_attn1 = SelfAttentionLayer(embed_dim, num_heads=num_heads)
        self.self_attn2 = SelfAttentionLayer(embed_dim, num_heads=num_heads)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for mod in self.classifier:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        nn.init.xavier_uniform_(self.segment_embed.weight)

    def _concat_and_mask(self, token_emb, token_mask, field_emb, field_mask):
        """
        Concatenate query and record into a single sequence with segment embeddings.

        Returns:
            combined: [batch, seq_len + 6, embed_dim]
            combined_mask: [batch, seq_len + 6]
        """
        B, Q, _ = token_emb.shape
        device = token_emb.device

        # Segment IDs: 0 for query tokens, 1 for record fields
        query_seg = torch.zeros(B, Q, dtype=torch.long, device=device)
        record_seg = torch.ones(B, self.NUM_FIELDS, dtype=torch.long, device=device)

        # Add segment embeddings
        token_emb = token_emb + self.segment_embed(query_seg)
        field_emb = field_emb + self.segment_embed(record_seg)

        # Concatenate: [batch, Q + 6, embed_dim]
        combined = torch.cat([token_emb, field_emb], dim=1)
        combined_mask = torch.cat([token_mask, field_mask], dim=1)

        return combined, combined_mask

    def _pool(self, combined, token_mask):
        """Pool from the combined sequence."""
        if self.pool_mode == "mean":
            # Mean pool over non-padding query tokens only (exclude record fields)
            Q = token_mask.shape[1]
            query_part = combined[:, :Q]
            mask = ~token_mask
            pooled = (query_part * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        else:
            # [CLS] token at index 0
            pooled = combined[:, 0]
        return pooled

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
        combined, combined_mask = self._concat_and_mask(
            token_emb, token_mask, field_emb, field_mask
        )

        combined = self.self_attn1(combined, combined_mask)
        combined = self.self_attn2(combined, combined_mask)

        pooled = self._pool(combined, token_mask)
        return self.classifier(pooled).squeeze(-1)

    def score_pair_with_attention(self, token_emb, token_mask, field_emb, field_mask):
        """
        Like score_pair but also returns self-attention weights from both layers.

        Returns:
            scores: [batch]
            attn_weights_1: [batch, num_heads, seq_len+6, seq_len+6]
            attn_weights_2: [batch, num_heads, seq_len+6, seq_len+6]
        """
        combined, combined_mask = self._concat_and_mask(
            token_emb, token_mask, field_emb, field_mask
        )

        combined, attn1 = self.self_attn1(
            combined, combined_mask, return_attn_weights=True
        )
        combined, attn2 = self.self_attn2(
            combined, combined_mask, return_attn_weights=True
        )

        pooled = self._pool(combined, token_mask)
        scores = self.classifier(pooled).squeeze(-1)
        return scores, attn1, attn2

    def score_all_pairs(self, token_emb, token_mask, field_emb, field_mask):
        """
        Score all N x N query-record pairs using in-batch negatives.

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
        q_exp = token_emb.unsqueeze(1).expand(N, N, seq_len, self.embed_dim)
        q_mask_exp = token_mask.unsqueeze(1).expand(N, N, seq_len)

        # Expand records: [N, N, 6, embed_dim]
        r_exp = field_emb.unsqueeze(0).expand(N, N, self.NUM_FIELDS, self.embed_dim)
        r_mask_exp = field_mask.unsqueeze(0).expand(N, N, self.NUM_FIELDS)

        # Flatten to [N*N, ...] for batch processing
        q_flat = q_exp.reshape(N * N, seq_len, self.embed_dim)
        q_mask_flat = q_mask_exp.reshape(N * N, seq_len)
        r_flat = r_exp.reshape(N * N, self.NUM_FIELDS, self.embed_dim)
        r_mask_flat = r_mask_exp.reshape(N * N, self.NUM_FIELDS)

        # Score all pairs
        scores_flat = self.score_pair(q_flat, q_mask_flat, r_flat, r_mask_flat)

        return scores_flat.view(N, N)


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
        num_suffixes,
        num_unit_ids,
        embed_dim=EMBED_DIM,
        pool_mode="cls",
        ca_num_heads=4,
    ):
        super().__init__()

        # Shared components
        self.char_cnn = CharacterCNN(embed_dim=embed_dim, char_embed_dim=16)
        self.query_encoder = SharedQueryEncoder(self.char_cnn, embed_dim)
        self.record_encoder = SharedRecordEncoder(
            num_union_names,
            num_desig_names,
            num_suffixes,
            num_unit_ids,
            embed_dim,
        )

        # Sync frozen number embeddings
        self.record_encoder.frozen_num_embed = self.query_encoder.frozen_num_embed

        # Task-specific heads
        self.dual_tower = DualTowerHead(self.record_encoder, embed_dim)
        self.cross_encoder = CrossEncoderHead(
            embed_dim, pool_mode=pool_mode, num_heads=ca_num_heads
        )

    def forward_dual_task(
        self,
        q_char_ids,
        q_is_number,
        q_numeric_ids,
        r_union_idx,
        r_desig_idx,
        r_prefix_hash,
        r_num_hash,
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
            r_prefix_hash,
            r_num_hash,
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
        # Score all N x N pairs using cross-attention
        rerank_scores = self.cross_encoder.score_all_pairs(
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
        r_prefix_hash,
        r_num_hash,
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
            r_prefix_hash,
            r_num_hash,
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
        r_prefix_hash,
        r_num_hash,
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
            r_prefix_hash,
            r_num_hash,
            r_suffix_idx,
            r_unit_id_idx,
        )
        return self.cross_encoder.score_pair(
            token_emb, padding_mask, field_emb, field_mask
        )

    def forward_reranking_with_attention(
        self,
        q_char_ids,
        q_is_number,
        q_numeric_ids,
        r_union_idx,
        r_desig_idx,
        r_prefix_hash,
        r_num_hash,
        r_suffix_idx,
        r_unit_id_idx,
    ):
        """
        Like forward_reranking but also returns self-attention weights.

        Returns:
            scores: [batch]
            attn_weights_1: [batch, num_heads, seq_len+6, seq_len+6]
            attn_weights_2: [batch, num_heads, seq_len+6, seq_len+6]
        """
        token_emb, padding_mask = self.query_encoder(
            q_char_ids, q_is_number, q_numeric_ids
        )
        field_emb, field_mask = self.record_encoder(
            r_union_idx,
            r_desig_idx,
            r_prefix_hash,
            r_num_hash,
            r_suffix_idx,
            r_unit_id_idx,
        )
        return self.cross_encoder.score_pair_with_attention(
            token_emb, padding_mask, field_emb, field_mask
        )
