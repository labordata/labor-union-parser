"""BiLSTM model for union name extraction with pointer-based designation selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import tokenize, MAX_TOKEN_LEN


class UnionNameExtractor(nn.Module):
    """
    Self-attention + BiLSTM extractor with:
    - Self-attention for token context (shared)
    - Set attention pooling for affiliation classification
    - BiLSTM + pointer selection for designation (picks which number token)
    - is_number feature to help identify number tokens (even rare/unseen ones)
    """

    def __init__(
        self,
        token_vocab_size: int,
        num_affs: int,
        token_embed_dim: int = 64,
        hidden_dim: int = 512,
        aff_embed_dim: int = 64,
        num_attn_heads: int = 4,
        num_feature_dim: int = 16,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(
            token_vocab_size, token_embed_dim, padding_idx=0
        )
        self.aff_embed = nn.Embedding(num_affs, aff_embed_dim)

        # Learned embedding for "is_number" feature (2 classes: not number, is number)
        self.num_feature_embed = nn.Embedding(2, num_feature_dim)

        # Combined embedding dimension
        combined_dim = token_embed_dim + num_feature_dim

        # Self-attention: tokens attend to each other
        self.self_attn = nn.MultiheadAttention(
            combined_dim, num_heads=num_attn_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(combined_dim)

        # Set attention for affiliation classification (order-invariant pooling)
        self.aff_attn = nn.Linear(combined_dim, 1)
        self.aff_classifier = nn.Linear(combined_dim, num_affs)

        # BiLSTM for designation selection
        self.lstm = nn.LSTM(
            combined_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Designation pointer: scores each position
        self.desig_scorer = nn.Linear(hidden_dim * 2 + aff_embed_dim, 1)

        # Learned null score for "no designation" (index 0)
        self.null_score = nn.Parameter(torch.zeros(1))

        self.num_affs = num_affs

    def forward(
        self, token_ids, token_mask, is_number=None, aff_labels=None, desig_labels=None
    ):
        """
        Forward pass.

        Args:
            token_ids: [batch, seq_len] token indices
            token_mask: [batch, seq_len] 1 for valid tokens, 0 for padding
            is_number: [batch, seq_len] 1 if token is a number, 0 otherwise
            aff_labels: [batch] affiliation indices (for training)
            desig_labels: [batch] designation position (0 = no desig, 1+ = token index + 1)

        Returns:
            Dictionary with predictions and optionally losses
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        token_emb = self.token_embed(token_ids)

        # Add is_number feature embedding
        if is_number is None:
            is_number = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=token_ids.device
            )
        num_feature_emb = self.num_feature_embed(is_number)
        token_emb = torch.cat([token_emb, num_feature_emb], dim=-1)

        # Self-attention: tokens attend to each other to build context
        key_padding_mask = token_mask == 0
        attn_out, _ = self.self_attn(
            token_emb, token_emb, token_emb, key_padding_mask=key_padding_mask
        )
        token_emb_ctx = self.attn_norm(token_emb + attn_out)  # residual connection

        # Affiliation classification via set attention pooling
        attn_scores = self.aff_attn(token_emb_ctx).squeeze(-1)  # [batch, seq]
        attn_scores = attn_scores.masked_fill(token_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(
            -1
        )  # [batch, seq, 1]
        pooled = (attn_weights * token_emb_ctx).sum(dim=1)  # [batch, embed_dim]
        aff_logits = self.aff_classifier(pooled)
        aff_idx = aff_logits.argmax(dim=1)

        # Get affiliation embedding for designation scoring
        if aff_labels is not None:
            aff_emb = self.aff_embed(aff_labels)
        else:
            aff_emb = self.aff_embed(aff_idx)

        aff_emb_broadcast = aff_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # BiLSTM for designation selection
        lstm_out, _ = self.lstm(token_emb_ctx)
        lstm_with_aff = torch.cat([lstm_out, aff_emb_broadcast], dim=-1)

        # Score each position for designation
        position_scores = self.desig_scorer(lstm_with_aff).squeeze(-1)  # [batch, seq]

        # Mask: only number tokens can be designation, also mask padding
        valid_desig_mask = (is_number == 1) & (token_mask == 1)
        position_scores = position_scores.masked_fill(~valid_desig_mask, float("-inf"))

        # Prepend null score (index 0 = no designation)
        null_scores = self.null_score.expand(batch_size, 1)  # [batch, 1]
        desig_scores = torch.cat(
            [null_scores, position_scores], dim=1
        )  # [batch, 1 + seq]

        # Prediction: argmax over [null, pos0, pos1, ...]
        desig_pred = desig_scores.argmax(
            dim=1
        )  # [batch] - 0 means no desig, 1+ means position

        results = {
            "aff_logits": aff_logits,
            "aff_idx": aff_idx,
            "desig_scores": desig_scores,
            "desig_pred": desig_pred,
        }

        if desig_labels is not None and aff_labels is not None:
            desig_loss = F.cross_entropy(desig_scores, desig_labels)
            aff_loss = F.cross_entropy(aff_logits, aff_labels)
            results["desig_loss"] = desig_loss
            results["aff_loss"] = aff_loss
            results["total_loss"] = aff_loss + desig_loss

        return results


def create_desig_label(text: str, desig_num: str, max_len: int = MAX_TOKEN_LEN) -> int:
    """Create designation label (position index) for training.

    Returns:
        0 if no designation, otherwise token_position + 1
    """
    if not desig_num or desig_num == "N/A":
        return 0

    tokens = tokenize(text)
    desig_str = str(desig_num).lstrip("0") or "0"

    # Find the last occurrence of the designation number as a token
    best_idx = None
    for i, token in enumerate(tokens[:max_len]):
        if token == desig_str:
            best_idx = i

    if best_idx is not None:
        return best_idx + 1  # +1 because 0 is reserved for "no designation"

    return 0  # Not found


def extract_desig_from_pred(text: str, desig_pred: int) -> str:
    """Extract designation number from pointer prediction.

    Args:
        text: Original text
        desig_pred: Predicted index (0 = no desig, 1+ = token position + 1)

    Returns:
        Designation string, or empty string if none
    """
    if desig_pred == 0:
        return ""

    tokens = tokenize(text)
    token_idx = desig_pred - 1  # Convert back to 0-indexed

    if token_idx < len(tokens):
        token = tokens[token_idx]
        if token.isdigit():
            return token

    return ""
