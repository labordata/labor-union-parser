"""CharCNN model for union name extraction with pointer-based designation selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .char_cnn import (
    CharacterCNN,
    tokenize_to_chars,
    SPECIAL_TOKEN_VOCAB,
)
from .tokenizer import MAX_TOKEN_LEN


class CharCNNExtractor(nn.Module):
    """
    Union name extractor using CharacterCNN for word token embeddings.

    Hybrid approach:
    - Words → CharacterCNN (typo robust)
    - Numbers/punct/space → simple embedding lookup (fast)

    Architecture:
    - CharacterCNN for word embeddings only
    - Embedding lookup for non-word tokens
    - Transformer encoder (stacked self-attention + FFN) for phrase patterns
    - Set attention pooling for affiliation classification
    - BiLSTM + pointer selection for designation
    """

    def __init__(
        self,
        num_affs: int,
        token_embed_dim: int = 64,
        hidden_dim: int = 512,
        aff_embed_dim: int = 64,
        num_attn_heads: int = 4,
        num_feature_dim: int = 16,
        char_embed_dim: int = 16,
        num_attn_layers: int = 3,
    ):
        super().__init__()

        self.token_embed_dim = token_embed_dim

        # CharacterCNN for word tokens only
        self.char_cnn = CharacterCNN(
            embed_dim=token_embed_dim,
            char_embed_dim=char_embed_dim,
        )

        # Simple embedding for non-word tokens (numbers, punct, space)
        self.special_embed = nn.Embedding(
            len(SPECIAL_TOKEN_VOCAB), token_embed_dim, padding_idx=0
        )

        self.aff_embed = nn.Embedding(num_affs, aff_embed_dim)

        # Learned embedding for "is_number" feature
        self.num_feature_embed = nn.Embedding(2, num_feature_dim)

        # Combined embedding dimension
        combined_dim = token_embed_dim + num_feature_dim

        # Transformer encoder for token context (stacked self-attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=combined_dim,
            nhead=num_attn_heads,
            dim_feedforward=combined_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attn_layers,
            enable_nested_tensor=False,  # Disabled for MPS compatibility
        )

        # Set attention for affiliation classification
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

        # Designation pointer
        self.desig_scorer = nn.Linear(hidden_dim * 2 + aff_embed_dim, 1)
        self.null_score = nn.Parameter(torch.zeros(1))

        self.num_affs = num_affs

    def forward(
        self,
        char_ids,
        token_mask,
        is_number=None,
        token_type=None,
        special_ids=None,
        aff_labels=None,
        desig_labels=None,
        aff_weight=None,
        aff_soft_labels=None,
    ):
        """
        Forward pass.

        Args:
            char_ids: [batch, seq_len, max_chars] character IDs for each token
            token_mask: [batch, seq_len] 1 for valid tokens, 0 for padding
            is_number: [batch, seq_len] 1 if token is a number, 0 otherwise
            token_type: [batch, seq_len] 0=word, 1=number, 2=space, 3=punct, 4=pad
            special_ids: [batch, seq_len] IDs for non-word tokens
            aff_labels: [batch] affiliation indices (for training)
            desig_labels: [batch] designation position (0 = no desig, 1+ = token index + 1)
            aff_soft_labels: [batch, num_affs] soft label distributions (optional, for soft label training)

        Returns:
            Dictionary with predictions and optionally losses
        """
        batch_size, seq_len, _ = char_ids.shape
        device = char_ids.device

        # Identify word vs non-word tokens
        is_word = (
            (token_type == 0)
            if token_type is not None
            else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        )

        # Get embeddings via CharacterCNN (blended with special embeddings below)
        if is_word.any():
            cnn_emb = self.char_cnn(char_ids)
        else:
            cnn_emb = torch.zeros(
                batch_size, seq_len, self.token_embed_dim, device=device
            )

        # Get embeddings for non-word tokens via lookup
        if special_ids is not None:
            special_emb = self.special_embed(special_ids)
        else:
            special_emb = torch.zeros(
                batch_size, seq_len, self.token_embed_dim, device=device
            )

        # Combine: use CNN for words, lookup for non-words
        is_word_expanded = is_word.unsqueeze(-1).float()
        token_emb = is_word_expanded * cnn_emb + (1 - is_word_expanded) * special_emb

        # Add is_number feature embedding
        if is_number is None:
            is_number = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )
        num_feature_emb = self.num_feature_embed(is_number)
        token_emb = torch.cat([token_emb, num_feature_emb], dim=-1)

        # Transformer encoder (stacked self-attention + FFN)
        # PyTorch expects src_key_padding_mask where True = ignore
        src_key_padding_mask = token_mask == 0
        token_emb_ctx = self.transformer_encoder(
            token_emb, src_key_padding_mask=src_key_padding_mask
        )

        # Affiliation classification via set attention pooling
        attn_scores = self.aff_attn(token_emb_ctx).squeeze(-1)
        attn_scores = attn_scores.masked_fill(token_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = (attn_weights * token_emb_ctx).sum(dim=1)
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

        # Score each position
        position_scores = self.desig_scorer(lstm_with_aff).squeeze(-1)

        # Mask: only number tokens can be designation
        valid_desig_mask = (is_number == 1) & (token_mask == 1)
        position_scores = position_scores.masked_fill(~valid_desig_mask, float("-inf"))

        # Prepend null score
        null_scores = self.null_score.expand(batch_size, 1)
        desig_scores = torch.cat([null_scores, position_scores], dim=1)
        desig_pred = desig_scores.argmax(dim=1)

        results = {
            "aff_logits": aff_logits,
            "aff_idx": aff_idx,
            "desig_scores": desig_scores,
            "desig_pred": desig_pred,
        }

        if desig_labels is not None and aff_labels is not None:
            desig_loss = F.cross_entropy(desig_scores, desig_labels)

            # Use soft labels if provided, otherwise hard labels
            if aff_soft_labels is not None:
                # Soft cross-entropy: -sum(target * log_softmax(logits))
                log_probs = F.log_softmax(aff_logits, dim=-1)
                aff_loss = -(aff_soft_labels * log_probs).sum(dim=-1).mean()
            else:
                aff_loss = F.cross_entropy(aff_logits, aff_labels, weight=aff_weight)

            results["desig_loss"] = desig_loss
            results["aff_loss"] = aff_loss
            results["total_loss"] = aff_loss + desig_loss

        return results


def create_desig_label(text: str, desig_num: str, max_len: int = MAX_TOKEN_LEN) -> int:
    """Create designation label for training.

    Returns:
        0 if no designation, otherwise token_position + 1
    """
    if not desig_num or desig_num == "N/A":
        return 0

    _, tokens, _, _ = tokenize_to_chars(text, max_tokens=max_len)
    desig_str = str(desig_num).lstrip("0") or "0"

    # Find the last occurrence of the designation number
    best_idx = None
    for i, token in enumerate(tokens[:max_len]):
        if token == desig_str:
            best_idx = i

    if best_idx is not None:
        return best_idx + 1

    return 0


def extract_desig_from_pred(
    text: str, desig_pred: int, max_len: int = MAX_TOKEN_LEN
) -> str:
    """Extract designation from model prediction."""
    if desig_pred == 0:
        return ""

    _, tokens, _, _ = tokenize_to_chars(text, max_tokens=max_len)
    token_idx = desig_pred - 1

    if token_idx < len(tokens):
        token = tokens[token_idx]
        if token.isdigit():
            return token

    return ""
