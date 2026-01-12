"""Three-stage contrastive extractor for labor union parsing.

Stage 1: Union vs Non-union detection (contrastive similarity to union centroid)
Stage 2: Affiliation classification via nearest centroid (distance > threshold = None/unrecognized)
Stage 3: Designation extraction using pointer network (with affiliation context)
"""

import functools
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .char_cnn import (
    SPECIAL_TOKEN_VOCAB,
    CharacterCNN,
    get_special_token_id,
    tokenize_to_chars,
)
from .conf import MAX_TOKENS


@functools.cache
def _load_fnum_lookup() -> dict:
    """Load the fnum lookup table from package data."""
    lookup_path = Path(__file__).parent / "weights" / "fnum_lookup.json"
    with open(lookup_path, "r") as f:
        return json.load(f)


def lookup_fnum(affiliation: str, designation: str) -> list[int]:
    """
    Look up filing numbers (fnum) for an affiliation and designation.

    Args:
        affiliation: Affiliation abbreviation (e.g., "SEIU", "IBT")
        designation: Local designation number as string (e.g., "1199")

    Returns:
        List of filing numbers. Empty list if no match found.
    """
    lookup = _load_fnum_lookup()
    key = f"{affiliation}|{designation}"
    return lookup.get(key, [])


class CrossAttentionEncoder(nn.Module):
    """Encoder with cross-attention pooling instead of mean pooling.

    Uses a learned query to attend over token embeddings, allowing the model
    to learn which tokens are most relevant for classification.
    """

    def __init__(
        self,
        char_cnn: CharacterCNN,
        embed_dim: int = 64,
        num_embed_dim: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        self.char_cnn = char_cnn
        self.char_embed_dim = char_cnn.embed_dim
        self.num_embed_dim = num_embed_dim
        self.input_dim = self.char_embed_dim + num_embed_dim

        # is_number embedding (0 = not number, 1 = number)
        self.num_embed = nn.Embedding(2, num_embed_dim)

        # Learned query for "what class is this?"
        self.query = nn.Parameter(torch.randn(1, 1, self.input_dim) * 0.02)

        # Cross-attention: query attends to token sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, char_ids, token_type, is_number, return_attention=False):
        batch_size = char_ids.shape[0]

        # Get token embeddings from CharCNN
        token_emb = self.char_cnn(char_ids)

        # Add is_number embedding
        num_emb = self.num_embed(is_number)
        token_emb = torch.cat([token_emb, num_emb], dim=-1)

        # Create padding mask (True = ignore)
        key_padding_mask = token_type == 4

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Cross-attention: query attends to tokens
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=token_emb,
            value=token_emb,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        # Remove sequence dimension and project
        pooled = attn_out.squeeze(1)
        proj = self.projector(pooled)
        normalized = F.normalize(proj, p=2, dim=-1)

        if return_attention:
            return normalized, attn_weights.squeeze(1)
        return normalized


class DesignationExtractor(nn.Module):
    """
    Standalone designation extractor using pointer network.

    Architecture:
    - CharCNN for word token embeddings
    - Embedding lookup for non-word tokens (numbers, punct, space)
    - is_number feature embedding
    - Transformer encoder for context
    - BiLSTM for sequence processing
    - Pointer scorer for position selection

    Takes predicted affiliation as input context to help select the right number.
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
        self.num_affs = num_affs

        # CharacterCNN for word tokens only
        self.char_cnn = CharacterCNN(
            embed_dim=token_embed_dim,
            char_embed_dim=char_embed_dim,
        )

        # Simple embedding for non-word tokens (numbers, punct, space)
        self.special_embed = nn.Embedding(
            len(SPECIAL_TOKEN_VOCAB), token_embed_dim, padding_idx=0
        )

        # Affiliation embedding (for context)
        self.aff_embed = nn.Embedding(num_affs, aff_embed_dim)

        # Learned embedding for "is_number" feature
        self.num_feature_embed = nn.Embedding(2, num_feature_dim)

        # Combined embedding dimension
        combined_dim = token_embed_dim + num_feature_dim

        # Transformer encoder for token context
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

        # BiLSTM for designation selection
        self.lstm = nn.LSTM(
            combined_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Designation pointer (takes LSTM output + affiliation embedding)
        self.desig_scorer = nn.Linear(hidden_dim * 2 + aff_embed_dim, 1)
        self.null_score = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        char_ids,
        token_mask,
        is_number=None,
        token_type=None,
        special_ids=None,
        aff_idx=None,
        desig_labels=None,
    ):
        """
        Forward pass.

        Args:
            char_ids: [batch, seq_len, max_chars] character IDs for each token
            token_mask: [batch, seq_len] 1 for valid tokens, 0 for padding
            is_number: [batch, seq_len] 1 if token is a number, 0 otherwise
            token_type: [batch, seq_len] 0=word, 1=number, 2=space, 3=punct, 4=pad
            special_ids: [batch, seq_len] IDs for non-word tokens
            aff_idx: [batch] affiliation indices (from Stage 2)
            desig_labels: [batch] designation position (0 = no desig, 1+ = token index + 1)

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

        # Get embeddings via CharacterCNN
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

        # Transformer encoder
        src_key_padding_mask = token_mask == 0
        token_emb_ctx = self.transformer_encoder(
            token_emb, src_key_padding_mask=src_key_padding_mask
        )

        # Get affiliation embedding for context
        if aff_idx is None:
            # Default to first affiliation if not provided
            aff_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
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

        # Prepend null score (position 0 = no designation)
        null_scores = self.null_score.expand(batch_size, 1)
        desig_scores = torch.cat([null_scores, position_scores], dim=1)
        desig_pred = desig_scores.argmax(dim=1)

        results = {
            "desig_scores": desig_scores,
            "desig_pred": desig_pred,
        }

        if desig_labels is not None:
            desig_loss = F.cross_entropy(desig_scores, desig_labels)
            results["desig_loss"] = desig_loss

        return results


def create_desig_label(text: str, desig_num: str, max_len: int = MAX_TOKENS) -> int:
    """Create designation label for training.

    Returns:
        0 if no designation, otherwise token_position + 1
    """
    if not desig_num:
        return 0

    _, tokens, _, _ = tokenize_to_chars(text, max_tokens=max_len)

    # Find the last occurrence of the designation number
    best_idx = None
    for i, token in enumerate(tokens[:max_len]):
        if token == desig_num:
            best_idx = i

    if best_idx is not None:
        return best_idx + 1

    return 0


def extract_desig_from_pred(
    text: str, desig_pred: int, max_len: int = MAX_TOKENS
) -> str:
    """Extract actual designation string from model prediction.

    Args:
        text: Original text
        desig_pred: Predicted position (0 = no desig, 1+ = token index + 1)
        max_len: Maximum token length

    Returns:
        Designation number string or empty string if not found
    """
    if desig_pred == 0:
        return ""

    _, tokens, _, _ = tokenize_to_chars(text, max_tokens=max_len)
    token_idx = desig_pred - 1

    if token_idx < len(tokens):
        token = tokens[token_idx]
        # Verify it looks like a number
        if token.isdigit() or (token and token[0].isdigit()):
            return token

    return ""


class Extractor:
    """
    Three-stage contrastive extractor for labor union names.

    Stage 1: Union vs Non-union detection using contrastive similarity.
    Stage 2: Affiliation classification via nearest centroid in contrastive space.
    Stage 3: Designation extraction using pointer network (with affiliation context).

    Example:
        >>> extractor = Extractor()
        >>> extractor.extract("SEIU Local 1199")
        {'is_union': True, 'affiliation': 'SEIU', 'designation': '1199', ...}
    """

    def __init__(
        self,
        device: str | None = None,
        union_threshold: float = 0.5,
        affiliation_threshold: float = 0.80,
    ):
        """
        Initialize the contrastive extractor.

        Args:
            device: Device to use (default: best available accelerator)
            union_threshold: Similarity threshold for union detection (default: 0.5).
                Texts with similarity >= threshold are classified as unions.
            affiliation_threshold: Similarity threshold for affiliation (default: 0.80).
                If similarity to nearest centroid < threshold, returns None (unrecognized).
        """
        if device is None:
            self.device = torch.accelerator.current_accelerator()
        else:
            self.device = device

        self.union_threshold = union_threshold
        self.affiliation_threshold = affiliation_threshold

        self._load_models()

    def _load_models(self):
        """Load all three stage models."""
        weights_dir = Path(__file__).parent / "weights"

        # Stage 1: Union detector (cross-attention)
        union_path = weights_dir / "union_detector.pt"
        union_checkpoint = torch.load(
            union_path, map_location=self.device, weights_only=False
        )

        char_cnn_union = CharacterCNN(embed_dim=64, char_embed_dim=16)
        self.union_encoder = CrossAttentionEncoder(
            char_cnn_union, embed_dim=64, num_embed_dim=8, num_heads=4
        )
        self.union_encoder.load_state_dict(union_checkpoint["model_state_dict"])
        self.union_encoder.to(self.device)
        self.union_encoder.eval()

        self.union_centroid = union_checkpoint["union_centroid"].to(self.device)

        # Stage 2: Affiliation classifier (cross-attention)
        aff_model_path = weights_dir / "contrastive_aff_model.pt"
        aff_centroids_path = weights_dir / "contrastive_aff_centroids.pt"

        aff_checkpoint = torch.load(
            aff_model_path, map_location=self.device, weights_only=False
        )

        char_cnn_aff = CharacterCNN(embed_dim=64, char_embed_dim=16)
        self.aff_encoder = CrossAttentionEncoder(
            char_cnn_aff, embed_dim=64, num_embed_dim=8, num_heads=4
        )
        self.aff_encoder.load_state_dict(aff_checkpoint["model_state_dict"])
        self.aff_encoder.to(self.device)
        self.aff_encoder.eval()

        self.aff_list = aff_checkpoint["aff_list"]
        self.aff_centroids = torch.load(aff_centroids_path, map_location=self.device)

        # Stage 3: Designation extractor (pointer network)
        desig_path = weights_dir / "designation_extractor.pt"
        desig_checkpoint = torch.load(
            desig_path, map_location=self.device, weights_only=False
        )

        self.desig_extractor = DesignationExtractor(
            num_affs=len(self.aff_list),
            token_embed_dim=64,
            hidden_dim=512,
            aff_embed_dim=64,
            num_attn_heads=4,
            num_feature_dim=16,
            char_embed_dim=16,
            num_attn_layers=3,
        )
        self.desig_extractor.load_state_dict(desig_checkpoint["model_state_dict"])
        self.desig_extractor.to(self.device)
        self.desig_extractor.eval()

    def _tokenize_batch(self, texts: list[str], max_tokens: int = 40):
        """Tokenize a batch of texts."""
        char_ids_list = []
        token_type_list = []
        is_number_list = []

        for text in texts:
            char_ids, _, is_number, token_type = tokenize_to_chars(
                text, max_tokens=max_tokens
            )
            char_ids_list.append(char_ids)
            token_type_list.append(token_type)
            is_number_list.append(is_number)

        return (
            torch.tensor(char_ids_list, dtype=torch.long, device=self.device),
            torch.tensor(token_type_list, dtype=torch.long, device=self.device),
            torch.tensor(is_number_list, dtype=torch.long, device=self.device),
        )

    def _tokenize_for_desig(self, texts: list[str], max_tokens: int = MAX_TOKENS):
        """Tokenize for designation extractor (includes special_ids)."""
        char_ids_list = []
        token_type_list = []
        is_number_list = []
        special_ids_list = []
        token_mask_list = []

        for text in texts:
            char_ids, tokens, is_number, token_type = tokenize_to_chars(
                text, max_tokens=max_tokens
            )
            char_ids_list.append(char_ids)
            token_type_list.append(token_type)
            is_number_list.append(is_number)

            # Build special_ids for non-word tokens
            special_ids = []
            for i, (tok, ttype) in enumerate(zip(tokens, token_type)):
                if ttype != 0:  # Not a word
                    special_ids.append(get_special_token_id(tok))
                else:
                    special_ids.append(0)  # Placeholder for words
            # Pad to max_tokens
            while len(special_ids) < max_tokens:
                special_ids.append(0)
            special_ids_list.append(special_ids[:max_tokens])

            # Token mask: 1 for valid tokens, 0 for padding
            mask = [1 if ttype != 4 else 0 for ttype in token_type]
            token_mask_list.append(mask)

        return {
            "char_ids": torch.tensor(
                char_ids_list, dtype=torch.long, device=self.device
            ),
            "token_type": torch.tensor(
                token_type_list, dtype=torch.long, device=self.device
            ),
            "is_number": torch.tensor(
                is_number_list, dtype=torch.long, device=self.device
            ),
            "special_ids": torch.tensor(
                special_ids_list, dtype=torch.long, device=self.device
            ),
            "token_mask": torch.tensor(
                token_mask_list, dtype=torch.long, device=self.device
            ),
        }

    def extract(self, text: str) -> dict:
        """
        Extract union status, affiliation, and designation from a text.

        Args:
            text: Input text (potential union name)

        Returns:
            Dictionary with:
                - is_union: Whether the text is a union
                - union_score: Similarity to union centroid
                - affiliation: Predicted affiliation (None if not union or unrecognized)
                - affiliation_unrecognized: True if is_union but affiliation unrecognized
                - designation: Extracted designation number
                - aff_score: Similarity to nearest affiliation centroid
        """
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str], batch_size: int = 256) -> list[dict]:
        """
        Extract from multiple texts.

        Args:
            texts: List of input texts
            batch_size: Maximum batch size for GPU processing (default 64)

        Returns:
            List of result dictionaries
        """
        if not texts:
            return []

        # Process in chunks to avoid memory issues with large batches
        if len(texts) > batch_size:
            import itertools

            results = []
            for batch in itertools.batched(texts, batch_size):
                results.extend(self._extract_batch_internal(list(batch)))
            return results

        return self._extract_batch_internal(texts)

    def _extract_batch_internal(self, texts: list[str]) -> list[dict]:
        """Internal batch processing (no chunking)."""
        # Stage 1: Union detection (using longer max_tokens for union detection)
        char_ids_union, token_type_union, is_number_union = self._tokenize_batch(
            texts, max_tokens=80
        )

        with torch.no_grad():
            union_emb = self.union_encoder(
                char_ids_union, token_type_union, is_number_union
            )
            union_sims = torch.matmul(
                union_emb, self.union_centroid.unsqueeze(0).T
            ).squeeze(-1)

        union_sims = union_sims.cpu().tolist()
        is_union_list = [sim >= self.union_threshold for sim in union_sims]

        # Find which texts are unions for stage 2
        union_indices = [i for i, is_union in enumerate(is_union_list) if is_union]
        union_texts = [texts[i] for i in union_indices]

        # Initialize results
        results = [None] * len(texts)

        # Non-unions get early return
        for i, (is_union, sim) in enumerate(zip(is_union_list, union_sims)):
            if not is_union:
                results[i] = {
                    "is_union": False,
                    "union_score": sim,
                    "affiliation": None,
                    "affiliation_unrecognized": False,
                    "designation": None,
                    "aff_score": None,
                }

        # Stage 2: Affiliation classification for unions
        if union_texts:
            char_ids_aff, token_type_aff, is_number_aff = self._tokenize_batch(
                union_texts, max_tokens=MAX_TOKENS
            )

            with torch.no_grad():
                aff_emb = self.aff_encoder(char_ids_aff, token_type_aff, is_number_aff)
                # Compute similarities to all centroids
                similarities = torch.matmul(aff_emb, self.aff_centroids.T)
                max_sims, max_indices = similarities.max(dim=1)

            max_indices_list = max_indices.cpu().tolist()
            max_sims_list = max_sims.cpu().tolist()

            # Determine affiliations
            affiliations = []
            aff_indices_for_desig = []  # Indices to pass to Stage 3
            for j in range(len(union_texts)):
                similarity = max_sims_list[j]
                pred_idx = max_indices_list[j]

                if similarity < self.affiliation_threshold:
                    affiliations.append(None)
                    # Use index 0 as fallback for unrecognized
                    aff_indices_for_desig.append(0)
                else:
                    affiliations.append(self.aff_list[pred_idx])
                    aff_indices_for_desig.append(pred_idx)

            # Stage 3: Designation extraction
            desig_batch = self._tokenize_for_desig(union_texts)
            aff_idx_tensor = torch.tensor(
                aff_indices_for_desig, dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                desig_out = self.desig_extractor(
                    char_ids=desig_batch["char_ids"],
                    token_mask=desig_batch["token_mask"],
                    is_number=desig_batch["is_number"],
                    token_type=desig_batch["token_type"],
                    special_ids=desig_batch["special_ids"],
                    aff_idx=aff_idx_tensor,
                )
                desig_preds = desig_out["desig_pred"].cpu().tolist()

            designations = [
                extract_desig_from_pred(text, pred)
                for text, pred in zip(union_texts, desig_preds)
            ]

            # Build results for union texts
            for j, orig_idx in enumerate(union_indices):
                results[orig_idx] = {
                    "is_union": True,
                    "union_score": union_sims[orig_idx],
                    "affiliation": affiliations[j],
                    "affiliation_unrecognized": affiliations[j] is None,
                    "designation": designations[j],
                    "aff_score": max_sims_list[j],
                }

        return results
