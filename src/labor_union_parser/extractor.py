"""Two-stage extractor for labor union parsing.

Stage 1: Union vs Non-union detection (contrastive similarity to union centroid)
Stage 2: Structured classifier + gazetteer scoring for field extraction
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .char_cnn import (
    CharacterCNN,
    tokenize_to_chars,
)
from .classifier import FIELDS, MAX_TOKENS, POINTER_FIELDS, StructuredClassifier
from .scoring import (
    _normalize_pointer_value,
    build_gazetteer_matrix,
    build_pointer_lookup,
    compute_record_features,
)
from .tokenizer import smart_truncate_nonspace


class AttentionPoolingEncoder(nn.Module):
    """Encoder with learned-query attention pooling for union detection."""

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

        self.num_embed = nn.Embedding(2, num_embed_dim)

        self.query = nn.Parameter(torch.randn(1, 1, self.input_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, char_ids, token_type, is_number):
        batch_size = char_ids.shape[0]

        char_emb = self.char_cnn(char_ids)
        is_number_mask = is_number.unsqueeze(-1).float()
        char_emb = char_emb * (1 - is_number_mask)

        num_feature_emb = self.num_embed(is_number)

        token_emb = torch.cat([char_emb, num_feature_emb], dim=-1)
        key_padding_mask = token_type == 4
        query = self.query.expand(batch_size, -1, -1)

        attn_out, _ = self.cross_attn(
            query=query,
            key=token_emb,
            value=token_emb,
            key_padding_mask=key_padding_mask,
        )

        pooled = attn_out.squeeze(1)
        proj = self.projector(pooled)
        return F.normalize(proj, p=2, dim=-1)


class Extractor:
    """Two-stage extractor for labor union names.

    Stage 1: Union vs Non-union detection using contrastive similarity.
    Stage 2: Structured classifier + gazetteer scoring for field extraction.
    """

    def __init__(
        self,
        device: str | None = None,
        union_threshold: float = 0.55,
    ):
        if device is None:
            self.device = torch.accelerator.current_accelerator(
                check_available=True
            ) or torch.device("cpu")
        else:
            self.device = device

        self.union_threshold = union_threshold
        self._load_models()

    def _load_models(self):
        """Load union detector and structured classifier."""
        weights_dir = Path(__file__).parent / "weights"

        # Stage 1: Union detector
        union_path = weights_dir / "union_detector.pt"
        union_checkpoint = torch.load(
            union_path, map_location=self.device, weights_only=False
        )

        char_cnn_union = CharacterCNN(embed_dim=64, char_embed_dim=16)
        self.union_encoder = AttentionPoolingEncoder(
            char_cnn_union, embed_dim=64, num_embed_dim=8, num_heads=4
        )
        self.union_encoder.load_state_dict(union_checkpoint["model_state_dict"])
        self.union_encoder.to(self.device)
        self.union_encoder.eval()

        self.union_centroid = union_checkpoint["union_centroid"].to(self.device)

        # Stage 2: Structured classifier + gazetteer (all bundled in one file)
        sc_path = weights_dir / "structured_classifier.pt"
        sc_ckpt = torch.load(sc_path, map_location=self.device, weights_only=False)

        self.field_vocabs = sc_ckpt["field_vocabs"]
        field_sizes = sc_ckpt["field_sizes"]

        self.structured_model = StructuredClassifier(
            field_sizes=field_sizes,
            d_model=sc_ckpt["d_model"],
            n_heads=4,
            n_layers=sc_ckpt["n_layers"],
            ff_dim=sc_ckpt["d_model"] * 2,
            dropout=0.0,
        )
        self.structured_model.load_state_dict(sc_ckpt["model_state"], strict=True)
        self.structured_model.to(self.device)
        self.structured_model.eval()

        # Gazetteer (bundled in checkpoint)
        fnum_to_records = sc_ckpt["gazetteer"]

        field_indices, field_known, record_fnums, records_list = build_gazetteer_matrix(
            fnum_to_records, self.field_vocabs
        )
        self.field_indices = {f: t.to(self.device) for f, t in field_indices.items()}
        self.field_known = {f: t.to(self.device) for f, t in field_known.items()}
        self.record_fnums = np.array(record_fnums)
        self.records_list = records_list
        self.n_records = len(record_fnums)

        # Pointer lookups: value_str -> record indices, and None record indices
        self.pointer_val_to_indices = {}
        self.pointer_none_indices = {}
        for f in POINTER_FIELDS:
            self.pointer_val_to_indices[f], self.pointer_none_indices[f] = (
                build_pointer_lookup(records_list, f)
            )

        # Inverse vocabs for decoding classification predictions
        self.inv_vocabs = {}
        for f in FIELDS:
            if f not in POINTER_FIELDS:
                self.inv_vocabs[f] = {i: v for v, i in self.field_vocabs[f].items()}

        # Learned scoring layer weights
        sw_path = weights_dir / "scoring_weights.pt"
        sw = torch.load(sw_path, map_location=self.device, weights_only=False)

        self.temperatures = sw["temperatures"]
        self.scoring_weight = sw["scoring_weight"].to(self.device)  # (12,)
        self.scoring_bias = sw["scoring_bias"]
        self.scoring_temperature = sw["scoring_temperature"]
        self.null_bias = sw["null_bias"]

    def _tokenize_for_union(self, texts, max_tokens=30):
        """Tokenize batch for union detector."""
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

    def _tokenize_for_structured(self, texts):
        """Tokenize batch for structured classifier using smart_truncate_nonspace."""
        all_char_ids = []
        all_masks = []
        all_token_strings = []

        for text in texts:
            tokens = smart_truncate_nonspace(text)
            all_char_ids.append([t["chars"] for t in tokens])
            all_masks.append([1 if t["token"] else 0 for t in tokens])
            all_token_strings.append([t["token"] for t in tokens])

        char_ids = torch.tensor(all_char_ids, dtype=torch.long, device=self.device)
        masks = torch.tensor(all_masks, dtype=torch.bool, device=self.device)
        return char_ids, masks, all_token_strings

    def _field_scores(self, log_probs, j):
        """Per-field confidence for the head's top prediction.

        log_probs: dict of (B, n_classes) tensors
        j: query index within the batch

        Returns dict of field -> probability of the argmax prediction.
        """
        scores = {}
        for f in ("union_name", "desig_name", "f_num", "desig_num", "prefix", "suffix"):
            scores[f] = log_probs[f][j].softmax(dim=-1).max().item()
        return scores

    def _score_gazetteer(self, log_probs, token_strings_batch):
        """Score all gazetteer records for a batch of queries.

        Assembles a 12-feature vector per (query, record) pair and applies
        the learned linear scoring layer.

        Feature layout: [lp_union, lp_desig, lp_fnum, lp_designum, lp_prefix, lp_suffix,
                         unk_union, unk_desig, unk_fnum, nf_designum, nf_prefix, nf_suffix]

        Returns list of (top_record_idx, top_score) per query.
        """
        B = len(token_strings_batch)
        R = self.n_records

        features = compute_record_features(
            log_probs,
            token_strings_batch,
            self.field_indices,
            self.field_known,
            self.pointer_val_to_indices,
            self.pointer_none_indices,
            R,
        )  # (B, R, 12)

        scores = (features * self.scoring_weight).sum(dim=2) + self.scoring_bias

        # Append null bias column → (B, R+1)
        null_col = torch.full((B, 1), self.null_bias, device=self.device)
        scores_with_null = torch.cat([scores, null_col], dim=1)

        # Best real record (always from the first R columns)
        top_indices = scores.argmax(dim=1)  # (B,)

        # Softmax over all columns (including null) for match_score
        scaled = scores_with_null / self.scoring_temperature
        top_scaled = scaled.gather(1, top_indices.unsqueeze(1)).squeeze(1)
        log_normalizers = torch.logsumexp(scaled, dim=1)  # (B,)
        match_probs = (top_scaled - log_normalizers).exp()  # (B,)

        # Check if null wins (argmax over R+1 columns points to last column)
        overall_winners = scores_with_null.argmax(dim=1)  # (B,)
        null_wins = (overall_winners == R).tolist()

        return list(zip(top_indices.tolist(), match_probs.tolist(), null_wins))

    def extract(self, text: str) -> dict:
        """Extract union fields from a single text."""
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str], batch_size: int = 256) -> list[dict]:
        """Extract from multiple texts."""
        if not texts:
            return []

        if len(texts) <= batch_size:
            return self._extract_batch_internal(texts)

        import itertools

        results = []
        for batch in itertools.batched(texts, batch_size):
            results.extend(self._extract_batch_internal(list(batch)))
        return results

    def _extract_batch_internal(self, texts: list[str]) -> list[dict]:
        """Internal batch processing."""
        # Stage 1: Union detection
        char_ids, token_type, is_number = self._tokenize_for_union(texts)

        with torch.no_grad():
            union_emb = self.union_encoder(char_ids, token_type, is_number)
            union_sims = torch.matmul(
                union_emb, self.union_centroid.unsqueeze(0).T
            ).squeeze(-1)

        union_sims_list = union_sims.cpu().tolist()

        # Stage 2: Structured classifier + gazetteer scoring (all texts)
        char_ids_sc, masks_sc, token_strings = self._tokenize_for_structured(texts)

        with torch.no_grad():
            logits = self.structured_model(char_ids_sc, masks_sc)
            log_probs = {
                f: F.log_softmax(logits[f] / self.temperatures[f], dim=-1)
                for f in FIELDS
            }

        matches = self._score_gazetteer(log_probs, token_strings)

        # Move log_probs to CPU to avoid per-element GPU→CPU sync in _field_scores
        log_probs_cpu = {f: lp.cpu() for f, lp in log_probs.items()}

        # Decode per-head argmax predictions
        head_preds = {}
        for f in ("union_name", "desig_name"):
            indices = log_probs_cpu[f].argmax(dim=-1)  # (B,)
            head_preds[f] = [self.inv_vocabs[f].get(idx.item(), "") for idx in indices]
        for f in ("desig_num", "prefix", "suffix"):
            indices = log_probs_cpu[f].argmax(dim=-1)  # (B,)
            preds = []
            for i, idx in enumerate(indices):
                pos = idx.item()
                if pos >= MAX_TOKENS:
                    preds.append("")
                else:
                    preds.append(
                        token_strings[i][pos] if pos < len(token_strings[i]) else ""
                    )
            head_preds[f] = preds

        results = []
        for i in range(len(texts)):
            rec_idx, match_score, null_won = matches[i]
            rec = self.records_list[rec_idx]
            field_scores = self._field_scores(log_probs_cpu, i)

            pred_union = head_preds["union_name"][i]
            pred_desig = head_preds["desig_name"][i]
            pred_desig_num = head_preds["desig_num"][i]
            pred_prefix = head_preds["prefix"][i]
            pred_suffix = head_preds["suffix"][i]

            # Detect conflicts between head predictions and matched record
            conflicts = []
            if pred_union != rec.get("union_name", ""):
                conflicts.append("union_name_mismatch")
            if pred_desig != rec.get("desig_name", ""):
                conflicts.append("desig_name_mismatch")

            rec_desig_num = _normalize_pointer_value(rec.get("desig_num", 0))
            if (pred_desig_num or None) != (rec_desig_num or None):
                conflicts.append("desig_num_mismatch")

            rec_prefix = _normalize_pointer_value(rec.get("prefix", 0))
            if (pred_prefix or None) != (rec_prefix or None):
                conflicts.append("prefix_mismatch")

            rec_suffix = _normalize_pointer_value(rec.get("suffix", ""))
            if (pred_suffix or None) != (rec_suffix or None):
                conflicts.append("suffix_mismatch")

            results.append(
                {
                    "is_union": union_sims_list[i] >= self.union_threshold,
                    "union_score": union_sims_list[i],
                    "union_name": pred_union,
                    "desig_name": pred_desig,
                    "desig_num": pred_desig_num,
                    "prefix": pred_prefix,
                    "suffix": pred_suffix,
                    "f_num": int(self.record_fnums[rec_idx]),
                    "match_found": not null_won,
                    "match_score": match_score,
                    "field_scores": field_scores,
                    "conflicts": conflicts,
                }
            )

        return results
