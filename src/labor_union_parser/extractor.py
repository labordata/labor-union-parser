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
)
from .tokenizer import smart_truncate_nonspace


class CrossAttentionEncoder(nn.Module):
    """Encoder with cross-attention pooling for union detection."""

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
        union_threshold: float = 0.9,
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
        self.union_encoder = CrossAttentionEncoder(
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
        self.scoring_bias = sw.get("scoring_bias", 0.0)
        self.scoring_temperature = sw.get("scoring_temperature", 1.0)

    def _tokenize_for_union(self, texts, max_tokens=80):
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

    def _field_scores(self, log_probs, j, query_toks, rec_idx):
        """Per-field probabilities for a single query–record pair.

        log_probs: dict of (B, n_classes) tensors
        j: query index within the batch

        Returns dict of field -> probability (or None if unknown/not-found).
        For pointer fields where the value appears at multiple token positions,
        probabilities are summed (logsumexp) across all matching positions.
        """
        scores = {}

        # Classification fields: single class index → exp(log_prob)
        for f in ("union_name", "desig_name", "f_num"):
            class_idx = self.field_indices[f][rec_idx].item()
            if self.field_known[f][rec_idx]:
                scores[f] = log_probs[f][j, class_idx].exp().item()
            else:
                scores[f] = None

        # Pointer fields: use first matching position (matches _score_gazetteer)
        tok_to_first_pos = {}
        for pos, tok in enumerate(query_toks):
            if tok and tok not in tok_to_first_pos:
                tok_to_first_pos[tok] = pos

        for f in ("desig_num", "prefix", "suffix"):
            rec = self.records_list[rec_idx]
            val = rec.get(f)
            normalized = _normalize_pointer_value(val)
            if normalized is None:
                scores[f] = log_probs[f][j, MAX_TOKENS].exp().item()
            else:
                pos = tok_to_first_pos.get(normalized)
                if pos is not None:
                    scores[f] = log_probs[f][j, pos].exp().item()
                else:
                    scores[f] = None

        return scores

    def _score_gazetteer(self, log_probs, token_strings_batch):
        """Score all gazetteer records for a batch of queries.

        Assembles a 12-feature vector per (query, record) pair and applies
        the learned linear scoring layer.

        Feature layout: [lp_union, lp_desig, lp_fnum, lp_designum, lp_prefix, lp_suffix,
                         unk_union, unk_desig, unk_fnum, nf_designum, nf_prefix, nf_suffix]

        Returns list of (top_record_idx, top_score) per query.
        """
        w = self.scoring_weight  # (12,)
        B = len(token_strings_batch)
        R = self.n_records

        # --- Build per-field log-prob arrays and indicators ---
        # Classification fields: lp where known, 0 where unknown
        # Unknown indicators: 1 where unknown, 0 where known
        cls_fields = ("union_name", "desig_name", "f_num")
        lp_cls = []  # will be features 0, 1, 2
        unk_cls = []  # will be features 6, 7, 8
        for f in cls_fields:
            field_lp = log_probs[f][:, self.field_indices[f]]  # (B, R)
            known = self.field_known[f]  # (R,)
            lp_cls.append(torch.where(known, field_lp, torch.zeros_like(field_lp)))
            unk_cls.append((~known).float().expand(B, -1))

        # Pointer fields: lp where found, 0 where not-found
        # Not-found indicators: 1 where not-found, 0 where found
        ptr_fields = ("desig_num", "prefix", "suffix")
        lp_ptr = []  # will be features 3, 4, 5
        nf_ptr = []  # will be features 9, 10, 11

        for f in ptr_fields:
            # Start with not-found for all records
            field_scores = torch.zeros(B, R, device=self.device)
            is_not_found = torch.ones(B, R, device=self.device)

            # Records with None value: use the "none" position log-prob
            none_idx = self.pointer_none_indices[f]
            if len(none_idx) > 0:
                field_scores[:, none_idx] = log_probs[f][:, MAX_TOKENS].unsqueeze(1)
                is_not_found[:, none_idx] = 0.0

            # Records with string values: scatter from query token positions
            val_to_idx = self.pointer_val_to_indices[f]
            all_batch_idx = []
            all_rec_idx = []
            all_positions = []

            for i, query_toks in enumerate(token_strings_batch):
                tok_to_pos = {}
                for pos, tok in enumerate(query_toks):
                    if tok and tok not in tok_to_pos:
                        tok_to_pos[tok] = pos
                for tok, pos in tok_to_pos.items():
                    rec_indices = val_to_idx.get(tok)
                    if rec_indices is not None:
                        n = len(rec_indices)
                        all_batch_idx.append(torch.full((n,), i, dtype=torch.long))
                        all_rec_idx.append(rec_indices)
                        all_positions.append((i, pos, n))

            if all_batch_idx:
                bi = torch.cat(all_batch_idx)
                ri = torch.cat(all_rec_idx).to(self.device)
                vals = torch.empty(bi.shape[0], device=self.device)
                offset = 0
                for i, pos, n in all_positions:
                    vals[offset : offset + n] = log_probs[f][i, pos]
                    offset += n
                field_scores[bi, ri] = vals
                is_not_found[bi, ri] = 0.0

            lp_ptr.append(field_scores)
            nf_ptr.append(is_not_found)

        # --- Compute score as dot product with weight vector ---
        # Instead of materializing (B, R, 12), compute weighted sum directly
        scores = torch.zeros(B, R, device=self.device)
        for i, feat in enumerate(lp_cls + lp_ptr + unk_cls + nf_ptr):
            scores += w[i] * feat
        scores += self.scoring_bias

        top_indices = scores.argmax(dim=1)  # (B,)
        scaled = scores / self.scoring_temperature
        top_scaled = scaled.gather(1, top_indices.unsqueeze(1)).squeeze(1)
        log_normalizers = torch.logsumexp(scaled, dim=1)  # (B,)
        match_probs = (top_scaled - log_normalizers).exp()  # (B,)

        return list(zip(top_indices.tolist(), match_probs.tolist()))

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

        results = []
        for i in range(len(texts)):
            rec_idx, match_score = matches[i]
            rec = self.records_list[rec_idx]
            field_scores = self._field_scores(log_probs, i, token_strings[i], rec_idx)
            results.append(
                {
                    "is_union": union_sims_list[i] >= self.union_threshold,
                    "union_score": union_sims_list[i],
                    "union_name": rec.get("union_name", ""),
                    "desig_name": rec.get("desig_name", ""),
                    "desig_num": str(rec.get("desig_num", 0) or ""),
                    "prefix": str(rec.get("prefix", 0) or ""),
                    "suffix": rec.get("suffix", ""),
                    "f_num": str(rec.get("f_num", "")),
                    "match_score": match_score,
                    "field_scores": field_scores,
                }
            )

        return results
