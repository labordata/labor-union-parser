"""Two-stage extractor for labor union parsing.

Stage 1: Union vs Non-union detection (contrastive similarity to union centroid)
Stage 2: Structured classifier + gazetteer scoring for field extraction
"""

import math
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
    POINTER_NOT_FOUND_LOG_PROB,
    build_gazetteer_matrix,
    build_pointer_lookup,
)
from .tokenizer import smart_truncate_nonspace


class CrossAttentionEncoder(nn.Module):
    """Encoder with cross-attention pooling for union detection.

    Uses frozen random embeddings for numbers to make them orthogonal.
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
        union_threshold: float = 0.5,
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
        self.structured_model.load_state_dict(sc_ckpt["model_state"], strict=False)
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

        # Pointer lookups
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

        # Per-record f_num weighting (bundled in checkpoint)
        fnum_train_counts = sc_ckpt["fnum_train_counts"]

        base_w, floor_w, sat = 0.6, 0.1, 16
        self.fnum_weights = np.zeros(self.n_records, dtype=np.float32)
        for i, fnum in enumerate(record_fnums):
            n = fnum_train_counts.get(str(fnum), 0)
            self.fnum_weights[i] = floor_w + (base_w - floor_w) * min(
                1, math.log1p(n) / math.log1p(sat)
            )
        self.fnum_weights_t = torch.tensor(self.fnum_weights, device=self.device)

    def _tokenize_for_union(self, texts, max_tokens=80):
        """Tokenize batch for union detector."""
        char_ids_list = []
        token_type_list = []
        is_number_list = []

        for text in texts:
            char_ids, _, is_number, token_type, _ = tokenize_to_chars(
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

    def _score_gazetteer(self, log_probs, token_strings_batch):
        """Score all gazetteer records for a batch of queries.

        Returns list of (top_record_idx, top_score) per query.
        """
        results = []
        for i in range(len(token_strings_batch)):
            scores = torch.zeros(self.n_records, device=self.device)

            for f in FIELDS:
                if f == "f_num":
                    continue
                if f not in POINTER_FIELDS:
                    field_lp = log_probs[f][i][self.field_indices[f]]
                    vocab_size = log_probs[f].shape[-1]
                    floor_lp = -math.log(vocab_size)
                    field_lp = torch.where(self.field_known[f], field_lp, floor_lp)
                    scores += field_lp
                else:
                    query_toks = token_strings_batch[i]
                    lp = log_probs[f][i]

                    tok_to_pos = {}
                    for pos, tok in enumerate(query_toks):
                        if tok and tok not in tok_to_pos:
                            tok_to_pos[tok] = pos

                    field_scores = torch.full(
                        (self.n_records,),
                        POINTER_NOT_FOUND_LOG_PROB[f],
                        device=self.device,
                    )

                    none_idx = self.pointer_none_indices[f]
                    if len(none_idx) > 0:
                        field_scores[none_idx] = lp[MAX_TOKENS]

                    val_to_idx = self.pointer_val_to_indices[f]
                    for tok, pos in tok_to_pos.items():
                        rec_indices = val_to_idx.get(tok)
                        if rec_indices is not None:
                            field_scores[rec_indices] = lp[pos]

                    scores += field_scores

            # Blend with f_num
            fnum_lp = log_probs["f_num"][i][self.field_indices["f_num"]]
            fnum_vocab_size = log_probs["f_num"].shape[-1]
            fnum_floor = -math.log(fnum_vocab_size)
            fnum_lp = torch.where(self.field_known["f_num"], fnum_lp, fnum_floor)

            w = self.fnum_weights_t
            scores = (1 - w) * scores + w * fnum_lp

            top_idx = scores.argmax().item()
            top_score = scores[top_idx].item()
            results.append((top_idx, top_score))

        return results

    def extract(self, text: str) -> dict:
        """Extract union fields from a single text."""
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str], batch_size: int = 256) -> list[dict]:
        """Extract from multiple texts."""
        if not texts:
            return []

        if len(texts) > batch_size:
            import itertools

            results = []
            for batch in itertools.batched(texts, batch_size):
                results.extend(self._extract_batch_internal(list(batch)))
            return results

        return self._extract_batch_internal(texts)

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
        is_union_list = [sim >= self.union_threshold for sim in union_sims_list]

        union_indices = [i for i, is_union in enumerate(is_union_list) if is_union]
        union_texts = [texts[i] for i in union_indices]

        # Initialize results
        results = [None] * len(texts)

        # Non-unions
        for i, (is_union, sim) in enumerate(zip(is_union_list, union_sims_list)):
            if not is_union:
                results[i] = {
                    "is_union": False,
                    "union_score": sim,
                    "union_name": "",
                    "desig_name": "",
                    "desig_num": "",
                    "prefix": "",
                    "suffix": "",
                    "f_num": "",
                    "match_score": "",
                }

        # Stage 2: Structured classifier + gazetteer scoring
        if union_texts:
            char_ids_sc, masks_sc, token_strings = self._tokenize_for_structured(
                union_texts
            )

            with torch.no_grad():
                logits = self.structured_model(char_ids_sc, masks_sc)
                log_probs = {f: F.log_softmax(logits[f], dim=-1) for f in FIELDS}

            matches = self._score_gazetteer(log_probs, token_strings)

            for j, orig_idx in enumerate(union_indices):
                rec_idx, score = matches[j]
                rec = self.records_list[rec_idx]
                results[orig_idx] = {
                    "is_union": True,
                    "union_score": union_sims_list[orig_idx],
                    "union_name": rec.get("union_name", ""),
                    "desig_name": rec.get("desig_name", ""),
                    "desig_num": str(rec.get("desig_num", 0) or ""),
                    "prefix": str(rec.get("prefix", 0) or ""),
                    "suffix": rec.get("suffix", ""),
                    "f_num": str(rec.get("f_num", "")),
                    "match_score": f"{score:.4f}",
                }

        return results
