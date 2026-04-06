#!/usr/bin/env python3
"""Train record embeddings via CBOW + supervised contrastive loss.

Each FMCS filing record is treated like a "sentence" with fields as "words."
Union names appearing in similar contexts (same employers, cities, industries)
get similar embeddings.

Loss 1 (CBOW, all 539K records):
  Mask one field, predict its embedding from the remaining fields via InfoNCE.

Loss 2 (ArcFace cosine classifier, ~200K labeled records):
  Cosine similarity classifier with ~150 classes (distinct union_names), scale=30,
  Cross-entropy on scaled cosine logits.

Uses the production smart_truncate_nonspace tokenizer. Word tokens use a shared
vocab embedding; number tokens use digit-level embeddings (10 digits + position)
so that different local numbers get distinct representations.

Number tokens are stripped from the union_name field in both pathways so that
embeddings capture union identity rather than local number.
"""

import json
import random
import re
import sqlite3
from collections import Counter
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader, Dataset

from labor_union_parser.tokenizer import smart_truncate_nonspace

print = partial(print, flush=True)  # noqa: A001

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_NAMES = [
    "union_name",
    "employer",
    "employer_city",
    "employer_state",
    "industry",
    "naics",
    "bargaining_unit_size",
    "establishment_size",
]
NUM_FIELDS = len(FIELD_NAMES)
MAX_TOKENS_PER_FIELD = 10
NUM_BLOOM_HASHES = 3
BLOOM_TABLE_SIZE = 4096


# ---------------------------------------------------------------------------
# Tokenization & vocab
# ---------------------------------------------------------------------------

SIMPLE_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^\s\w]")


def tokenize_field_simple(value: str) -> list[dict]:
    """Fast regex tokenizer for non-union-name fields."""
    result = []
    for tok in SIMPLE_TOKEN_RE.findall(value.strip().lower()):
        is_num = tok.isdigit()
        if is_num:
            tok = tok.lstrip("0") or "0"
        result.append({"token": tok, "is_num": is_num})
    return result[:MAX_TOKENS_PER_FIELD]


_smart_cache: dict[str, list[dict]] = {}


def tokenize_field_smart(value: str) -> list[dict]:
    """Production tokenizer for union_name fields (cached)."""
    if value in _smart_cache:
        return _smart_cache[value]

    tokens = smart_truncate_nonspace(value, max_tokens=MAX_TOKENS_PER_FIELD)
    result = [
        {"token": t["token"], "is_num": bool(t["is_num"])} for t in tokens if t["token"]
    ]
    _smart_cache[value] = result
    return result


def _tokenize(field: str, value: str) -> list[dict]:
    """Tokenize a field value — smart tokenizer for union_name, simple for rest."""
    if field == "union_name":
        return tokenize_field_smart(value)
    return tokenize_field_simple(value)


def build_token_vocab(records: list[dict], min_count: int = 2) -> dict[str, int]:
    """Build a shared token vocabulary for word tokens across all fields.

    Numbers are excluded — they use digit embeddings instead.
    Index 0 = padding, 1 = UNK.
    """
    counter: Counter = Counter()
    for rec in records:
        for field in FIELD_NAMES:
            val = rec[field]
            if val:
                for tok in _tokenize(field, val):
                    if not tok["is_num"]:
                        counter[tok["token"]] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counter.most_common():
        if count >= min_count:
            vocab[token] = len(vocab)

    return vocab


def bloom_hash_ids(number_str: str) -> list[int]:
    """Hash a number string into NUM_BLOOM_HASHES table indices.

    Each number gets a unique (with high probability) set of indices
    into a shared embedding table. The embeddings at those indices
    are summed to produce the number's representation.
    """
    import hashlib

    ids = []
    for seed in range(NUM_BLOOM_HASHES):
        h = hashlib.md5(f"{seed}:{number_str}".encode()).hexdigest()
        ids.append(int(h, 16) % BLOOM_TABLE_SIZE)
    return ids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_f7_records(db_path: str) -> list[dict]:
    """Load all records from f7.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT union_name, employer, employer_city, employer_state,
                  industry, naics, bargaining_unit_size, establishment_size
           FROM f7"""
    ).fetchall()
    conn.close()

    records = []
    for row in rows:
        rec = {}
        for field in FIELD_NAMES:
            val = row[field]
            if val is None:
                rec[field] = ""
            elif isinstance(val, (int, float)):
                if val == 0:
                    rec[field] = ""
                else:
                    rec[field] = str(int(val))
            else:
                rec[field] = val.strip()
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RecordDataset(Dataset):
    """Token-indexed multi-field record dataset for CBOW loss.

    Stores per token position:
        token_ids:   [N, NUM_FIELDS, MAX_TOKENS_PER_FIELD] int32 — vocab ID (words)
        bloom_ids:   [N, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES] int16
        is_number:   [N, NUM_FIELDS, MAX_TOKENS_PER_FIELD] bool
        token_lens:  [N, NUM_FIELDS] int8 — tokens per field
    """

    def __init__(
        self,
        records: list[dict],
        token_vocab: dict[str, int],
    ):
        N = len(records)
        unk_id = token_vocab["<UNK>"]
        self.token_ids = np.zeros((N, NUM_FIELDS, MAX_TOKENS_PER_FIELD), dtype=np.int32)
        self.bloom_ids = np.zeros(
            (N, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES),
            dtype=np.int16,
        )
        self.is_number = np.zeros((N, NUM_FIELDS, MAX_TOKENS_PER_FIELD), dtype=np.bool_)
        self.token_lens = np.zeros((N, NUM_FIELDS), dtype=np.int8)

        for i, rec in enumerate(records):
            for j, field in enumerate(FIELD_NAMES):
                val = rec[field]
                if not val:
                    continue
                tokens = _tokenize(field, val)
                # Strip numbers from union_name field — embeddings should
                # capture union identity, not local number
                if field == "union_name":
                    tokens = [tok for tok in tokens if not tok["is_num"]]
                if not tokens:
                    continue
                n_tok = min(len(tokens), MAX_TOKENS_PER_FIELD)
                self.token_lens[i, j] = n_tok
                for t in range(n_tok):
                    tok = tokens[t]
                    if tok["is_num"]:
                        self.is_number[i, j, t] = True
                        self.bloom_ids[i, j, t] = bloom_hash_ids(tok["token"])
                    else:
                        self.token_ids[i, j, t] = token_vocab.get(tok["token"], unk_id)

        # present = has at least 1 token
        n_fields_present = (self.token_lens > 0).sum(axis=1)

        # Filter records with < 2 present fields (need 2+ for CBOW)
        valid = n_fields_present >= 2
        n_dropped = N - valid.sum()
        if n_dropped > 0:
            print(f"  Dropping {n_dropped} records with < 2 present fields")
            self.token_ids = self.token_ids[valid]
            self.bloom_ids = self.bloom_ids[valid]
            self.is_number = self.is_number[valid]
            self.token_lens = self.token_lens[valid]

        print(f"  CBOW records: {len(self)}")

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        token_ids = torch.from_numpy(self.token_ids[idx].astype(np.int64))
        bloom_ids = torch.from_numpy(self.bloom_ids[idx].astype(np.int64))
        is_number = torch.from_numpy(self.is_number[idx])
        token_lens = torch.from_numpy(self.token_lens[idx].astype(np.int64))
        present = token_lens > 0

        # Random field to mask
        present_indices = present.nonzero(as_tuple=True)[0]
        mask_idx = present_indices[torch.randint(len(present_indices), (1,))].item()

        return {
            "token_ids": token_ids,
            "bloom_ids": bloom_ids,
            "is_number": is_number,
            "token_lens": token_lens,
            "present": present,
            "mask_idx": mask_idx,
        }


class LabeledQueryDataset(Dataset):
    """Single-field dataset for ArcFace prototype loss.

    Tokenizes query texts as union_name field (numbers stripped) and pairs
    with union_name class labels.
    """

    def __init__(
        self,
        examples: list[dict],
        token_vocab: dict[str, int],
    ):
        unk_id = token_vocab["<UNK>"]
        N = len(examples)
        self.token_ids = np.zeros((N, MAX_TOKENS_PER_FIELD), dtype=np.int32)
        self.bloom_ids = np.zeros(
            (N, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES), dtype=np.int16
        )
        self.is_number = np.zeros((N, MAX_TOKENS_PER_FIELD), dtype=np.bool_)
        self.token_lens = np.zeros(N, dtype=np.int8)
        self.union_names = [ex["union_name"] for ex in examples]

        for i, ex in enumerate(examples):
            tokens = _tokenize("union_name", ex["query"])
            # Strip numbers — classify by union identity, not local number
            tokens = [tok for tok in tokens if not tok["is_num"]]
            if not tokens:
                continue
            n_tok = min(len(tokens), MAX_TOKENS_PER_FIELD)
            self.token_lens[i] = n_tok
            for t in range(n_tok):
                tok = tokens[t]
                self.token_ids[i, t] = token_vocab.get(tok["token"], unk_id)

        # Build union_name -> class index mapping
        unique_unions = sorted(set(self.union_names))
        self.union_to_idx = {un: i for i, un in enumerate(unique_unions)}
        self.n_classes = len(unique_unions)

        print(f"  ArcFace queries: {N}, Unions: {self.n_classes}")

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        token_ids = torch.from_numpy(self.token_ids[idx].astype(np.int64))
        bloom_ids = torch.from_numpy(self.bloom_ids[idx].astype(np.int64))
        is_number = torch.from_numpy(self.is_number[idx])
        token_len = int(self.token_lens[idx])
        union_label = self.union_to_idx[self.union_names[idx]]

        return {
            "token_ids": token_ids,
            "bloom_ids": bloom_ids,
            "is_number": is_number,
            "token_len": token_len,
            "union_label": union_label,
        }


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def build_rope_cache(
    seq_len: int, d_model: int, device: torch.device, base: float = 10000.0
):
    """Precompute cos/sin for rotary position embeddings.

    Returns cos, sin tensors of shape [seq_len, d_model].
    """
    half = d_model // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)  # [seq_len, half]
    # Duplicate each angle for the pair (x, y) → interleaved
    cos_vals = torch.cos(angles).repeat(1, 2)  # [seq_len, d_model]
    sin_vals = torch.sin(angles).repeat(1, 2)  # [seq_len, d_model]
    return cos_vals, sin_vals


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to x: [B, seq_len, d_model]."""
    # Rotate pairs: (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
    d = x.shape[-1]
    half = d // 2
    x_rot = torch.stack([-x[..., half:], x[..., :half]], dim=-1)
    x_rot = x_rot.reshape(x.shape)  # [-x1, x0] interleaved as [half:] then [:half]...
    # Simpler: just rotate pairs
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_h = cos[: x.shape[1], :half]
    sin_h = sin[: x.shape[1], :half]
    out1 = x1 * cos_h - x2 * sin_h
    out2 = x1 * sin_h + x2 * cos_h
    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RecordEmbeddingModel(nn.Module):
    """Record embedding model with CBOW + ArcFace prototype losses.

    Word tokens: shared vocab embedding lookup.
    Number tokens: bloom hash embedding (sum of k lookups from shared table).
    Within-field RoPE attention pools tokens into field embeddings.
    CBOW predicts masked field from mean-pooled context (no cross-field attention).
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 0.07,
        vocab_size: int = 1,
        n_classes: int = 1,
        arcface_scale: float = 30.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.arcface_scale = arcface_scale

        # Shared word token embedding (index 0 = pad)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.token_embed.weight, std=0.01)
        with torch.no_grad():
            self.token_embed.weight[0].zero_()

        # Bloom embedding table for numbers
        self.bloom_embed = nn.Embedding(BLOOM_TABLE_SIZE, d_model)
        nn.init.normal_(self.bloom_embed.weight, std=0.01)

        # Learned field-type embeddings (additive — for cross-field attention)
        self.field_type_embed = nn.Embedding(NUM_FIELDS, d_model)
        nn.init.normal_(self.field_type_embed.weight, std=0.01)

        # Within-field attention with RoPE: pool tokens into one field embedding
        self.token_q = nn.Linear(d_model, d_model)
        self.token_k = nn.Linear(d_model, d_model)
        self.token_v = nn.Linear(d_model, d_model)
        self.token_attn_out = nn.Linear(d_model, d_model)

        # CBOW projection: context mean -> predicted target embedding
        self.cbow_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        for m in self.cbow_proj:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        # Prototype-based cosine classifier for union_name
        self.prototypes = nn.Linear(d_model, n_classes, bias=False)
        nn.init.xavier_normal_(self.prototypes.weight)

    def encode_fields(self, token_ids, bloom_ids, is_number, token_lens):
        """Encode all fields for a batch.

        Word tokens → vocab embedding.
        Number tokens → bloom hash embedding (sum of k lookups).
        Within-field RoPE attention pools tokens per field.

        Args:
            token_ids:  [B, NUM_FIELDS, MAX_TOKENS_PER_FIELD]
            bloom_ids:  [B, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES]
            is_number:  [B, NUM_FIELDS, MAX_TOKENS_PER_FIELD] bool
            token_lens: [B, NUM_FIELDS]

        Returns:
            field_embs: [B, NUM_FIELDS, d_model]
        """
        B = token_ids.shape[0]
        device = token_ids.device
        F_MAX = MAX_TOKENS_PER_FIELD

        # --- Word token embeddings ---
        flat_word_ids = token_ids.reshape(-1)
        word_embs = self.token_embed(flat_word_ids)  # [B*NF*F_MAX, d_model]
        word_embs = word_embs.view(B, NUM_FIELDS, F_MAX, self.d_model)

        # --- Number token embeddings via bloom hashing (sparse) ---
        num_flat = is_number.reshape(-1)  # [B*NF*F_MAX]
        tok_embs = word_embs  # start with word embeddings everywhere
        if num_flat.any():
            num_indices = num_flat.nonzero(as_tuple=True)[0]
            bloom_at_nums = bloom_ids.reshape(-1, NUM_BLOOM_HASHES)[num_indices]
            num_embs = self.bloom_embed(bloom_at_nums).sum(dim=1)  # [K, d_model]
            tok_embs = tok_embs.reshape(-1, self.d_model).clone()
            tok_embs[num_indices] = num_embs
            tok_embs = tok_embs.view(B, NUM_FIELDS, F_MAX, self.d_model)

        # --- Within-field attention with RoPE: pool tokens per field ---
        flat_lens = token_lens.view(B * NUM_FIELDS)
        tok_embs_flat = tok_embs.view(B * NUM_FIELDS, F_MAX, self.d_model)

        # Skip attention for empty fields — only process non-empty ones
        nonempty = flat_lens > 0
        pooled = torch.zeros(B * NUM_FIELDS, self.d_model, device=device)

        if nonempty.any():
            active = tok_embs_flat[nonempty]  # [A, F_MAX, d_model]
            active_lens = flat_lens[nonempty]  # [A]

            # Q, K, V projections
            Q = self.token_q(active)  # [A, F_MAX, d_model]
            K = self.token_k(active)
            V = self.token_v(active)

            # Apply RoPE to Q and K
            rope_cos, rope_sin = build_rope_cache(F_MAX, self.d_model, device)
            Q = apply_rope(Q, rope_cos, rope_sin)
            K = apply_rope(K, rope_cos, rope_sin)

            # Scaled dot-product attention with padding mask
            scale = self.d_model**0.5
            scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # [A, F_MAX, F_MAX]

            # Mask: set padded key positions to -inf
            tok_range = torch.arange(F_MAX, device=device)
            tok_valid = tok_range.unsqueeze(0) < active_lens.unsqueeze(1)  # [A, F_MAX]

            attn_mask = ~tok_valid.unsqueeze(1)  # [A, 1, F_MAX]
            scores = scores.masked_fill(attn_mask, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            attn_out = torch.bmm(weights, V)  # [A, F_MAX, d_model]
            attn_out = self.token_attn_out(attn_out)

            # Mean-pool the attended token embeddings (masked)
            tok_mask_f = tok_valid.unsqueeze(-1).float()
            summed = (attn_out * tok_mask_f).sum(dim=1)
            denom = tok_mask_f.sum(dim=1).clamp(min=1)
            pooled[nonempty] = summed / denom

        # Reshape: [B, NUM_FIELDS, d_model]
        field_embs = pooled.view(B, NUM_FIELDS, self.d_model)

        # Add field-type embeddings (additive — clean separation for attention)
        field_type_ids = torch.arange(NUM_FIELDS, device=device)
        field_embs = field_embs + self.field_type_embed(field_type_ids).unsqueeze(0)

        return field_embs

    def encode_query(self, token_ids, bloom_ids, is_number, token_lens):
        """Encode single-field query texts (union_name only).

        Wraps inputs into the multi-field format with only field 0 populated,
        then returns the union_name embedding.

        Args:
            token_ids:  [B, MAX_TOKENS_PER_FIELD]
            bloom_ids:  [B, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES]
            is_number:  [B, MAX_TOKENS_PER_FIELD] bool
            token_lens: [B]

        Returns:
            union_embs: [B, d_model]
        """
        B = token_ids.shape[0]

        # Wrap into multi-field tensors with only field 0 populated
        full_token_ids = token_ids.new_zeros(B, NUM_FIELDS, MAX_TOKENS_PER_FIELD)
        full_token_ids[:, 0] = token_ids
        full_bloom_ids = bloom_ids.new_zeros(
            B, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES
        )
        full_bloom_ids[:, 0] = bloom_ids
        full_is_number = is_number.new_zeros(B, NUM_FIELDS, MAX_TOKENS_PER_FIELD)
        full_is_number[:, 0] = is_number
        full_token_lens = token_lens.new_zeros(B, NUM_FIELDS)
        full_token_lens[:, 0] = token_lens

        field_embs = self.encode_fields(
            full_token_ids, full_bloom_ids, full_is_number, full_token_lens
        )
        return field_embs[:, 0]  # [B, d_model]

    def cbow_loss(self, field_embs, present, mask_idx):
        """CBOW loss: predict masked field from context fields via InfoNCE.

        Mean-pools the raw context field embeddings (excluding the masked
        field) and projects to predict the target field embedding.
        """
        B = field_embs.shape[0]
        device = field_embs.device
        batch_range = torch.arange(B, device=device)

        # Target: field embedding of the masked field
        target_embs = field_embs[batch_range, mask_idx]

        # Context: mean-pool raw field embeddings (excluding masked field)
        context_mask = present.clone()
        context_mask[batch_range, mask_idx] = False
        context_mask_f = context_mask.unsqueeze(-1).float()

        context_sum = (field_embs * context_mask_f).sum(dim=1)
        context_count = context_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        context_mean = context_sum / context_count

        predicted = F.normalize(self.cbow_proj(context_mean), dim=1)
        target_embs = F.normalize(target_embs, dim=1)

        logits = torch.mm(predicted, target_embs.t()) / self.temperature
        labels = torch.arange(B, device=device)

        loss = F.cross_entropy(logits, labels)
        top1_acc = (logits.argmax(dim=1) == labels).float().mean()

        return loss, top1_acc

    def prototype_loss(self, union_embs, union_labels):
        """Cosine classifier loss on union_name embeddings.

        Args:
            union_embs: [B, d_model] — union_name field embeddings
            union_labels: [B] — class indices for parent union_name
        """
        # Cosine similarity: normalize both embeddings and prototypes
        embs_norm = F.normalize(union_embs, dim=1)
        w_norm = F.normalize(self.prototypes.weight, dim=1)
        cos_sim = embs_norm @ w_norm.t()  # [B, n_classes]

        logits = self.arcface_scale * cos_sim

        loss = F.cross_entropy(logits, union_labels)
        top1_acc = (logits.argmax(dim=1) == union_labels).float().mean()

        return loss, top1_acc


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class RecordEmbeddingModule(L.LightningModule):
    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 0.07,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        epochs: int = 20,
        vocab_size: int = 1,
        n_classes: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vocab_size", "n_classes"])
        self.model = RecordEmbeddingModel(
            d_model=d_model,
            temperature=temperature,
            vocab_size=vocab_size,
            n_classes=n_classes,
        )
        self._lr = lr
        self._weight_decay = weight_decay
        self._epochs = epochs

    def training_step(self, batch, batch_idx):
        cbow_batch, arcface_batch = batch["cbow"], batch["arcface"]

        # CBOW loss on multi-field records
        field_embs = self.model.encode_fields(
            cbow_batch["token_ids"],
            cbow_batch["bloom_ids"],
            cbow_batch["is_number"],
            cbow_batch["token_lens"],
        )
        cbow_loss, cbow_acc = self.model.cbow_loss(
            field_embs, cbow_batch["present"], cbow_batch["mask_idx"]
        )

        # ArcFace loss on labeled queries
        union_embs = self.model.encode_query(
            arcface_batch["token_ids"],
            arcface_batch["bloom_ids"],
            arcface_batch["is_number"],
            arcface_batch["token_len"],
        )
        proto_loss, proto_acc = self.model.prototype_loss(
            union_embs, arcface_batch["union_label"]
        )

        loss = cbow_loss + proto_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/cbow_loss", cbow_loss)
        self.log("train/proto_loss", proto_loss)
        self.log("train/cbow_acc", cbow_acc, prog_bar=True)
        self.log("train/proto_acc", float(proto_acc), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        cbow_batch, arcface_batch = batch["cbow"], batch["arcface"]

        field_embs = self.model.encode_fields(
            cbow_batch["token_ids"],
            cbow_batch["bloom_ids"],
            cbow_batch["is_number"],
            cbow_batch["token_lens"],
        )
        cbow_loss, cbow_acc = self.model.cbow_loss(
            field_embs, cbow_batch["present"], cbow_batch["mask_idx"]
        )

        union_embs = self.model.encode_query(
            arcface_batch["token_ids"],
            arcface_batch["bloom_ids"],
            arcface_batch["is_number"],
            arcface_batch["token_len"],
        )
        proto_loss, proto_acc = self.model.prototype_loss(
            union_embs, arcface_batch["union_label"]
        )

        loss = cbow_loss + proto_loss

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/cbow_loss", cbow_loss, sync_dist=True)
        self.log("val/proto_loss", proto_loss, sync_dist=True)
        self.log("val/cbow_acc", cbow_acc, prog_bar=True, sync_dist=True)
        self.log("val/proto_acc", float(proto_acc), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._epochs
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train record embeddings (CBOW + ArcFace)"
    )
    parser.add_argument("--db", default="f7.db", help="Path to f7.db")
    parser.add_argument(
        "--training-examples",
        default="training/data/training_examples.json",
        help="Path to training_examples.json",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--save", default="training/data/record_embeddings.pt")
    parser.add_argument(
        "--resume", default=None, help="Checkpoint to resume from (partial load)"
    )
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load CBOW data (f7.db multi-field records)
    print("Loading f7 records...")
    records = load_f7_records(args.db)
    print(f"  {len(records)} records loaded")

    print("Building token vocabulary (words only, numbers use bloom hashing)...")
    token_vocab = build_token_vocab(records, min_count=args.min_token_count)
    print(f"  {len(token_vocab)} word tokens (min_count={args.min_token_count})")

    print("Building CBOW dataset...")
    cbow_dataset = RecordDataset(records, token_vocab)

    # Load ArcFace data (training_examples.json)
    print("Loading training examples...")
    with open(args.training_examples) as f:
        all_examples = json.load(f)

    # Filter out f_num=-100 (catch-all "unknown local" — not a real class)
    all_examples = [ex for ex in all_examples if ex["f_num"] != -100]

    train_examples = [ex for ex in all_examples if ex["split"] == "train"]
    val_examples = [ex for ex in all_examples if ex["split"] == "val"]
    print(
        f"  {len(train_examples)} train, {len(val_examples)} val (after filtering f_num=-100)"
    )

    print("Building ArcFace datasets...")
    # Build shared union_to_idx from all examples (train + val need same class mapping)
    all_unions = sorted(set(ex["union_name"] for ex in all_examples))
    shared_union_to_idx = {un: i for i, un in enumerate(all_unions)}
    n_classes = len(shared_union_to_idx)

    arcface_train = LabeledQueryDataset(train_examples, token_vocab)
    arcface_val = LabeledQueryDataset(val_examples, token_vocab)
    # Override per-dataset mappings with shared mapping
    arcface_train.union_to_idx = shared_union_to_idx
    arcface_train.n_classes = n_classes
    arcface_val.union_to_idx = shared_union_to_idx
    arcface_val.n_classes = n_classes

    # CBOW train/val split (simple random)
    N = len(cbow_dataset)
    indices = list(range(N))
    random.shuffle(indices)
    n_val = int(N * 0.05)
    cbow_train = torch.utils.data.Subset(cbow_dataset, indices[n_val:])
    cbow_val = torch.utils.data.Subset(cbow_dataset, indices[:n_val])
    print(f"  CBOW train: {len(cbow_train)}, val: {len(cbow_val)}")

    # Combined dataloaders — cycle shorter one
    train_loader = CombinedLoader(
        {
            "cbow": DataLoader(
                cbow_train, batch_size=args.batch_size, shuffle=True, num_workers=0
            ),
            "arcface": DataLoader(
                arcface_train, batch_size=args.batch_size, shuffle=True, num_workers=0
            ),
        },
        mode="max_size_cycle",
    )
    val_loader = CombinedLoader(
        {
            "cbow": DataLoader(
                cbow_val, batch_size=args.batch_size, shuffle=False, num_workers=0
            ),
            "arcface": DataLoader(
                arcface_val, batch_size=args.batch_size, shuffle=False, num_workers=0
            ),
        },
        mode="max_size_cycle",
    )

    # Model
    module = RecordEmbeddingModule(
        d_model=args.d_model,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        vocab_size=len(token_vocab),
        n_classes=n_classes,
    )
    # Resume from checkpoint (partial load)
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        old_sd = ckpt["state_dict"]
        # Strip "model." prefix from lightning checkpoints
        old_sd = {k.removeprefix("model."): v for k, v in old_sd.items()}
        # Handle prototype size mismatch (new f_nums added)
        if "prototypes.weight" in old_sd:
            old_proto = old_sd["prototypes.weight"]
            new_proto = module.model.prototypes.weight.data
            if old_proto.shape != new_proto.shape:
                n_copy = min(old_proto.shape[0], new_proto.shape[0])
                new_proto[:n_copy] = old_proto[:n_copy]
                print(
                    f"  Prototypes: copied {n_copy}/{new_proto.shape[0]} "
                    f"(was {old_proto.shape[0]})"
                )
                del old_sd["prototypes.weight"]
        missing, unexpected = module.model.load_state_dict(old_sd, strict=False)
        if missing:
            print(f"  Missing keys (reinitialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

    param_count = sum(p.numel() for p in module.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Train
    trainer = L.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        default_root_dir="training/data/lightning_logs",
        accelerator="auto",
        enable_progress_bar=True,
    )
    trainer.fit(module, train_loader, val_loader)

    # Save checkpoint
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.model.state_dict(),
            "hparams": {
                "d_model": args.d_model,
                "temperature": args.temperature,
            },
            "token_vocab": token_vocab,
            "union_to_idx": shared_union_to_idx,
        },
        save_path,
    )
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
