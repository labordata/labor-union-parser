#!/usr/bin/env python3
"""Train record embeddings via CBOW + supervised contrastive loss.

Each FMCS filing record is treated like a "sentence" with fields as "words."
Union names appearing in similar contexts (same employers, cities, industries)
get similar embeddings.

Loss 1 (CBOW, all 539K records):
  Mask one field, predict its embedding from the remaining fields via InfoNCE.

Loss 2 (ArcFace cosine classifier, ~200K labeled records):
  Cosine similarity classifier with ~8K classes (distinct f_nums), scale=30,
  no margin. Cross-entropy on scaled cosine logits.

Uses the production smart_truncate_nonspace tokenizer. Word tokens use a shared
vocab embedding; number tokens use digit-level embeddings (10 digits + position)
so that different local numbers get distinct representations.
"""

import csv
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


def load_fnum_labels(labeled_csv: str) -> dict[str, int]:
    """Load union_name -> f_num mapping from labeled_data.csv."""
    text_to_fnum = {}
    with open(labeled_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            fnum = row.get("f_num", "").strip()
            if not fnum:
                continue
            try:
                fnum_int = int(float(fnum))
            except (ValueError, TypeError):
                continue
            text_to_fnum[text.lower()] = fnum_int

    return text_to_fnum


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RecordDataset(Dataset):
    """Token-indexed record dataset.

    Stores per token position:
        token_ids:   [N, NUM_FIELDS, MAX_TOKENS_PER_FIELD] int32 — vocab ID (words)
        bloom_ids:   [N, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES] int16
        is_number:   [N, NUM_FIELDS, MAX_TOKENS_PER_FIELD] bool
        token_lens:  [N, NUM_FIELDS] int8 — tokens per field
        f_num:       [N] int32 (-1 if unlabeled)
    """

    def __init__(
        self,
        records: list[dict],
        text_to_fnum: dict[str, int],
        token_vocab: dict[str, int],
        distinct_texts: list[str] | None = None,
        text_to_text_idx: dict[str, int] | None = None,
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
        self.f_num = np.full(N, -1, dtype=np.int32)
        self.text_idx = np.full(N, -1, dtype=np.int32)

        for i, rec in enumerate(records):
            for j, field in enumerate(FIELD_NAMES):
                val = rec[field]
                if not val:
                    continue
                tokens = _tokenize(field, val)
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

            union_name = rec["union_name"].strip()
            union_name_lower = union_name.lower()
            if union_name_lower in text_to_fnum:
                self.f_num[i] = text_to_fnum[union_name_lower]
            if text_to_text_idx is not None and union_name in text_to_text_idx:
                self.text_idx[i] = text_to_text_idx[union_name]

        # present = has at least 1 token
        n_fields_present = (self.token_lens > 0).sum(axis=1)

        # Filter records with < 2 present fields
        valid = n_fields_present >= 2
        n_dropped = N - valid.sum()
        if n_dropped > 0:
            print(f"  Dropping {n_dropped} records with < 2 present fields")
            self.token_ids = self.token_ids[valid]
            self.bloom_ids = self.bloom_ids[valid]
            self.is_number = self.is_number[valid]
            self.token_lens = self.token_lens[valid]
            self.f_num = self.f_num[valid]
            self.text_idx = self.text_idx[valid]

        self.n_labeled = int((self.f_num >= 0).sum())
        unique_fnums = sorted(set(self.f_num[self.f_num >= 0].tolist()))
        self.fnum_to_idx = {fn: i for i, fn in enumerate(unique_fnums)}
        self.n_classes = len(unique_fnums)

        print(
            f"  Records: {len(self)}, Labeled: {self.n_labeled}, "
            f"F_nums: {self.n_classes}"
        )

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

        fn = self.f_num[idx]
        fnum_label = self.fnum_to_idx.get(int(fn), -1) if fn >= 0 else -1

        return {
            "token_ids": token_ids,
            "bloom_ids": bloom_ids,
            "is_number": is_number,
            "token_lens": token_lens,
            "present": present,
            "mask_idx": mask_idx,
            "fnum_label": fnum_label,
            "text_idx": self.text_idx[idx],
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
    Cross-field attention models field interactions.
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

        # Cross-field attention: model interactions between fields
        self.field_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

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

        # Prototype-based cosine classifier for f_num
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

    def cross_attend(self, field_embs, field_pad_mask):
        """Apply cross-field attention with the given padding mask."""
        # Ensure at least one field is unmasked
        all_empty = ~(~field_pad_mask).any(dim=1)
        safe_mask = field_pad_mask.clone()
        safe_mask[all_empty, 0] = False

        out, _ = self.field_attn(
            field_embs,
            field_embs,
            field_embs,
            key_padding_mask=safe_mask,
        )
        return out

    def cbow_loss(self, field_embs, present, mask_idx):
        """CBOW loss: predict masked field from context fields via InfoNCE.

        Cross-field attention is applied here with the masked field excluded
        from keys/values, so context embeddings can't leak target info.
        The target embedding uses the pre-attention field embedding.
        """
        B = field_embs.shape[0]
        device = field_embs.device
        batch_range = torch.arange(B, device=device)

        # Target: pre-attention embedding of the masked field
        target_embs = field_embs[batch_range, mask_idx]

        # Context: cross-attend with masked field excluded from keys/values
        context_pad_mask = ~present.clone()
        context_pad_mask[batch_range, mask_idx] = True  # mask out the target

        context_embs = self.cross_attend(field_embs, context_pad_mask)

        # Mean-pool context field embeddings (excluding masked field)
        context_mask = present.clone()
        context_mask[batch_range, mask_idx] = False
        context_mask_f = context_mask.unsqueeze(-1).float()

        context_sum = (context_embs * context_mask_f).sum(dim=1)
        context_count = context_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        context_mean = context_sum / context_count

        predicted = F.normalize(self.cbow_proj(context_mean), dim=1)
        target_embs = F.normalize(target_embs, dim=1)

        logits = torch.mm(predicted, target_embs.t()) / self.temperature
        labels = torch.arange(B, device=device)

        loss = F.cross_entropy(logits, labels)
        top1_acc = (logits.argmax(dim=1) == labels).float().mean()

        return loss, top1_acc

    def prototype_loss(self, field_embs, fnum_labels):
        """Cosine classifier loss on union_name field embeddings.

        Uses pre-cross-attention field embeddings so the union_name embedding
        is a standalone representation (within-field attention only).
        """
        labeled_mask = fnum_labels >= 0
        n_labeled = labeled_mask.sum().item()
        if n_labeled < 2:
            return torch.tensor(0.0, device=field_embs.device), 0, 0.0

        union_embs = field_embs[labeled_mask, 0]  # [N_lab, d_model]
        labels = fnum_labels[labeled_mask]

        # Cosine similarity: normalize both embeddings and prototypes
        embs_norm = F.normalize(union_embs, dim=1)
        w_norm = F.normalize(self.prototypes.weight, dim=1)
        logits = self.arcface_scale * (embs_norm @ w_norm.t())

        loss = F.cross_entropy(logits, labels)
        top1_acc = (logits.argmax(dim=1) == labels).float().mean()

        return loss, n_labeled, top1_acc


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
        repulsive_margin: float = 0.2,
        repulsive_weight: float = 1.0,
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
        self._repulsive_margin = repulsive_margin
        self._repulsive_weight = repulsive_weight
        self._repulsive_enabled = False

    def setup_repulsive_mining(
        self,
        distinct_texts: list[str],
        token_vocab: dict[str, int],
        k_neighbors: int = 8,
    ):
        """Prepare data for number-overlap repulsive loss.

        For each distinct union_name text, extract number tokens and
        pre-tokenize for embedding computation. Between epochs, find k-NN
        and precompute which neighbor pairs have non-overlapping numbers.
        """
        N = len(distinct_texts)
        unk_id = token_vocab["<UNK>"]

        token_ids = np.zeros((N, NUM_FIELDS, MAX_TOKENS_PER_FIELD), dtype=np.int64)
        bloom_ids = np.zeros(
            (N, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES), dtype=np.int64
        )
        is_number = np.zeros((N, NUM_FIELDS, MAX_TOKENS_PER_FIELD), dtype=np.bool_)
        token_lens = np.zeros((N, NUM_FIELDS), dtype=np.int64)

        # Extract number token sets for each text
        number_sets: list[frozenset[str]] = []
        for i, text in enumerate(distinct_texts):
            tokens = _tokenize("union_name", text)
            n_tok = min(len(tokens), MAX_TOKENS_PER_FIELD)
            token_lens[i, 0] = n_tok
            nums = set()
            for t in range(n_tok):
                tok = tokens[t]
                if tok["is_num"]:
                    is_number[i, 0, t] = True
                    bloom_ids[i, 0, t] = bloom_hash_ids(tok["token"])
                    nums.add(tok["token"])
                else:
                    token_ids[i, 0, t] = token_vocab.get(tok["token"], unk_id)
            number_sets.append(frozenset(nums))

        self._hn_token_ids = torch.from_numpy(token_ids)
        self._hn_bloom_ids = torch.from_numpy(bloom_ids)
        self._hn_is_number = torch.from_numpy(is_number)
        self._hn_token_lens = torch.from_numpy(token_lens)
        self._hn_number_sets = number_sets
        self._hn_k = k_neighbors
        self._hn_bank = None  # [N, d_model]
        # Repulsive pairs: list of (i, j) where i,j are text indices that are
        # nearest neighbors with non-overlapping numbers
        self._repulsive_pairs = None  # [P, 2] tensor
        self._repulsive_enabled = True

        n_with_nums = sum(1 for s in number_sets if s)
        print(
            f"  Repulsive mining: {N} distinct texts, {n_with_nums} with numbers, "
            f"k={k_neighbors} neighbors"
        )

    @torch.no_grad()
    def _update_repulsive_pairs(self):
        """Recompute embeddings, find k-NN, identify repulsive pairs."""
        device = next(self.parameters()).device
        N = len(self._hn_token_ids)
        batch_size = 512
        embeddings = []

        self.model.eval()
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            field_embs = self.model.encode_fields(
                self._hn_token_ids[start:end].to(device),
                self._hn_bloom_ids[start:end].to(device),
                self._hn_is_number[start:end].to(device),
                self._hn_token_lens[start:end].to(device),
            )
            union_emb = F.normalize(field_embs[:, 0, :], dim=-1)
            embeddings.append(union_emb.cpu())
        self.model.train()

        self._hn_bank = torch.cat(embeddings, dim=0)  # [N, d_model]

        # Find k nearest neighbors via chunked cosine similarity
        k = self._hn_k
        emb_np = self._hn_bank.numpy().astype(np.float32)
        pairs = []
        chunk = 2000
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            sims = emb_np[start:end] @ emb_np.T  # [chunk, N]
            for i in range(end - start):
                sims[i, start + i] = -2.0  # exclude self
            top_k_idx = np.argpartition(sims, -k, axis=1)[:, -k:]

            for i in range(end - start):
                idx_i = start + i
                nums_i = self._hn_number_sets[idx_i]
                if not nums_i:
                    continue  # no numbers → no repulsion
                for j in top_k_idx[i]:
                    nums_j = self._hn_number_sets[j]
                    if not nums_j:
                        continue
                    # Both have numbers, check for overlap
                    if nums_i.isdisjoint(nums_j):
                        pairs.append((idx_i, int(j)))

        if pairs:
            self._repulsive_pairs = torch.tensor(pairs, dtype=torch.long)
        else:
            self._repulsive_pairs = None

        # Move bank to device
        self._hn_bank = self._hn_bank.to(device)

        n_pairs = len(pairs) if pairs else 0
        print(f"  Repulsive pairs: {n_pairs} (from {N} texts, k={k})")

    def on_train_epoch_start(self):
        """Update repulsive pairs at the start of each epoch."""
        if self._repulsive_enabled:
            self._update_repulsive_pairs()

    def repulsive_loss(self, field_embs, text_idx):
        """Repulsive loss: push apart nearest neighbors with non-overlapping numbers.

        For each batch record, look up its precomputed repulsive pairs.
        Use the live batch embedding for the anchor and the bank embedding
        for the neighbor. Apply a margin hinge loss.
        """
        if self._repulsive_pairs is None or self._hn_bank is None:
            return torch.tensor(0.0, device=field_embs.device), 0

        # Find batch records that appear in repulsive pairs as anchors
        valid = text_idx >= 0
        if not valid.any():
            return torch.tensor(0.0, device=field_embs.device), 0

        device = field_embs.device
        batch_text_idx = text_idx.long()  # [B]

        # Get live union_name embeddings for the batch
        union_embs = F.normalize(field_embs[:, 0, :], dim=-1)  # [B, d_model]

        # For each batch record, find its repulsive neighbors in the pair list
        # Build a set of anchor text indices that appear in pairs
        pair_anchors = self._repulsive_pairs[:, 0]  # [P]
        pair_targets = self._repulsive_pairs[:, 1]  # [P]

        # Find which batch records are anchors in any pair
        # Use broadcasting: [B] vs [P] → match
        batch_in_pairs = batch_text_idx.unsqueeze(1) == pair_anchors.to(
            device
        ).unsqueeze(
            0
        )  # [B, P]

        # For each batch record, collect all its repulsive target indices
        # Get the batch indices and pair indices where there's a match
        batch_indices, pair_indices = batch_in_pairs.nonzero(as_tuple=True)

        if len(batch_indices) == 0:
            return torch.tensor(0.0, device=device), 0

        # Cap to avoid huge computation if too many pairs match
        max_pairs = 2048
        if len(batch_indices) > max_pairs:
            perm = torch.randperm(len(batch_indices), device=device)[:max_pairs]
            batch_indices = batch_indices[perm]
            pair_indices = pair_indices[perm]

        # Anchor embeddings (live, from batch)
        anchor_embs = union_embs[batch_indices]  # [M, d_model]

        # Target embeddings (from bank, detached)
        target_text_idx = pair_targets[pair_indices.cpu()].to(device)
        target_embs = self._hn_bank[target_text_idx]  # [M, d_model]

        # Cosine similarity between anchor and repulsive target
        cos_sim = (anchor_embs * target_embs).sum(dim=-1)  # [M]

        # Margin hinge: penalize when cos_sim > (1 - margin)
        threshold = 1.0 - self._repulsive_margin
        loss = F.relu(cos_sim - threshold).mean()

        return loss, len(batch_indices)

    def _encode(self, batch):
        return self.model.encode_fields(
            batch["token_ids"],
            batch["bloom_ids"],
            batch["is_number"],
            batch["token_lens"],
        )

    def training_step(self, batch, batch_idx):
        field_embs = self._encode(batch)

        cbow_loss, cbow_acc = self.model.cbow_loss(
            field_embs, batch["present"], batch["mask_idx"]
        )
        proto_loss, n_labeled, proto_acc = self.model.prototype_loss(
            field_embs, batch["fnum_label"]
        )
        loss = cbow_loss + proto_loss

        if self._repulsive_enabled:
            rep_loss, n_rep = self.repulsive_loss(field_embs, batch["text_idx"])
            loss = loss + self._repulsive_weight * rep_loss
            self.log("train/rep_loss", rep_loss)
            self.log("train/n_rep_pairs", float(n_rep))

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/cbow_loss", cbow_loss)
        self.log("train/proto_loss", proto_loss)
        self.log("train/cbow_acc", cbow_acc, prog_bar=True)
        self.log("train/proto_acc", float(proto_acc), prog_bar=True)
        self.log("train/n_labeled", float(n_labeled))

        return loss

    def validation_step(self, batch, batch_idx):
        field_embs = self._encode(batch)

        cbow_loss, cbow_acc = self.model.cbow_loss(
            field_embs, batch["present"], batch["mask_idx"]
        )
        proto_loss, _, proto_acc = self.model.prototype_loss(
            field_embs, batch["fnum_label"]
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
        "--labeled-csv",
        default="training/data/labeled_data.csv",
        help="Path to labeled_data.csv",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--save", default="training/data/record_embeddings.pt")
    parser.add_argument(
        "--resume", default=None, help="Checkpoint to resume from (partial load)"
    )
    parser.add_argument(
        "--repulsive-k",
        type=int,
        default=8,
        help="Number of nearest neighbors to check for number-overlap repulsion (0 to disable)",
    )
    parser.add_argument("--repulsive-margin", type=float, default=0.2)
    parser.add_argument("--repulsive-weight", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("Loading f7 records...")
    records = load_f7_records(args.db)
    print(f"  {len(records)} records loaded")

    print("Loading f_num labels...")
    text_to_fnum = load_fnum_labels(args.labeled_csv)
    print(f"  {len(text_to_fnum)} text->f_num mappings")

    print("Building token vocabulary (words only, numbers use bloom hashing)...")
    token_vocab = build_token_vocab(records, min_count=args.min_token_count)
    print(f"  {len(token_vocab)} word tokens (min_count={args.min_token_count})")

    # Build distinct union_name texts deduplicated by token signature
    # (tokenizer normalizes case/whitespace, so many raw texts map to the same tokens)
    token_sig_to_text: dict[tuple, str] = {}
    for rec in records:
        un = rec["union_name"].strip()
        if not un:
            continue
        tokens = _tokenize("union_name", un)
        sig = tuple((t["token"], t["is_num"]) for t in tokens)
        if sig not in token_sig_to_text:
            token_sig_to_text[sig] = un
    distinct_texts = sorted(token_sig_to_text.values())
    # Map each raw text to its deduplicated index via token signature
    text_to_text_idx: dict[str, int] = {}
    text_idx_lookup = {t: i for i, t in enumerate(distinct_texts)}
    sig_to_idx: dict[tuple, int] = {}
    for sig, canon_text in token_sig_to_text.items():
        sig_to_idx[sig] = text_idx_lookup[canon_text]
    for rec in records:
        un = rec["union_name"].strip()
        if not un:
            continue
        tokens = _tokenize("union_name", un)
        sig = tuple((t["token"], t["is_num"]) for t in tokens)
        text_to_text_idx[un] = sig_to_idx[sig]
    print(f"  {len(distinct_texts)} distinct union_name texts (after tokenizer dedup)")

    print("Building dataset...")
    dataset = RecordDataset(
        records, text_to_fnum, token_vocab, distinct_texts, text_to_text_idx
    )

    # Stratified split
    N = len(dataset)
    indices = list(range(N))
    random.shuffle(indices)

    labeled_idx = [i for i in indices if dataset.f_num[i] >= 0]
    unlabeled_idx = [i for i in indices if dataset.f_num[i] < 0]

    n_val_labeled = int(len(labeled_idx) * args.val_frac)
    n_val_unlabeled = int(len(unlabeled_idx) * args.val_frac)

    val_indices = labeled_idx[:n_val_labeled] + unlabeled_idx[:n_val_unlabeled]
    train_indices = labeled_idx[n_val_labeled:] + unlabeled_idx[n_val_unlabeled:]

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model
    module = RecordEmbeddingModule(
        d_model=args.d_model,
        temperature=args.temperature,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        vocab_size=len(token_vocab),
        n_classes=dataset.n_classes,
        repulsive_margin=args.repulsive_margin,
        repulsive_weight=args.repulsive_weight,
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

    # Setup repulsive mining (push apart NN with non-overlapping numbers)
    if args.repulsive_k > 0:
        print("Setting up repulsive mining...")
        module.setup_repulsive_mining(distinct_texts, token_vocab, args.repulsive_k)

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
            "fnum_to_idx": dataset.fnum_to_idx,
        },
        save_path,
    )
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
