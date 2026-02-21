#!/usr/bin/env python3
"""
Structured multi-head classifier: encoder + Transformer with per-field
classification heads.

Three encoder modes (--encoder):
  char         : character-level CNN over raw text (up to 200 chars)
  token-charcnn: tokenize → CharCNN per token (20 tokens × 20 chars)
  token-embed  : tokenize → learned token embedding (20 tokens)

Fields: union_name, desig_name, desig_num, prefix, suffix, union_unit
"""

import json
import math
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.char_cnn import (
    CharacterCNN,
    tokenize_to_chars,
)

DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
MODEL_DIR = DATA_DIR


def model_path(encoder="char"):
    return MODEL_DIR / f"structured_classifier_{encoder}.pt"


# Default for backward compat
MODEL_PATH = model_path("char")

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Character vocabulary for char encoder: printable ASCII
CHAR_VOCAB = {chr(i): i + 1 for i in range(32, 127)}  # 1-indexed, 0 = padding
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for padding
MAX_CHAR_LEN = 200

# Token-level settings
MAX_TOKENS = 20
MAX_CHARS_PER_TOKEN = 20

FIELDS = ["union_name", "desig_name", "desig_num", "prefix", "suffix", "f_num"]

# Fields that use pointer heads instead of classification heads.
# Pointer heads predict which input token position contains the field value.
POINTER_FIELDS = set()  # populated by --pointer-fields flag


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_chars(text, max_len=MAX_CHAR_LEN):
    """Convert text to character index list (for char encoder)."""
    return [CHAR_VOCAB.get(c, 0) for c in text[:max_len]]


def smart_truncate_nonspace(text, max_tokens=MAX_TOKENS):
    """Tokenize, drop spaces, keep first N tokens, recover lost numbers."""
    full_chars, full_tokens, full_is_num, full_token_types, full_num_ids = (
        tokenize_to_chars(text, 999)
    )

    nonspace = []
    for i, tt in enumerate(full_token_types):
        if full_tokens[i] and tt != 2:  # not space, not empty
            nonspace.append(
                {
                    "chars": full_chars[i],
                    "token": full_tokens[i],
                    "is_num": full_is_num[i],
                    "token_type": tt,
                }
            )

    trunc = nonspace[:max_tokens]

    # Recover lost numbers
    trunc_numbers = {d["token"] for d in trunc if d["is_num"]}
    lost_numbers = [
        d for d in nonspace if d["is_num"] and d["token"] not in trunc_numbers
    ]

    if lost_numbers:
        replace_positions = []
        for i in range(len(trunc) - 1, -1, -1):
            if not trunc[i]["is_num"] and trunc[i]["token"]:
                replace_positions.append(i)
                if len(replace_positions) >= len(lost_numbers):
                    break
        replace_positions.reverse()
        for pos, lost in zip(replace_positions, lost_numbers):
            trunc[pos] = lost

    # Pad
    while len(trunc) < max_tokens:
        trunc.append(
            {
                "chars": [0] * MAX_CHARS_PER_TOKEN,
                "token": "",
                "is_num": 0,
                "token_type": 4,
            }
        )

    return trunc


def build_token_vocab(train_examples):
    """Build token→idx vocabulary from training data."""
    token_counts = {}
    for ex in train_examples:
        tokens = smart_truncate_nonspace(ex["query"])
        for t in tokens:
            tok = t["token"]
            if tok:
                token_counts[tok] = token_counts.get(tok, 0) + 1

    # Sort by frequency descending, assign indices
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, _ in sorted(token_counts.items(), key=lambda x: -x[1]):
        vocab[tok] = len(vocab)

    return vocab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _find_pointer_label(tokens, target_value):
    """Find which token position contains the target value.

    Returns position index (0..MAX_TOKENS-1) or MAX_TOKENS for NONE.
    Matches numbers by stripped value, non-numbers case-insensitively.
    """
    target_str = str(target_value).strip()
    if not target_str or target_str == "0":
        return MAX_TOKENS  # NONE
    target_lower = target_str.lower()
    target_num = target_str.lstrip("0") or "0"
    for i, t in enumerate(tokens):
        if t["is_num"] and t["token"] == target_num:
            return i
        if not t["is_num"] and t["token"] == target_lower:
            return i
    return MAX_TOKENS  # NONE


class StructuredDataset(Dataset):
    def __init__(self, examples, field_vocabs, encoder="char", token_vocab=None):
        self.encoder = encoder
        self.labels = {f: [] for f in FIELDS}
        # Pointer labels: position index or MAX_TOKENS for NONE
        self.pointer_labels = {f: [] for f in POINTER_FIELDS}

        # Collect valid queries and records
        queries = []
        records = []
        for ex in examples:
            if not ex["records"]:
                continue
            rec = ex["records"][0]
            label_row = {}
            for f in FIELDS:
                if f in POINTER_FIELDS:
                    continue  # pointer fields don't need vocab lookup
                val = _get_field_value(rec, f)
                idx = field_vocabs[f].get(val)
                # Unseen values: use -100 so CE loss ignores them
                label_row[f] = idx if idx is not None else -100

            queries.append(ex["query"])
            records.append(rec)
            for f in FIELDS:
                if f not in POINTER_FIELDS:
                    self.labels[f].append(label_row[f])

        for f in FIELDS:
            if f not in POINTER_FIELDS:
                self.labels[f] = torch.tensor(self.labels[f], dtype=torch.long)

        # Pre-encode all queries at init time
        if encoder == "char":
            self.char_ids_list = [encode_chars(q) for q in queries]
            # Pointer labels can't work with char encoder (no token positions)
            if POINTER_FIELDS:
                raise ValueError("Pointer fields require a token-level encoder")
        elif encoder == "token-charcnn":
            self.token_char_ids = []
            self.masks = []
            self.token_strings = []  # needed for pointer scoring at eval
            for i, q in enumerate(queries):
                tokens = smart_truncate_nonspace(q)
                self.token_char_ids.append([t["chars"] for t in tokens])
                self.masks.append([1 if t["token"] else 0 for t in tokens])
                self.token_strings.append([t["token"] for t in tokens])
                # Compute pointer labels
                for f in POINTER_FIELDS:
                    val = _get_field_value(records[i], f)
                    self.pointer_labels[f].append(_find_pointer_label(tokens, val))
        elif encoder == "token-embed":
            assert token_vocab is not None
            self.token_ids = []
            self.masks = []
            self.token_strings = []
            for i, q in enumerate(queries):
                tokens = smart_truncate_nonspace(q)
                self.token_ids.append(
                    [
                        (
                            token_vocab.get(t["token"], token_vocab["<UNK>"])
                            if t["token"]
                            else 0
                        )
                        for t in tokens
                    ]
                )
                self.masks.append([1 if t["token"] else 0 for t in tokens])
                self.token_strings.append([t["token"] for t in tokens])
                for f in POINTER_FIELDS:
                    val = _get_field_value(records[i], f)
                    self.pointer_labels[f].append(_find_pointer_label(tokens, val))

        for f in POINTER_FIELDS:
            self.pointer_labels[f] = torch.tensor(
                self.pointer_labels[f], dtype=torch.long
            )

        self._len = len(queries)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        labels = {f: self.labels[f][idx] for f in FIELDS if f not in POINTER_FIELDS}
        for f in POINTER_FIELDS:
            labels[f] = self.pointer_labels[f][idx]

        if self.encoder == "char":
            return {"char_ids": self.char_ids_list[idx]}, labels
        elif self.encoder == "token-charcnn":
            return {
                "char_ids": self.token_char_ids[idx],
                "mask": self.masks[idx],
            }, labels
        elif self.encoder == "token-embed":
            return {
                "token_ids": self.token_ids[idx],
                "mask": self.masks[idx],
            }, labels
        raise ValueError(f"Unknown encoder: {self.encoder}")


def collate_char(batch):
    """Collate for char encoder: variable-length char sequences."""
    inputs_list, labels_list = zip(*batch)
    char_ids_list = [inp["char_ids"] for inp in inputs_list]
    max_len = max(len(ids) for ids in char_ids_list)
    padded = torch.zeros(len(char_ids_list), max_len, dtype=torch.long)
    mask = torch.zeros(len(char_ids_list), max_len, dtype=torch.bool)
    for i, ids in enumerate(char_ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = True
    labels = {f: torch.stack([el[f] for el in labels_list]) for f in FIELDS}
    return {"char_ids": padded, "mask": mask}, labels


def collate_token_charcnn(batch):
    """Collate for token-charcnn: fixed-size (B, MAX_TOKENS, MAX_CHARS_PER_TOKEN)."""
    inputs_list, labels_list = zip(*batch)
    char_ids = torch.tensor([inp["char_ids"] for inp in inputs_list], dtype=torch.long)
    mask = torch.tensor([inp["mask"] for inp in inputs_list], dtype=torch.bool)
    labels = {f: torch.stack([el[f] for el in labels_list]) for f in FIELDS}
    return {"char_ids": char_ids, "mask": mask}, labels


def collate_token_embed(batch):
    """Collate for token-embed: fixed-size (B, MAX_TOKENS)."""
    inputs_list, labels_list = zip(*batch)
    token_ids = torch.tensor(
        [inp["token_ids"] for inp in inputs_list], dtype=torch.long
    )
    mask = torch.tensor([inp["mask"] for inp in inputs_list], dtype=torch.bool)
    labels = {f: torch.stack([el[f] for el in labels_list]) for f in FIELDS}
    return {"token_ids": token_ids, "mask": mask}, labels


COLLATE_FNS = {
    "char": collate_char,
    "token-charcnn": collate_token_charcnn,
    "token-embed": collate_token_embed,
}


def _get_field_value(record, field):
    """Get field value from record."""
    if field == "f_num":
        return record["f_num"]
    if field == "desig_num":
        return record.get("desig_num", 0)
    return record.get(field, "")


def build_field_vocabs(train_examples):
    """Build value→idx mappings for each field from training data."""
    vocabs = {}
    for f in FIELDS:
        values = set()
        for ex in train_examples:
            if ex["records"]:
                values.add(_get_field_value(ex["records"][0], f))
        vocabs[f] = {v: i for i, v in enumerate(sorted(values, key=str))}
    return vocabs


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_freqs(dim, max_len, theta=10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos_freqs, sin_freqs):
    """Apply rotary position embedding. x: (batch, seq, heads, head_dim)"""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    seq_len = x.shape[1]
    cos_f = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(2)
    sin_f = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(2)
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x2 * cos_f + x1 * sin_f
    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=MAX_CHAR_LEN):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        cos_freqs, sin_freqs = precompute_freqs(self.head_dim, max_seq_len)
        self.register_buffer("cos_freqs", cos_freqs)
        self.register_buffer("sin_freqs", sin_freqs)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = apply_rope(q, self.cos_freqs, self.sin_freqs)
        k = apply_rope(k, self.cos_freqs, self.sin_freqs)

        q = q.transpose(1, 2)  # (B, heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, max_seq_len=MAX_CHAR_LEN):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.ff(self.norm2(x))
        return x


class StructuredClassifier(nn.Module):
    def __init__(
        self,
        field_sizes,
        encoder="char",
        d_model=256,
        n_heads=4,
        n_layers=2,
        ff_dim=512,
        dropout=0.1,
        # char encoder params
        cnn_channels=128,
        cnn_kernels=(3, 5, 7),
        # token-embed params
        token_vocab_size=None,
    ):
        super().__init__()
        self.encoder_type = encoder

        if encoder == "char":
            max_seq_len = MAX_CHAR_LEN
            self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, 64, padding_idx=0)
            self.convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(64, cnn_channels, k, padding=k // 2),
                        nn.GELU(),
                    )
                    for k in cnn_kernels
                ]
            )
            cnn_out_dim = cnn_channels * len(cnn_kernels)
            self.cnn_proj = nn.Linear(cnn_out_dim, d_model)
            self.cnn_norm = nn.LayerNorm(d_model)
            self.cnn_dropout = nn.Dropout(dropout)

        elif encoder == "token-charcnn":
            max_seq_len = MAX_TOKENS
            # Reuse the CharacterCNN from the dual task model
            self.char_cnn = CharacterCNN(embed_dim=d_model, char_embed_dim=16)

        elif encoder == "token-embed":
            max_seq_len = MAX_TOKENS
            assert token_vocab_size is not None
            self.token_emb = nn.Embedding(token_vocab_size, d_model, padding_idx=0)

        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, dropout, max_seq_len)
                for _ in range(n_layers)
            ]
        )

        # Per-field classification heads (for non-pointer fields)
        self.heads = nn.ModuleDict(
            {
                f: nn.Linear(d_model, n_classes)
                for f, n_classes in field_sizes.items()
                if f not in POINTER_FIELDS
            }
        )

        # Pointer heads: project each position to a scalar score + learned NONE
        self.pointer_heads = nn.ModuleDict(
            {f: nn.Linear(d_model, 1) for f in POINTER_FIELDS}
        )
        # Learned NONE score per pointer field
        self.pointer_none = nn.ParameterDict(
            {f: nn.Parameter(torch.zeros(1)) for f in POINTER_FIELDS}
        )

    def forward(self, inputs, mask=None):
        if self.encoder_type == "char":
            char_ids = inputs  # (B, S) character indices
            x = self.char_emb(char_ids)  # (B, S, 64)
            x_t = x.transpose(1, 2)
            conv_outs = [conv(x_t) for conv in self.convs]
            x = torch.cat(conv_outs, dim=1)
            x = x.transpose(1, 2)
            x = self.cnn_dropout(self.cnn_norm(self.cnn_proj(x)))

        elif self.encoder_type == "token-charcnn":
            char_ids = inputs  # (B, MAX_TOKENS, MAX_CHARS_PER_TOKEN)
            x = self.char_cnn(char_ids)  # (B, MAX_TOKENS, d_model)

        elif self.encoder_type == "token-embed":
            token_ids = inputs  # (B, MAX_TOKENS)
            x = self.token_emb(token_ids)  # (B, MAX_TOKENS, d_model)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Mean pool (masked) for classification heads
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).float()
            pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(
                min=1
            )
        else:
            pooled = x.mean(dim=1)

        # Classification head logits
        logits = {f: head(pooled) for f, head in self.heads.items()}

        # Pointer head logits: (B, MAX_TOKENS+1) — positions + NONE
        for f in POINTER_FIELDS:
            pos_scores = self.pointer_heads[f](x).squeeze(-1)  # (B, S)
            # Mask out padding positions
            if mask is not None:
                pos_scores = pos_scores.masked_fill(~mask, float("-inf"))
            # Append NONE score
            none_score = self.pointer_none[f].expand(pos_scores.shape[0], 1)
            logits[f] = torch.cat([pos_scores, none_score], dim=1)  # (B, S+1)

        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def evaluate(model, loader, field_vocabs, encoder):
    model.eval()
    correct = {f: 0 for f in FIELDS}
    total = {f: 0 for f in FIELDS}
    null_labels = {
        "prefix": field_vocabs.get("prefix", {}).get(0),
        "suffix": field_vocabs.get("suffix", {}).get(""),
        "desig_name": field_vocabs.get("desig_name", {}).get("LU"),
        "desig_num": field_vocabs.get("desig_num", {}).get(0),
    }
    non_null_correct = {f: 0 for f in FIELDS}
    non_null_total = {f: 0 for f in FIELDS}

    with torch.no_grad():
        for inputs, labels in loader:
            model_input, mask = get_model_input(inputs, encoder, DEVICE)
            logits = model(model_input, mask)
            for f in FIELDS:
                y = labels[f].to(DEVICE)
                preds = logits[f].argmax(dim=-1)
                correct[f] += (preds == y).sum().item()
                total[f] += y.shape[0]

                if f in POINTER_FIELDS:
                    # For pointer fields, "null" = NONE (position MAX_TOKENS)
                    nl = MAX_TOKENS
                else:
                    nl = null_labels.get(f)
                if nl is not None:
                    non_null_mask = y != nl
                    if non_null_mask.any():
                        non_null_correct[f] += (
                            (preds[non_null_mask] == y[non_null_mask]).sum().item()
                        )
                        non_null_total[f] += non_null_mask.sum().item()

    results = {}
    for f in FIELDS:
        acc = correct[f] / total[f] if total[f] > 0 else 0
        nn_acc = (
            non_null_correct[f] / non_null_total[f] if non_null_total[f] > 0 else None
        )
        results[f] = (acc, nn_acc, non_null_total[f])
    return results


def get_model_input(inputs, encoder, device):
    """Extract the right tensor from inputs dict based on encoder type."""
    mask = inputs["mask"].to(device)
    if encoder == "token-embed":
        return inputs["token_ids"].to(device), mask
    elif encoder in ("token-charcnn", "char"):
        return inputs["char_ids"].to(device), mask
    raise ValueError(f"Unknown encoder: {encoder}")


@click.command()
@click.option("--epochs", default=20, help="Number of training epochs")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--d-model", default=256, help="Model hidden dimension")
@click.option("--n-layers", default=2, help="Number of transformer layers")
@click.option(
    "--encoder",
    default="char",
    type=click.Choice(["char", "token-charcnn", "token-embed"]),
    help="Encoder type",
)
@click.option(
    "--pointer-fields",
    multiple=True,
    type=click.Choice(["desig_num", "prefix", "suffix"]),
    help="Fields to use pointer heads for (can repeat). E.g. --pointer-fields desig_num",
)
@click.option(
    "--fnum-weight-decay",
    default=0.01,
    type=float,
    help="Weight decay for f_num head (default: 0.01, same as rest of model)",
)
def main(
    epochs,
    batch_size,
    lr,
    d_model,
    n_layers,
    encoder,
    pointer_fields,
    fnum_weight_decay,
):
    # Set global pointer fields
    global POINTER_FIELDS
    POINTER_FIELDS = set(pointer_fields)

    print(f"Device: {DEVICE}")
    print(f"Encoder: {encoder}")
    if POINTER_FIELDS:
        print(f"Pointer fields: {sorted(POINTER_FIELDS)}")
    print("Loading data...")

    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for ex in all_examples:
        splits[ex["split"]].append(ex)

    print(
        f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    # Build vocabularies from training data
    field_vocabs = build_field_vocabs(splits["train"])
    field_sizes = {f: len(v) for f, v in field_vocabs.items()}
    print("Field sizes:", {f: n for f, n in field_sizes.items()})

    # Build token vocab if needed
    token_vocab = None
    if encoder == "token-embed":
        token_vocab = build_token_vocab(splits["train"])
        print(f"Token vocabulary: {len(token_vocab)} tokens")

    # Datasets
    train_ds = StructuredDataset(splits["train"], field_vocabs, encoder, token_vocab)
    val_ds = StructuredDataset(splits["val"], field_vocabs, encoder, token_vocab)
    test_ds = StructuredDataset(splits["test"], field_vocabs, encoder, token_vocab)
    print(
        f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # Report pointer label stats
    for f in POINTER_FIELDS:
        ptr_labels = train_ds.pointer_labels[f]
        n_none = (ptr_labels == MAX_TOKENS).sum().item()
        n_found = (ptr_labels < MAX_TOKENS).sum().item()
        print(
            f"  {f} pointer: {n_found} found in text, {n_none} NONE ({n_none/len(ptr_labels)*100:.1f}%)"
        )

    collate_fn = COLLATE_FNS[encoder]
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = StructuredClassifier(
        field_sizes=field_sizes,
        encoder=encoder,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        ff_dim=d_model * 2,
        dropout=0.1,
        token_vocab_size=len(token_vocab) if token_vocab else None,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if fnum_weight_decay != 0.01 and "f_num" in model.heads:
        fnum_params = list(model.heads["f_num"].parameters())
        fnum_ids = {id(p) for p in fnum_params}
        other_params = [p for p in model.parameters() if id(p) not in fnum_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": 0.01},
                {"params": fnum_params, "weight_decay": fnum_weight_decay},
            ],
            lr=lr,
        )
        print(f"f_num head weight decay: {fnum_weight_decay}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, labels in train_loader:
            model_input, mask = get_model_input(inputs, encoder, DEVICE)
            logits = model(model_input, mask)

            loss = 0.0
            for f in FIELDS:
                y = labels[f].to(DEVICE)
                loss = loss + F.cross_entropy(logits[f], y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Evaluate
        val_results = evaluate(model, val_loader, field_vocabs, encoder)
        mean_val_acc = np.mean([acc for acc, _, _ in val_results.values()])

        print(
            f"\nEpoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )
        for f in FIELDS:
            acc, nn_acc, nn_total = val_results[f]
            msg = f"  {f:15s} val={acc:.4f}"
            if nn_acc is not None:
                msg += f"  non-null={nn_acc:.4f} ({nn_total})"
            print(msg)

        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            save_dict = {
                "model_state": model.state_dict(),
                "field_vocabs": field_vocabs,
                "field_sizes": field_sizes,
                "d_model": d_model,
                "n_layers": n_layers,
                "encoder": encoder,
                "pointer_fields": sorted(POINTER_FIELDS),
            }
            if token_vocab is not None:
                save_dict["token_vocab"] = token_vocab
            save_path = model_path(encoder)
            torch.save(save_dict, save_path)
            print(f"  ** Saved {save_path.name} (mean val acc={mean_val_acc:.4f})")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Loading best model for test evaluation...")
    ckpt = torch.load(model_path(encoder), weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_results = evaluate(model, test_loader, field_vocabs, encoder)
    print("\nTest results:")
    for f in FIELDS:
        acc, nn_acc, nn_total = test_results[f]
        msg = f"  {f:15s} test={acc:.4f}"
        if nn_acc is not None:
            msg += f"  non-null={nn_acc:.4f} ({nn_total})"
        print(msg)

    # Compare with LogReg baselines
    print("\nLogReg baselines for reference:")
    print("  union_name      test=0.9915")
    print("  desig_name      test=0.9592 (0.05 weighted)")
    print("  desig_num       (pending)")
    print("  prefix          test=0.9595 (0.05 weighted)")
    print("  suffix          test=0.9328 (0.05 weighted)")
    print("  f_num           (new — ~34K classes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
