#!/usr/bin/env python3
"""
Structured multi-head classifier: single CharCNN + Transformer encoder
with per-field classification heads.

Architecture:
  - Character embedding → 1D CNN (local n-gram features)
  - 2 Transformer layers with RoPE (global reasoning)
  - Mean pooling → per-field linear classification heads

Fields: union_name, desig_name, desig_num, prefix, suffix
"""

import json
import math
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
MODEL_PATH = DATA_DIR / "structured_classifier.pt"

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Character vocabulary: printable ASCII
CHAR_VOCAB = {chr(i): i + 1 for i in range(32, 127)}  # 1-indexed, 0 = padding
CHAR_VOCAB_SIZE = len(CHAR_VOCAB) + 1  # +1 for padding
MAX_QUERY_LEN = 200

FIELDS = ["union_name", "desig_name", "desig_num", "prefix", "suffix", "union_unit"]


def encode_chars(text, max_len=MAX_QUERY_LEN):
    """Convert text to character index tensor."""
    ids = [CHAR_VOCAB.get(c, 0) for c in text[:max_len]]
    return ids


class StructuredDataset(Dataset):
    def __init__(self, examples, field_vocabs):
        self.queries = []
        self.labels = {}
        for f in FIELDS:
            self.labels[f] = []

        for ex in examples:
            if not ex["records"]:
                continue
            rec = ex["records"][0]
            # Check all fields have valid labels
            skip = False
            label_row = {}
            for f in FIELDS:
                val = _get_field_value(rec, f)
                idx = field_vocabs[f].get(val)
                if idx is None:
                    skip = True
                    break
                label_row[f] = idx
            if skip:
                continue

            self.queries.append(ex["query"])
            for f in FIELDS:
                self.labels[f].append(label_row[f])

        for f in FIELDS:
            self.labels[f] = torch.tensor(self.labels[f], dtype=torch.long)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        char_ids = encode_chars(self.queries[idx])
        labels = {f: self.labels[f][idx] for f in FIELDS}
        return char_ids, labels


def collate_fn(batch):
    """Pad character sequences and stack labels."""
    char_ids_list, labels_list = zip(*batch)
    max_len = max(len(ids) for ids in char_ids_list)
    padded = torch.zeros(len(char_ids_list), max_len, dtype=torch.long)
    mask = torch.zeros(len(char_ids_list), max_len, dtype=torch.bool)
    for i, ids in enumerate(char_ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[i, : len(ids)] = True
    labels = {f: torch.stack([el[f] for el in labels_list]) for f in FIELDS}
    return padded, mask, labels


def _get_field_value(record, field):
    """Get field value from record."""
    if field == "union_unit":
        uid = record.get("unit_id", "")
        if uid:
            return f"{record['union_name']}|{uid}"
        else:
            return record["union_name"]
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
    """Apply rotary position embedding to input tensor.

    x: (batch, seq, heads, head_dim)
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    # Slice freqs to match sequence length
    seq_len = x.shape[1]
    cos_f = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq, 1, dim/2)
    sin_f = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(2)
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x2 * cos_f + x1 * sin_f
    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        cos_freqs, sin_freqs = precompute_freqs(self.head_dim, MAX_QUERY_LEN)
        self.register_buffer("cos_freqs", cos_freqs)
        self.register_buffer("sin_freqs", sin_freqs)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply RoPE to q and k
        q = apply_rope(q, self.cos_freqs, self.sin_freqs)
        k = apply_rope(k, self.cos_freqs, self.sin_freqs)

        # Attention
        q = q.transpose(1, 2)  # (B, heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # mask: (B, S) -> (B, 1, 1, S)
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, n_heads)
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
        d_model=256,
        n_heads=4,
        n_layers=2,
        ff_dim=512,
        cnn_channels=128,
        cnn_kernels=(3, 5, 7),
        dropout=0.1,
    ):
        super().__init__()
        self.char_emb = nn.Embedding(CHAR_VOCAB_SIZE, 64, padding_idx=0)

        # CharCNN: parallel convolutions with different kernel sizes
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

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )

        # Per-field classification heads
        self.heads = nn.ModuleDict(
            {f: nn.Linear(d_model, n_classes) for f, n_classes in field_sizes.items()}
        )

    def forward(self, char_ids, mask=None):
        # char_ids: (B, S) of character indices
        x = self.char_emb(char_ids)  # (B, S, 64)

        # CNN: (B, S, 64) -> (B, 64, S) -> conv -> (B, C, S)
        x_t = x.transpose(1, 2)
        conv_outs = [conv(x_t) for conv in self.convs]  # list of (B, cnn_channels, S)
        x = torch.cat(conv_outs, dim=1)  # (B, cnn_out_dim, S)
        x = x.transpose(1, 2)  # (B, S, cnn_out_dim)
        x = self.cnn_dropout(self.cnn_norm(self.cnn_proj(x)))  # (B, S, d_model)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Mean pool (masked)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        else:
            pooled = x.mean(dim=1)

        # Per-field logits
        logits = {f: head(pooled) for f, head in self.heads.items()}
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def evaluate(model, loader, field_vocabs):
    model.eval()
    correct = {f: 0 for f in FIELDS}
    total = {f: 0 for f in FIELDS}
    # Track non-null accuracy
    null_labels = {
        "prefix": field_vocabs["prefix"].get(0),
        "suffix": field_vocabs["suffix"].get(""),
        "desig_name": field_vocabs["desig_name"].get("LU"),
        "desig_num": field_vocabs["desig_num"].get(0),
    }
    non_null_correct = {f: 0 for f in FIELDS}
    non_null_total = {f: 0 for f in FIELDS}

    with torch.no_grad():
        for char_ids, mask, labels in loader:
            char_ids = char_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(char_ids, mask)
            for f in FIELDS:
                y = labels[f].to(DEVICE)
                preds = logits[f].argmax(dim=-1)
                correct[f] += (preds == y).sum().item()
                total[f] += y.shape[0]
                # Non-null
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


@click.command()
@click.option("--epochs", default=20, help="Number of training epochs")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--d-model", default=256, help="Model hidden dimension")
@click.option("--n-layers", default=2, help="Number of transformer layers")
def main(epochs, batch_size, lr, d_model, n_layers):
    print(f"Device: {DEVICE}")
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

    # Datasets
    train_ds = StructuredDataset(splits["train"], field_vocabs)
    val_ds = StructuredDataset(splits["val"], field_vocabs)
    test_ds = StructuredDataset(splits["test"], field_vocabs)
    print(
        f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

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
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        ff_dim=d_model * 2,
        dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for char_ids, mask, labels in train_loader:
            char_ids = char_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(char_ids, mask)

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
        val_results = evaluate(model, val_loader, field_vocabs)
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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "field_vocabs": field_vocabs,
                    "field_sizes": field_sizes,
                    "d_model": d_model,
                    "n_layers": n_layers,
                },
                MODEL_PATH,
            )
            print(f"  ** Saved (mean val acc={mean_val_acc:.4f})")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Loading best model for test evaluation...")
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_results = evaluate(model, test_loader, field_vocabs)
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
    print("  union_unit      test=0.9493")

    print("\nDone!")


if __name__ == "__main__":
    main()
