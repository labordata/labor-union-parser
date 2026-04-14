#!/usr/bin/env python3
"""Train the union vs non-union detector.

Uses the same FastText+RoPE encoder as the f_num classifier.
Trains with ArcFace angular margin against a learned union prototype
on the hypersphere.

Usage:
    python training/train_union_detector.py
    python training/train_union_detector.py --epochs 30 --patience 10
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from labor_union_parser.extractor import UnionDetectorEncoder
from labor_union_parser.tokenizer import tokenize_for_arcface

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
MODEL_PATH = WEIGHTS_DIR / "union_detector.pt"

ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
N_BUCKETS = 50000
EMBED_DIM = 64


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


F7_DB_PATH = DATA_DIR / "../../f7.db"
N_EMPLOYER_NEGATIVES = 20000


def load_data():
    import sqlite3

    with open(DATA_DIR / "training_examples.json") as f:
        all_examples = json.load(f)

    def _is_union(ex):
        return ex["records"] and ex.get("reason_missing_fnum") not in (
            "multi-union",
            "multi-local",
        )

    splits = {}
    for split_name, split_filter in [
        ("train", lambda ex: ex["split"] == "train"),
        ("val", lambda ex: ex["split"] in ("val", "test")),
    ]:
        union = [
            ex["query"] for ex in all_examples if _is_union(ex) and split_filter(ex)
        ]
        nonunion = [
            ex["query"] for ex in all_examples if not _is_union(ex) and split_filter(ex)
        ]
        splits[split_name] = (union, nonunion)

    # Add F7 employer names as hard negatives (train only)
    f7_path = F7_DB_PATH.resolve()
    if f7_path.exists():
        conn = sqlite3.connect(str(f7_path))
        rows = conn.execute(
            "SELECT DISTINCT employer FROM f7 "
            "WHERE employer IS NOT NULL AND employer != '' "
            "ORDER BY employer"
        ).fetchall()
        conn.close()
        all_employers = [r[0] for r in rows]

        rng = random.Random(42)
        n_sample = min(N_EMPLOYER_NEGATIVES, len(all_employers))
        employer_sample = rng.sample(all_employers, n_sample)

        # 90/10 train/val split
        n_val = n_sample // 10
        splits["val"] = (splits["val"][0], splits["val"][1] + employer_sample[:n_val])
        splits["train"] = (
            splits["train"][0],
            splits["train"][1] + employer_sample[n_val:],
        )
        print(
            f"F7 employers: {n_sample} sampled ({n_sample - n_val} train, {n_val} val)"
        )
    else:
        print(f"Warning: {f7_path} not found, skipping employer negatives")

    return splits


def build_vocab(texts):
    """Build token vocab from a list of texts."""
    counter = Counter()
    for text in texts:
        tokens, is_num, ng_ids, ng_counts, bl_ids = tokenize_for_arcface(text)
        for tok in tokens:
            counter[tok] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, count in counter.most_common():
        if count >= 2:
            vocab[tok] = len(vocab)
    return vocab


def tokenize_and_collate(texts, vocab, device):
    """Tokenize texts and collate into tensors."""
    features = [tokenize_for_arcface(text) for text in texts]
    B = len(features)
    max_len = max(len(f[0]) for f in features)
    max_ngrams = len(features[0][2][0]) if features[0][2] else 32

    from labor_union_parser.tokenizer import NUM_BLOOM_HASHES

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num_t = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, (tokens, is_num, ng_ids, ng_counts, bl_ids) in enumerate(features):
        L = len(tokens)
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(
            [vocab.get(tok, 1) for tok in tokens], dtype=torch.long
        )
        ngram_ids[i, :L] = torch.tensor(ng_ids, dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ng_counts, dtype=torch.long)
        bloom_ids[i, :L] = torch.tensor(bl_ids, dtype=torch.long)
        is_num_t[i, :L] = torch.tensor([float(n) for n in is_num], dtype=torch.float)

    return (
        token_ids.to(device),
        ngram_ids.to(device),
        ngram_counts.to(device),
        bloom_ids.to(device),
        is_num_t.to(device),
        lengths.to(device),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, prototype, collated, labels, device, batch_size=1024):
    model.eval()
    proto = F.normalize(prototype, dim=0).to(device)
    tk, ng, nc, bl, isn, ln = collated

    y_true = labels.tolist()
    y_scores = []

    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            emb = model(
                tk[i : i + batch_size],
                ng[i : i + batch_size],
                nc[i : i + batch_size],
                bl[i : i + batch_size],
                isn[i : i + batch_size],
                ln[i : i + batch_size],
            )
            sims = (emb @ proto).cpu().tolist()
            y_scores.extend(sims)

    return np.array(y_true), np.array(y_scores)


def compute_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    n_pos = (y_true == 1).sum()
    n_neg = len(y_true) - n_pos
    total_errors = (1 - tpr) * n_pos + fpr * n_neg
    return float(thresholds[np.argmin(total_errors)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--epochs", default=30, help="Max training epochs")
@click.option("--batch-size", default=1024, help="Batch size")
@click.option("--lr", default=1e-3, type=float)
@click.option("--patience", default=10, help="Early stopping patience")
def main(epochs, batch_size, lr, patience):
    random.seed(42)
    torch.manual_seed(42)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load data
    splits = load_data()
    train_union, train_nonunion = splits["train"]
    val_union, val_nonunion = splits["val"]

    print(f"Union: {len(train_union)} train, {len(val_union)} val")
    print(f"Non-union: {len(train_nonunion)} train, {len(val_nonunion)} val")

    # Build vocab from training texts
    print("Building vocab...")
    all_train = train_union + train_nonunion
    vocab = build_vocab(all_train)
    print(f"Vocab: {len(vocab)} tokens")

    # Tokenize and collate
    print("Tokenizing...")
    train_texts = train_union + train_nonunion
    train_labels = torch.tensor(
        [1] * len(train_union) + [0] * len(train_nonunion), dtype=torch.float
    )
    val_texts = val_union + val_nonunion
    val_labels = torch.tensor(
        [1] * len(val_union) + [0] * len(val_nonunion), dtype=torch.float
    )

    train_collated = tokenize_and_collate(train_texts, vocab, device)
    val_collated = tokenize_and_collate(val_texts, vocab, device)
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Model
    model = UnionDetectorEncoder(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_buckets=N_BUCKETS,
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
    ).to(device)
    union_prototype = torch.nn.Parameter(
        F.normalize(torch.randn(EMBED_DIM), dim=0).to(device)
    )

    optimizer = torch.optim.AdamW(list(model.parameters()) + [union_prototype], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(train_texts)
    best_val_acc = 0.0
    best_state = None
    best_proto = None
    wait = 0

    tk, ng, nc, bl, isn, ln = train_collated

    print(
        f"\n{'Epoch':>7} | {'Loss':>8} | {'Val Acc':>7} | "
        f"{'FN':>5} | {'FP':>5} | {'Time':>6}"
    )
    print("-" * 52)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()

        indices = list(range(n_train))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = indices[start : start + batch_size]
            b_tk = tk[idx]
            b_ng = ng[idx]
            b_nc = nc[idx]
            b_bl = bl[idx]
            b_isn = isn[idx]
            b_ln = ln[idx]
            labels = train_labels[idx].to(device)

            embeddings = model(b_tk, b_ng, b_nc, b_bl, b_isn, b_ln)
            proto = F.normalize(union_prototype, dim=0)
            cos_sim = embeddings @ proto

            # Angular margin for union examples
            theta = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
            is_union = labels == 1
            margin_cos = torch.where(
                is_union, torch.cos(theta + ARCFACE_MARGIN), cos_sim
            )
            logits = ARCFACE_SCALE * margin_cos
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        y_true, y_scores = evaluate(
            model, union_prototype.data, val_collated, val_labels, device
        )
        threshold = compute_threshold(y_true, y_scores)
        preds = y_scores > threshold
        val_acc = accuracy_score(y_true, preds)
        fn = int(((y_true == 1) & ~preds).sum())
        fp = int(((y_true == 0) & preds).sum())
        elapsed = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_proto = union_prototype.data.cpu().clone()
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(
            f"  {epoch + 1:>2}/{epochs} | {avg_loss:>8.4f} | {val_acc:>6.1%} | "
            f"{fn:>5} | {fp:>5} | {elapsed:>5.1f}s{marker}"
        )

        if wait >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    y_true, y_scores = evaluate(model, best_proto, val_collated, val_labels, device)
    threshold = compute_threshold(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    preds = y_scores > threshold

    n_union = int((y_true == 1).sum())
    n_nonunion = int((y_true == 0).sum())
    fn = int(((y_true == 1) & ~preds).sum())
    fp = int(((y_true == 0) & preds).sum())

    print("\nFinal results:")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Optimal threshold: {threshold:.4f}")
    print(f"  False negatives:   {fn}/{n_union} ({fn / n_union:.4f})")
    print(f"  False positives:   {fp}/{n_nonunion} ({fp / n_nonunion:.4f})")

    # Save
    torch.save(
        {
            "model_state_dict": best_state,
            "union_centroid": best_proto,
            "optimal_threshold": threshold,
            "vocab": vocab,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "n_buckets": N_BUCKETS,
            "vocab_size": len(vocab),
            "embed_dim": EMBED_DIM,
        },
        MODEL_PATH,
    )
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
