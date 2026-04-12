#!/usr/bin/env python3
"""Train the union vs non-union detector.

Uses AttentionPoolingEncoder from the production package.
Trains a contrastive model with a learned union prototype on the
hypersphere using ArcFace angular margin.

Usage:
    python training/train_union_detector.py
    python training/train_union_detector.py --epochs 30 --patience 10
"""

import json
import random
import time
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from labor_union_parser.char_cnn import CharacterCNN, tokenize_to_chars
from labor_union_parser.extractor import AttentionPoolingEncoder

print = partial(print, flush=True)  # noqa: A001

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
MODEL_PATH = WEIGHTS_DIR / "union_detector.pt"

MAX_TOKENS = 30
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
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

    return splits


def tokenize_texts(texts):
    """Tokenize a list of texts into tensors."""
    char_ids_list = []
    token_type_list = []
    is_number_list = []

    for text in texts:
        char_ids, _, is_number, token_type = tokenize_to_chars(
            text, max_tokens=MAX_TOKENS
        )
        char_ids_list.append(char_ids)
        token_type_list.append(token_type)
        is_number_list.append(is_number)

    return (
        torch.tensor(char_ids_list, dtype=torch.long),
        torch.tensor(token_type_list, dtype=torch.long),
        torch.tensor(is_number_list, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model, prototype, char_ids, token_type, is_number, labels, device, batch_size=1024
):
    """Compute cosine similarities and return (y_true, y_scores)."""
    model.eval()
    proto = F.normalize(prototype, dim=0).to(device)

    y_true = labels.tolist()
    y_scores = []

    with torch.no_grad():
        for i in range(0, len(labels), batch_size):
            ci = char_ids[i : i + batch_size].to(device)
            tt = token_type[i : i + batch_size].to(device)
            isn = is_number[i : i + batch_size].to(device)
            emb = model(ci, tt, isn)
            sims = (emb @ proto).cpu().tolist()
            y_scores.extend(sims)

    return np.array(y_true), np.array(y_scores)


def compute_threshold(y_true, y_scores):
    """Find threshold that minimizes total errors."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    n_pos = (y_true == 1).sum()
    n_neg = len(y_true) - n_pos
    total_errors = (1 - tpr) * n_pos + fpr * n_neg
    return float(thresholds[np.argmin(total_errors)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--epochs", default=30, help="Number of training epochs")
@click.option("--batch-size", default=1024, help="Batch size")
@click.option("--lr", default=1e-3, help="Learning rate")
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

    # Tokenize all data upfront
    print("Tokenizing...")
    train_texts = train_union + train_nonunion
    train_labels = torch.tensor(
        [1] * len(train_union) + [0] * len(train_nonunion), dtype=torch.float
    )
    val_texts = val_union + val_nonunion
    val_labels = torch.tensor(
        [1] * len(val_union) + [0] * len(val_nonunion), dtype=torch.float
    )

    train_ci, train_tt, train_isn = tokenize_texts(train_texts)
    val_ci, val_tt, val_isn = tokenize_texts(val_texts)
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Model
    char_cnn = CharacterCNN(embed_dim=64, char_embed_dim=16)
    model = AttentionPoolingEncoder(
        char_cnn, embed_dim=64, num_embed_dim=8, num_heads=4
    ).to(device)
    union_prototype = torch.nn.Parameter(F.normalize(torch.randn(64), dim=0).to(device))

    optimizer = torch.optim.AdamW(list(model.parameters()) + [union_prototype], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = len(train_texts)
    best_val_acc = 0.0
    best_state = None
    best_proto = None
    wait = 0

    print(
        f"\n{'Epoch':>7} | {'Loss':>8} | {'Val Acc':>7} | {'FN':>5} | {'FP':>5} | {'Time':>6}"
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
            batch_idx = indices[start : start + batch_size]
            ci = train_ci[batch_idx].to(device)
            tt = train_tt[batch_idx].to(device)
            isn = train_isn[batch_idx].to(device)
            labels = train_labels[batch_idx].to(device)

            embeddings = model(ci, tt, isn)
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
            model, union_prototype.data, val_ci, val_tt, val_isn, val_labels, device
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

    # Final evaluation with best model
    y_true, y_scores = evaluate(
        model, best_proto, val_ci, val_tt, val_isn, val_labels, device
    )
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
        },
        MODEL_PATH,
    )
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
