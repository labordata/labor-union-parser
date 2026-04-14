#!/usr/bin/env python3
"""Fit Platt scaling for union detector cosine similarities.

Fits sigmoid(a * cos_sim + b) to calibrate raw cosine similarities
into probabilities.

Reads:
  - src/labor_union_parser/weights/union_detector.pt
  - training/data/training_examples.json
  - training/data/f7.db (optional, for employer negatives)

Writes:
  - training/data/platt_params.json
"""

import json
import random
import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F

from labor_union_parser.extractor import UnionDetectorEncoder
from labor_union_parser.tokenizer import NUM_BLOOM_HASHES, tokenize_for_arcface

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
BATCH_SIZE = 256


def _collate(batch_features, vocab, device):
    B = len(batch_features)
    max_len = max(len(f[0]) for f in batch_features)
    max_ngrams = len(batch_features[0][2][0]) if batch_features[0][2] else 32

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num_t = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, (tokens, is_num, ng_ids, ng_counts, bl_ids) in enumerate(batch_features):
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


def main():
    device = "cpu"

    with open(DATA_DIR / "training_examples.json") as f:
        all_examples = json.load(f)

    # Load union detector
    ud_ckpt = torch.load(
        WEIGHTS_DIR / "union_detector.pt", map_location=device, weights_only=False
    )

    encoder = UnionDetectorEncoder(
        d_model=ud_ckpt["d_model"],
        n_heads=ud_ckpt["n_heads"],
        n_layers=ud_ckpt["n_layers"],
        n_buckets=ud_ckpt["n_buckets"],
        vocab_size=ud_ckpt["vocab_size"],
        embed_dim=ud_ckpt["embed_dim"],
    )
    encoder.load_state_dict(ud_ckpt["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    centroid = ud_ckpt["union_centroid"].to(device)
    vocab = ud_ckpt["vocab"]

    # Build val set
    val_union = [
        ex
        for ex in all_examples
        if ex["split"] in ("val", "test")
        and ex.get("records")
        and ex.get("reason_missing_fnum") not in ("multi-union", "multi-local")
    ]
    val_nonunion_texts = [
        ex["query"]
        for ex in all_examples
        if ex["split"] in ("val", "test")
        and (
            not ex.get("records")
            or ex.get("reason_missing_fnum") in ("multi-union", "multi-local")
        )
    ]

    # Add F7 employer negatives
    f7_path = (DATA_DIR / "f7.db").resolve()
    if f7_path.exists():
        conn = sqlite3.connect(str(f7_path))
        rows = conn.execute(
            "SELECT DISTINCT employer FROM f7 "
            "WHERE employer IS NOT NULL AND employer != '' "
            "ORDER BY employer"
        ).fetchall()
        conn.close()
        rng = random.Random(42)
        employers = rng.sample([r[0] for r in rows], min(20000, len(rows)))
        n_val = len(employers) // 10
        val_nonunion_texts += employers[:n_val]

    # Subsample union to match non-union count for balanced Platt fit
    # so that 0.5 probability is the natural decision threshold
    rng = random.Random(42)
    n_nonunion = len(val_nonunion_texts)
    if len(val_union) > n_nonunion:
        val_union_sample = rng.sample(val_union, n_nonunion)
    else:
        val_union_sample = val_union

    print(
        f"Val: {len(val_union_sample)} union (subsampled from {len(val_union)}), "
        f"{n_nonunion} non-union"
    )

    all_texts = [ex["query"] for ex in val_union_sample] + val_nonunion_texts
    all_labels = [1.0] * len(val_union_sample) + [0.0] * n_nonunion

    # Collect cosine similarities
    sims = []
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        features = [tokenize_for_arcface(text) for text in batch_texts]
        collated = _collate(features, vocab, device)
        with torch.no_grad():
            emb = encoder(*collated)
            sims.extend((emb @ centroid).cpu().tolist())

    sims_t = torch.tensor(sims)
    labels_t = torch.tensor(all_labels)

    # Fit sigmoid(a * sim + b)
    log_a = torch.nn.Parameter(torch.zeros(1))
    b = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_a, b], lr=0.01)

    raw_nll = F.binary_cross_entropy_with_logits(sims_t, labels_t).item()

    for step in range(500):
        optimizer.zero_grad()
        a = log_a.exp()
        logits = a * sims_t + b
        loss = F.binary_cross_entropy_with_logits(logits, labels_t)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 499:
            print(
                f"  Step {step:3d}: NLL={loss.item():.4f}  "
                f"a={log_a.exp().item():.4f}  b={b.item():.4f}"
            )

    platt_a = log_a.exp().item()
    platt_b = b.item()
    final_nll = F.binary_cross_entropy_with_logits(
        platt_a * sims_t + platt_b, labels_t
    ).item()
    print(f"  NLL: {raw_nll:.4f} -> {final_nll:.4f}")

    # Calibration table
    probs = torch.sigmoid(platt_a * sims_t + platt_b)
    correct = (labels_t == 1).float()
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\n  {'Conf range':>12}  {'Count':>6}  {'Accuracy':>8}  {'Avg conf':>8}")
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean().item()
        avg_conf = probs[mask].mean().item()
        print(
            f"  [{lo:.1f}, {hi:.1f})  {mask.sum().item():>6}  {acc:>8.1%}  {avg_conf:>8.4f}"
        )

    out_path = DATA_DIR / "platt_params.json"
    with open(out_path, "w") as f:
        json.dump({"platt_a": platt_a, "platt_b": platt_b}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
