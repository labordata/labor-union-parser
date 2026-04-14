#!/usr/bin/env python3
"""Fit temperature scaling for ArcFace f_num and union_name heads.

Reads:
  - training/data/arcface_classifier.ckpt
  - training/data/training_examples.json

Writes:
  - training/data/arcface_temperatures.json
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from labor_union_parser.arcface_model import ArcFaceModel
from labor_union_parser.tokenizer import NUM_BLOOM_HASHES, tokenize_for_arcface

DATA_DIR = Path(__file__).parent / "data"
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


def fit_temperature(logits, targets, name, lr=0.01, steps=500):
    nll_before = F.cross_entropy(logits, targets).item()

    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_temp], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        temp = log_temp.exp()
        loss = F.cross_entropy(logits / temp, targets)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: NLL={loss.item():.4f}  T={temp.item():.4f}")

    temp = log_temp.exp().item()
    nll_after = F.cross_entropy(logits / temp, targets).item()
    print(f"  {name}: T={temp:.4f}, NLL {nll_before:.4f} -> {nll_after:.4f}")
    return temp


def show_calibration(logits, targets, temp, name):
    probs = F.softmax(logits / temp, dim=1)
    top_probs, top_preds = probs.max(dim=1)
    correct = (top_preds == targets).float()

    bins = [
        (0, 0.5),
        (0.5, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 0.95),
        (0.95, 0.99),
        (0.99, 1.0),
    ]
    print(f"\n  {name} calibration (T={temp:.4f}):")
    print(f"  {'Conf range':>12}  {'Count':>6}  {'Accuracy':>8}  {'Avg conf':>8}")
    for lo, hi in bins:
        mask = (top_probs >= lo) & (top_probs < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean().item()
        avg_conf = top_probs[mask].mean().item()
        print(
            f"  [{lo:.2f}, {hi:.2f})  {mask.sum().item():>6}  {acc:>8.1%}  {avg_conf:>8.4f}"
        )


def main():
    device = "cpu"

    with open(DATA_DIR / "training_examples.json") as f:
        all_examples = json.load(f)

    ckpt = torch.load(
        DATA_DIR / "arcface_classifier.ckpt", map_location=device, weights_only=False
    )

    fnum_to_idx = ckpt["fnum_to_idx"]
    field_vocabs = ckpt["field_vocabs"]
    union_vocab = {v: idx - 1 for v, idx in field_vocabs["union_name"].items()}
    vocab = ckpt["vocab"]

    model = ArcFaceModel(
        n_classes=ckpt["n_classes"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        n_buckets=ckpt["n_buckets"],
        vocab_size=len(vocab),
        scale=ckpt["arcface_scale"],
        field_sizes=ckpt["field_sizes"],
    )

    sd = {
        k: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder.") or k.startswith("classifier.") or k == "union_scale"
    }
    for buf_key in (
        "classifier.field_map",
        "classifier.desig_bloom",
        "classifier.proto_to_class",
    ):
        sd.pop(buf_key, None)
    model.load_state_dict(sd, strict=False)
    model.classifier.field_map = ckpt["field_map"]
    model.classifier.desig_bloom = ckpt["desig_bloom"]
    model.classifier.proto_to_class = ckpt["proto_to_class"]
    model.to(device)
    model.eval()

    # Val examples
    val_fnum = [
        ex
        for ex in all_examples
        if ex["split"] == "val"
        and ex.get("records")
        and ex["records"][0]["f_num"] != -100
        and ex["records"][0]["f_num"] in fnum_to_idx
    ]
    val_union = [
        ex
        for ex in all_examples
        if ex["split"] == "val"
        and ex.get("records")
        and ex["records"][0].get("union_name", "")
        and ex["records"][0]["union_name"] in union_vocab
    ]

    print(f"Val: {len(val_fnum)} with f_num, {len(val_union)} with union_name")

    # Collect f_num logits
    print("\nCollecting f_num logits...")
    fnum_logits_list, fnum_targets_list = [], []
    for i in range(0, len(val_fnum), BATCH_SIZE):
        batch_ex = val_fnum[i : i + BATCH_SIZE]
        features = [tokenize_for_arcface(ex["query"]) for ex in batch_ex]
        collated = _collate(features, vocab, device)
        targets = [fnum_to_idx[ex["records"][0]["f_num"]] for ex in batch_ex]
        with torch.no_grad():
            class_logits, _ = model(*collated)
        fnum_logits_list.append(class_logits.cpu())
        fnum_targets_list.append(torch.tensor(targets, dtype=torch.long))

    # Collect union logits
    print("Collecting union logits...")
    union_logits_list, union_targets_list = [], []
    for i in range(0, len(val_union), BATCH_SIZE):
        batch_ex = val_union[i : i + BATCH_SIZE]
        features = [tokenize_for_arcface(ex["query"]) for ex in batch_ex]
        collated = _collate(features, vocab, device)
        targets = [union_vocab[ex["records"][0]["union_name"]] for ex in batch_ex]
        with torch.no_grad():
            _, union_logits = model(*collated)
        union_logits_list.append(union_logits.cpu())
        union_targets_list.append(torch.tensor(targets, dtype=torch.long))

    # Fit
    print("\n--- f_num temperature ---")
    fnum_temp = fit_temperature(
        torch.cat(fnum_logits_list), torch.cat(fnum_targets_list), "f_num"
    )
    print("\n--- union_name temperature ---")
    union_temp = fit_temperature(
        torch.cat(union_logits_list), torch.cat(union_targets_list), "union_name"
    )

    show_calibration(
        torch.cat(fnum_logits_list), torch.cat(fnum_targets_list), fnum_temp, "f_num"
    )
    show_calibration(
        torch.cat(union_logits_list),
        torch.cat(union_targets_list),
        union_temp,
        "union_name",
    )

    out_path = DATA_DIR / "arcface_temperatures.json"
    with open(out_path, "w") as f:
        json.dump(
            {"fnum_temperature": fnum_temp, "union_temperature": union_temp},
            f,
            indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
