#!/usr/bin/env python3
"""Optimize scoring weights via precomputed scores + scipy.

One forward pass to cache per-query:
  - base_scores: sum of 5 temperature-scaled non-f_num head log-probs (n_records,)
  - fnum_lp: f_num log-prob per record (n_records,)
  - fnum_known: bool mask per record (n_records,)
  - target_idx: index of correct record

Then optimize: score = base + w_known * fnum_lp * known + w_unknown * fnum_lp * ~known
"""

import json
import math
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_structured_classifier import (
    StructuredDataset,
    collate_fn,
)

from labor_union_parser.classifier import (
    FIELDS,
    MAX_TOKENS,
    POINTER_FIELDS,
    StructuredClassifier,
)
from labor_union_parser.scoring import (
    POINTER_NOT_FOUND_LOG_PROB,
    build_gazetteer_matrix,
    build_pointer_lookup,
)
from labor_union_parser.tokenizer import smart_truncate_nonspace

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"

DEVICE = torch.accelerator.current_accelerator() or torch.device("cpu")


def load_model(ckpt):
    model = StructuredClassifier(
        field_sizes=ckpt["field_sizes"],
        d_model=ckpt["d_model"],
        n_heads=4,
        n_layers=ckpt["n_layers"],
        ff_dim=ckpt["d_model"] * 2,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@click.command()
@click.option("--batch-size", default=256)
@click.option("--split", default="val", type=click.Choice(["val", "test"]))
def main(batch_size, split):
    print(f"Device: {DEVICE}")

    # Load checkpoint
    weights_dir = (
        Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
    )
    ckpt = torch.load(
        weights_dir / "structured_classifier.pt",
        weights_only=False,
        map_location=DEVICE,
    )

    model = load_model(ckpt)
    field_vocabs = ckpt["field_vocabs"]
    fnum_to_records = ckpt["gazetteer"]

    # Load temperatures
    temps_path = DATA_DIR / "temperatures.json"
    with open(temps_path) as f:
        temps = json.load(f)
    head_temperatures = {f: temps[f] for f in FIELDS if f != "f_num" and f in temps}
    fnum_a, fnum_b = temps["f_num_a"], temps["f_num_b"]

    # Build per-class f_num temperature vector
    fnum_vocab = field_vocabs["f_num"]
    fnum_train_counts = ckpt["fnum_train_counts"]
    n_fnum_classes = len(fnum_vocab)
    fnum_class_temps = torch.ones(n_fnum_classes, device=DEVICE)
    for fnum_val, class_idx in fnum_vocab.items():
        count = fnum_train_counts.get(str(fnum_val), 0)
        fnum_class_temps[class_idx] = math.exp(fnum_a + fnum_b * math.log1p(count))

    # Build gazetteer
    field_indices, field_known, record_fnums, records_list = build_gazetteer_matrix(
        fnum_to_records, field_vocabs
    )
    record_fnums_array = np.array(record_fnums)
    n_records = len(record_fnums)
    print(f"Scoring against {n_records} gazetteer records")

    field_indices = {f: t.to(DEVICE) for f, t in field_indices.items()}
    field_known = {f: t.to(DEVICE) for f, t in field_known.items()}

    # Pointer lookups
    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    # Load eval data
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    eval_examples = [
        ex for ex in all_examples if ex["split"] == split and ex["records"]
    ]
    print(f"Eval examples: {len(eval_examples)}")

    eval_ds = StructuredDataset(eval_examples, field_vocabs)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Precompute target f_nums and token strings
    target_fnums = []
    query_token_strings = []
    for ex in eval_examples:
        rec = ex["records"][0]
        target_fnums.append(rec["f_num"])
        tokens = smart_truncate_nonspace(ex["query"])
        query_token_strings.append([t["token"] for t in tokens])

    # One forward pass: cache base_scores, fnum_lp, fnum_known per query
    # Features: 5 non-fnum heads + fnum log-prob + fnum_unknown indicator
    non_fnum_fields = [f for f in FIELDS if f != "f_num"]
    feature_names = non_fnum_fields + ["fnum_lp", "fnum_unknown"]
    n_features = len(feature_names)
    print(f"Features ({n_features}): {feature_names}")

    print("Precomputing per-field scores...")
    # Store per-field scores: dict of field -> list of (n_records,) arrays
    all_field_scores = {f: [] for f in non_fnum_fields}
    all_fnum_lp = []
    fnum_known_np = field_known["f_num"].cpu().numpy()  # same for all queries

    example_idx = 0
    with torch.no_grad():
        for inputs, labels in eval_loader:
            char_ids = inputs["char_ids"].to(DEVICE)
            mask = inputs["mask"].to(DEVICE)
            logits = model(char_ids, mask)

            # Apply temperatures
            scaled_logits = {}
            for f in FIELDS:
                if f == "f_num":
                    scaled_logits[f] = logits[f] / fnum_class_temps
                else:
                    scaled_logits[f] = logits[f] / head_temperatures[f]
            log_probs = {f: F.log_softmax(scaled_logits[f], dim=-1) for f in FIELDS}

            batch_size_actual = char_ids.shape[0]
            for i in range(batch_size_actual):
                for f in non_fnum_fields:
                    if f not in POINTER_FIELDS:
                        field_lp = log_probs[f][i][field_indices[f]]
                        vocab_size = log_probs[f].shape[-1]
                        floor_lp = -math.log(vocab_size)
                        field_lp = torch.where(field_known[f], field_lp, floor_lp)
                        all_field_scores[f].append(field_lp.cpu().numpy())
                    else:
                        query_toks = query_token_strings[example_idx]
                        lp = log_probs[f][i]
                        tok_to_pos = {}
                        for pos, tok in enumerate(query_toks):
                            if tok and tok not in tok_to_pos:
                                tok_to_pos[tok] = pos
                        field_scores = torch.full(
                            (n_records,), POINTER_NOT_FOUND_LOG_PROB[f], device=DEVICE
                        )
                        none_idx = pointer_none_indices[f]
                        if len(none_idx) > 0:
                            field_scores[none_idx] = lp[MAX_TOKENS]
                        val_to_idx = pointer_val_to_indices[f]
                        for tok, pos in tok_to_pos.items():
                            rec_indices = val_to_idx.get(tok)
                            if rec_indices is not None:
                                field_scores[rec_indices] = lp[pos]
                        all_field_scores[f].append(field_scores.cpu().numpy())

                # f_num log-prob per record
                fnum_lp = log_probs["f_num"][i][field_indices["f_num"]]
                fnum_vocab_size = log_probs["f_num"].shape[-1]
                fnum_floor = -math.log(fnum_vocab_size)
                fnum_lp = torch.where(field_known["f_num"], fnum_lp, fnum_floor)
                all_fnum_lp.append(fnum_lp.cpu().numpy())

                example_idx += 1

            if example_idx % 1000 < batch_size:
                print(f"  {example_idx}/{len(eval_examples)}")

    # Stack into arrays: each (n_queries, n_records)
    field_arrays = {f: np.stack(all_field_scores[f]) for f in non_fnum_fields}
    fnum_lp = np.stack(all_fnum_lp)

    n_queries = len(eval_examples)
    print(
        f"\nPrecomputed {n_queries} queries x {n_records} records x {n_features} features"
    )

    target_fnums_np = np.array(target_fnums)
    fnum_unknown_f = (~fnum_known_np).astype(np.float32)  # (n_records,) indicator

    def compute_top1(w):
        # score = sum(w[0:5] * head_lps) + w[5] * fnum_lp + w[6] * fnum_unknown_indicator
        scores = np.zeros((n_queries, n_records), dtype=np.float32)
        for j, f in enumerate(non_fnum_fields):
            scores += w[j] * field_arrays[f]
        scores += w[5] * fnum_lp
        scores += w[6] * fnum_unknown_f  # broadcast (n_records,) across queries
        preds = scores.argmax(axis=1)
        pred_fnums = record_fnums_array[preds]
        return (pred_fnums == target_fnums_np).sum()

    # Precompute base score (5 non-fnum heads, all weight=1.0)
    # so we only vary fnum_lp weight and fnum_unknown penalty
    print("Precomputing base scores (5 heads summed)...")
    base_scores = np.zeros((n_queries, n_records), dtype=np.float32)
    for f in non_fnum_fields:
        base_scores += field_arrays[f]

    def compute_top1_fast(wf, wu):
        scores = base_scores + wf * fnum_lp + wu * fnum_unknown_f
        preds = scores.argmax(axis=1)
        return (record_fnums_array[preds] == target_fnums_np).sum()

    # Grid search over fnum_lp weight and fnum_unknown penalty
    print("\n=== Grid search: fnum_lp weight x fnum_unknown penalty ===")
    best_correct = 0
    best_wf = 0.0
    best_wu = 0.0
    for wf in np.arange(0.0, 1.01, 0.1):
        row = []
        for wu in np.arange(-5.0, 1.01, 0.5):
            correct = compute_top1_fast(wf, wu)
            row.append(f"{n_queries - correct:3d}")
            if correct > best_correct:
                best_correct = correct
                best_wf = wf
                best_wu = wu
        print(f"  wf={wf:.1f}: [{', '.join(row)}]  (wu=-5.0 to 1.0)")

    print(f"\nCoarse best: fnum_lp={best_wf:.2f}, fnum_unknown={best_wu:.2f}")
    print(
        f"Top-1: {best_correct}/{n_queries} = {best_correct/n_queries:.4f} ({n_queries - best_correct} errors)"
    )

    # Fine grid around best
    print("\n=== Fine grid around best ===")
    for wf in np.arange(best_wf - 0.15, best_wf + 0.16, 0.02):
        for wu in np.arange(best_wu - 1.0, best_wu + 1.01, 0.1):
            correct = compute_top1_fast(wf, wu)
            if correct > best_correct:
                best_correct = correct
                best_wf = wf
                best_wu = wu
                print(
                    f"  New best: fnum_lp={wf:.2f}, fnum_unknown={wu:.2f} -> {correct}/{n_queries} ({n_queries - correct} errors)"
                )

    print("\nFinal best:")
    print("  5 non-fnum heads: w=1.0 (temperature-scaled)")
    print(f"  fnum_lp: w={best_wf:.4f}")
    print(f"  fnum_unknown penalty: w={best_wu:.4f}")
    print(
        f"  Top-1: {best_correct}/{n_queries} = {best_correct/n_queries:.4f} ({n_queries - best_correct} errors)"
    )


if __name__ == "__main__":
    main()
