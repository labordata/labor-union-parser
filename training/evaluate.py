#!/usr/bin/env python3
"""End-to-end pipeline evaluation on labeled data.

Uses the Extractor API and evaluates:
  1. f_num accuracy on union examples (correct gazetteer match)
  2. Non-union rejection (correctly identifies non-union texts)
"""

import itertools
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from labor_union_parser import Extractor

SCRIPT_DIR = Path(__file__).parent

BATCH_SIZE = 256


def main():
    with open(SCRIPT_DIR / "data/training_examples.json") as f:
        all_examples = json.load(f)

    union_examples = [
        ex for ex in all_examples if ex["split"] == "test" and ex["records"]
    ]
    non_union_examples = [
        ex for ex in all_examples if ex["split"] == "test" and not ex["records"]
    ]
    print(f"Test: {len(union_examples)} union, {len(non_union_examples)} non-union")

    extractor = Extractor()

    # --- Evaluate union examples (f_num accuracy) ---
    texts = [ex["query"] for ex in union_examples]
    true_fnums = [str(ex["records"][0]["f_num"]) for ex in union_examples]

    results = []
    pbar = tqdm(total=len(texts), desc="union")
    for batch in itertools.batched(texts, BATCH_SIZE):
        batch = list(batch)
        results.extend(extractor.extract_batch(batch))
        pbar.update(len(batch))
    pbar.close()

    score_fields = [
        "union_name",
        "desig_name",
        "f_num",
        "desig_num",
        "prefix",
        "suffix",
    ]

    errors = []
    for text, true_fnum, result in zip(texts, true_fnums, results):
        if result["f_num"] != true_fnum:
            row = {
                "text": text[:80],
                "true_fnum": true_fnum,
                "pred_fnum": result["f_num"],
                "is_union": result["is_union"],
                "union_score": f"{result['union_score']:.4f}",
                "match_score": f"{result['match_score']:.4f}",
                "pred_union_name": result["union_name"],
            }
            fs = result.get("field_scores", {})
            for f in score_fields:
                val = fs.get(f)
                row[f"score_{f}"] = f"{val:.4f}" if val is not None else ""
            errors.append(row)

    n_union = len(union_examples)
    correct = n_union - len(errors)
    not_detected = sum(1 for e in errors if not e["is_union"])
    wrong_match = sum(1 for e in errors if e["is_union"])

    print(f"\nUnion f_num accuracy: {correct}/{n_union} = {correct/n_union:.4f}")
    print(f"  {len(errors)} errors:")
    print(f"    Not detected as union: {not_detected}")
    print(f"    Wrong gazetteer match: {wrong_match}")

    # --- Evaluate non-union examples (rejection accuracy) ---
    non_union_texts = [ex["query"] for ex in non_union_examples]

    non_union_results = []
    pbar = tqdm(total=len(non_union_texts), desc="non-union")
    for batch in itertools.batched(non_union_texts, BATCH_SIZE):
        batch = list(batch)
        non_union_results.extend(extractor.extract_batch(batch))
        pbar.update(len(batch))
    pbar.close()

    false_positives = []
    for text, result in zip(non_union_texts, non_union_results):
        if result["is_union"]:
            false_positives.append(
                {
                    "text": text[:80],
                    "pred_fnum": result["f_num"],
                    "union_score": f"{result['union_score']:.4f}",
                    "match_score": f"{result['match_score']:.4f}",
                    "pred_union_name": result["union_name"],
                }
            )

    n_non_union = len(non_union_examples)
    rejected = n_non_union - len(false_positives)
    print(
        f"\nNon-union rejection: {rejected}/{n_non_union} = {rejected/n_non_union:.4f}"
    )
    if false_positives:
        print(f"  {len(false_positives)} false positives:")
        for fp in false_positives:
            print(
                f"    {fp['text'][:60]}  -> {fp['pred_union_name']} (match_score={fp['match_score']})"
            )

    # --- End-to-end summary ---
    n_total = n_union + n_non_union
    total_errors = len(errors) + len(false_positives)
    total_correct = n_total - total_errors
    print(
        f"\nEnd-to-end: {total_correct}/{n_total} = {total_correct/n_total:.4f} ({total_errors} errors)"
    )

    # --- Precision/recall at match_score thresholds ---
    # Collect (match_score, is_correct) for all predictions flagged as union
    union_preds = []
    for true_fnum, result in zip(true_fnums, results):
        if result["is_union"]:
            union_preds.append((result["match_score"], result["f_num"] == true_fnum))
    for result in non_union_results:
        if result["is_union"]:
            union_preds.append((result["match_score"], False))

    thresholds = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    print(
        f"\n{'threshold':>10s}  {'precision':>9s}  {'recall':>9s}  {'f1':>9s}  {'accepted':>8s}  {'correct':>7s}  {'wrong':>5s}"
    )
    print("-" * 72)
    for t in thresholds:
        accepted = [(s, c) for s, c in union_preds if s >= t]
        tp = sum(1 for _, c in accepted if c)
        fp = sum(1 for _, c in accepted if not c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_union
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(
            f"{t:>10.2f}  {precision:>9.4f}  {recall:>9.4f}  {f1:>9.4f}  {tp + fp:>8d}  {tp:>7d}  {fp:>5d}"
        )

    # --- Save error details ---
    if errors:
        pd.DataFrame(errors).to_csv(
            SCRIPT_DIR / "data/pipeline_errors.csv", index=False
        )
    if false_positives:
        pd.DataFrame(false_positives).to_csv(
            SCRIPT_DIR / "data/pipeline_false_positives.csv", index=False
        )
    print("\nErrors saved to training/data/pipeline_errors.csv")


if __name__ == "__main__":
    main()
