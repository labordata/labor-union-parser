#!/usr/bin/env python3
"""End-to-end pipeline evaluation on labeled data.

Uses the Extractor API and evaluates combined is_union + f_num accuracy:
  - Union example correct: is_union=True AND f_num matches
  - Non-union example correct: is_union=False
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
    n_union = len(union_examples)
    n_non_union = len(non_union_examples)
    n_total = n_union + n_non_union
    print(f"Test: {n_union} union, {n_non_union} non-union")

    extractor = Extractor()

    # --- Run extractor on all test examples ---
    all_texts = [ex["query"] for ex in union_examples + non_union_examples]
    all_results = []
    pbar = tqdm(total=len(all_texts))
    for batch in itertools.batched(all_texts, BATCH_SIZE):
        batch = list(batch)
        all_results.extend(extractor.extract_batch(batch))
        pbar.update(len(batch))
    pbar.close()

    union_results = all_results[:n_union]
    non_union_results = all_results[n_union:]

    true_fnums = [str(ex["records"][0]["f_num"]) for ex in union_examples]
    score_fields = [
        "union_name",
        "desig_name",
        "f_num",
        "desig_num",
        "prefix",
        "suffix",
    ]

    # --- Union errors: is_union=False OR wrong f_num ---
    union_errors = []
    for text, true_fnum, result in zip(
        [ex["query"] for ex in union_examples], true_fnums, union_results
    ):
        is_correct = result["is_union"] and result["f_num"] == true_fnum
        if not is_correct:
            row = {
                "type": "false_negative" if not result["is_union"] else "wrong_match",
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
            union_errors.append(row)

    false_negatives = sum(1 for e in union_errors if e["type"] == "false_negative")
    wrong_matches = sum(1 for e in union_errors if e["type"] == "wrong_match")

    # --- Non-union errors: is_union=True ---
    false_positives = []
    for text, result in zip(
        [ex["query"] for ex in non_union_examples], non_union_results
    ):
        if result["is_union"]:
            false_positives.append(
                {
                    "type": "false_positive",
                    "text": text[:80],
                    "pred_fnum": result["f_num"],
                    "union_score": f"{result['union_score']:.4f}",
                    "match_score": f"{result['match_score']:.4f}",
                    "pred_union_name": result["union_name"],
                }
            )

    # --- Per-field accuracy (union examples where is_union=True) ---
    field_correct = {f: 0 for f in score_fields}
    field_total = 0
    for ex, result in zip(union_examples, union_results):
        if not result["is_union"]:
            continue
        field_total += 1
        rec = ex["records"][0]
        if result["union_name"] == rec.get("union_name", ""):
            field_correct["union_name"] += 1
        if result["desig_name"] == rec.get("desig_name", ""):
            field_correct["desig_name"] += 1
        if result["f_num"] == str(rec.get("f_num", "")):
            field_correct["f_num"] += 1
        if result["desig_num"] == str(rec.get("desig_num", 0) or ""):
            field_correct["desig_num"] += 1
        if result["prefix"] == str(rec.get("prefix", 0) or ""):
            field_correct["prefix"] += 1
        if result["suffix"] == (rec.get("suffix", "") or ""):
            field_correct["suffix"] += 1

    # --- Summary ---
    total_errors = len(union_errors) + len(false_positives)
    total_correct = n_total - total_errors

    print(
        f"\nEnd-to-end: {total_correct}/{n_total} = {total_correct / n_total:.4f} ({total_errors} errors)"
    )
    print(f"  False negatives (union, is_union=False): {false_negatives}")
    print(f"  Wrong match (union, is_union=True, wrong f_num): {wrong_matches}")
    print(f"  False positives (non-union, is_union=True): {len(false_positives)}")

    print(f"\nPer-field accuracy ({field_total} union examples with is_union=True):")
    for f in score_fields:
        acc = field_correct[f] / field_total if field_total else 0
        print(f"  {f:>12s}: {field_correct[f]}/{field_total} = {acc:.4f}")

    if false_positives:
        print("\nFalse positives:")
        for fp in false_positives:
            print(
                f"  {fp['text'][:60]}  -> {fp['pred_union_name']} (match_score={fp['match_score']})"
            )

    # --- Precision/recall at match_score thresholds ---
    # A prediction is "accepted" if is_union=True AND match_score >= threshold
    # True positive: union example accepted with correct f_num
    union_preds = []
    for true_fnum, result in zip(true_fnums, union_results):
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
    all_errors = union_errors + false_positives
    if all_errors:
        pd.DataFrame(all_errors).to_csv(
            SCRIPT_DIR / "data/pipeline_errors.csv", index=False
        )
        print(
            f"\nAll {len(all_errors)} errors saved to training/data/pipeline_errors.csv"
        )


if __name__ == "__main__":
    main()
