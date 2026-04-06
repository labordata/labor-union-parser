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

from labor_union_parser import Extractor

SCRIPT_DIR = Path(__file__).parent

BATCH_SIZE = 256


def compute_test_metrics():
    """Run extractor on test split and return metrics dict.

    Returns dict with keys:
        n_scored, wrong_matches, false_negatives, false_match_no_fnum,
        false_positives, n_is_union,
        union_errors (list), false_positive_errors (list)
    """
    with open(SCRIPT_DIR / "data/training_examples.json") as f:
        all_examples = json.load(f)

    def _is_union(ex):
        return ex["records"] and ex.get("reason_missing_fnum") != "multi-union"

    union_examples = [
        ex for ex in all_examples if ex["split"] == "test" and _is_union(ex)
    ]
    non_union_examples = [
        ex for ex in all_examples if ex["split"] == "test" and not _is_union(ex)
    ]
    n_union = len(union_examples)
    n_total = n_union + len(non_union_examples)

    extractor = Extractor()

    all_texts = [ex["query"] for ex in union_examples + non_union_examples]
    all_results = []
    for batch in itertools.batched(all_texts, BATCH_SIZE):
        all_results.extend(extractor.extract_batch(list(batch)))

    union_results = all_results[:n_union]
    non_union_results = all_results[n_union:]

    true_fnums = [ex["records"][0]["f_num"] for ex in union_examples]

    # --- Union errors ---
    union_errors = []
    for ex, true_fnum, result in zip(union_examples, true_fnums, union_results):
        text = ex["query"]
        reason = ex.get("reason_missing_fnum")
        if reason == "potentially resolvable":
            continue
        if true_fnum == -100:
            is_correct = result["is_union"] and not result["match_found"]
        else:
            is_correct = result["is_union"] and result["f_num"] == true_fnum
        if not is_correct:
            if not result["is_union"]:
                error_type = "false_negative"
            elif true_fnum == -100:
                error_type = "false_match_no_fnum"
            else:
                error_type = "wrong_match"
            union_errors.append(
                {
                    "type": error_type,
                    "text": text[:80],
                    "true_fnum": true_fnum,
                    "pred_fnum": result["f_num"],
                    "is_union": result["is_union"],
                    "union_score": f"{result['union_score']:.4f}",
                    "match_score": f"{result['match_score']:.4f}",
                    "pred_union_name": result["union_name"],
                }
            )

    false_negatives = sum(1 for e in union_errors if e["type"] == "false_negative")
    wrong_matches = sum(1 for e in union_errors if e["type"] == "wrong_match")
    false_match_no_fnum = sum(
        1 for e in union_errors if e["type"] == "false_match_no_fnum"
    )

    # --- Non-union errors ---
    false_positive_errors = []
    for text, result in zip(
        [ex["query"] for ex in non_union_examples], non_union_results
    ):
        if result["is_union"]:
            false_positive_errors.append(
                {
                    "type": "false_positive",
                    "text": text[:80],
                    "pred_fnum": result["f_num"],
                    "union_score": f"{result['union_score']:.4f}",
                    "match_score": f"{result['match_score']:.4f}",
                    "pred_union_name": result["union_name"],
                }
            )

    # --- f_num + union_name accuracy ---
    # Get the set of f_nums the model knows about
    known_fnums = set(extractor.idx_to_fnum.values())

    n_is_union = 0
    fnum_correct = 0
    fnum_total = 0
    fnum_invocab_correct = 0
    fnum_invocab_total = 0
    union_correct = 0
    union_total = 0
    for ex, result in zip(union_examples, union_results):
        if not result["is_union"]:
            continue
        n_is_union += 1
        rec = ex["records"][0]
        true_fnum = rec.get("f_num", -100)
        if true_fnum != -100:
            fnum_total += 1
            if result["f_num"] == true_fnum:
                fnum_correct += 1
            if true_fnum in known_fnums:
                fnum_invocab_total += 1
                if result["f_num"] == true_fnum:
                    fnum_invocab_correct += 1
        true_union = rec.get("union_name", "")
        if true_union:
            union_total += 1
            if result["union_name"] == true_union:
                union_correct += 1

    n_potentially_resolvable = sum(
        1
        for ex in union_examples
        if ex.get("reason_missing_fnum") == "potentially resolvable"
    )
    n_scored = n_total - n_potentially_resolvable

    return {
        "n_scored": n_scored,
        "wrong_matches": wrong_matches,
        "false_negatives": false_negatives,
        "false_match_no_fnum": false_match_no_fnum,
        "false_positives": len(false_positive_errors),
        "n_is_union": n_is_union,
        "fnum_accuracy": fnum_correct / fnum_total if fnum_total else 0,
        "fnum_invocab_accuracy": (
            fnum_invocab_correct / fnum_invocab_total if fnum_invocab_total else 0
        ),
        "fnum_invocab_correct": fnum_invocab_correct,
        "fnum_invocab_total": fnum_invocab_total,
        "union_accuracy": union_correct / union_total if union_total else 0,
        "fnum_correct": fnum_correct,
        "fnum_total": fnum_total,
        "union_correct": union_correct,
        "union_total": union_total,
        "union_errors": union_errors,
        "false_positive_errors": false_positive_errors,
        "union_examples": union_examples,
        "union_results": union_results,
        "non_union_results": non_union_results,
        "true_fnums": true_fnums,
    }


def main():
    m = compute_test_metrics()

    union_examples = m["union_examples"]
    union_results = m["union_results"]
    non_union_results = m["non_union_results"]
    true_fnums = m["true_fnums"]
    union_errors = m["union_errors"]
    false_positive_errors = m["false_positive_errors"]
    n_scored = m["n_scored"]
    total_errors = len(union_errors) + len(false_positive_errors)
    total_correct = n_scored - total_errors
    n_union = len(union_examples)
    n_non_union = len(non_union_results)

    print(f"Test: {n_union} union, {n_non_union} non-union")

    print(
        f"\nEnd-to-end: {total_correct}/{n_scored} = {total_correct / n_scored:.4f} "
        f"({total_errors} errors)"
    )
    print(f"  False negatives (union, is_union=False): {m['false_negatives']}")
    print(f"  Wrong match (union, is_union=True, wrong f_num): {m['wrong_matches']}")
    print(f"  False match on no-fnum examples: {m['false_match_no_fnum']}")
    print(f"  False positives (non-union, is_union=True): {m['false_positives']}")

    print(f"\nAccuracy ({m['n_is_union']} union examples with is_union=True):")
    print(
        f"  f_num:           {m['fnum_correct']}/{m['fnum_total']} = {m['fnum_accuracy']:.4f}"
    )
    print(
        f"  f_num (in-vocab): {m['fnum_invocab_correct']}/{m['fnum_invocab_total']} = {m['fnum_invocab_accuracy']:.4f}"
    )
    print(
        f"  union_name:      {m['union_correct']}/{m['union_total']} = {m['union_accuracy']:.4f}"
    )

    # --- match_found breakdown (only examples with known f_num) ---
    null_total = 0
    null_correct_fnum = 0
    match_total = 0
    match_correct_fnum = 0
    for true_fnum, result in zip(true_fnums, union_results):
        if not result["is_union"] or true_fnum == -100:
            continue
        if result["match_found"]:
            match_total += 1
            if result["f_num"] == true_fnum:
                match_correct_fnum += 1
        else:
            null_total += 1
            if result["f_num"] == true_fnum:
                null_correct_fnum += 1

    n_with_fnum = match_total + null_total
    print(f"\nMatch found breakdown ({n_with_fnum} with-fnum examples, is_union=True):")
    if match_total:
        print(
            f"  match_found=True:  {match_correct_fnum}/{match_total} correct f_num "
            f"({match_correct_fnum / match_total:.4f})"
        )
    if null_total:
        print(
            f"  match_found=False: {null_correct_fnum}/{null_total} correct f_num "
            f"({null_correct_fnum / null_total:.4f})"
        )

    # --- No-fnum breakdown by reason ---
    nofnum_by_reason = {}
    for ex, result in zip(union_examples, union_results):
        true_fnum = ex["records"][0]["f_num"]
        if true_fnum != -100:
            continue
        reason = ex.get("reason_missing_fnum", "unknown")
        if reason not in nofnum_by_reason:
            nofnum_by_reason[reason] = {
                "total": 0,
                "is_union": 0,
                "matched": 0,
                "null": 0,
            }
        bucket = nofnum_by_reason[reason]
        bucket["total"] += 1
        if result["is_union"]:
            bucket["is_union"] += 1
            if result["match_found"]:
                bucket["matched"] += 1
            else:
                bucket["null"] += 1

    n_nofnum = sum(b["total"] for b in nofnum_by_reason.values())
    print(f"\nNo-fnum breakdown ({n_nofnum} examples):")
    for reason in sorted(nofnum_by_reason, key=lambda r: -nofnum_by_reason[r]["total"]):
        b = nofnum_by_reason[reason]
        not_union = b["total"] - b["is_union"]
        scored = "excluded" if reason == "potentially resolvable" else "scored"
        print(
            f"  {reason:>25s} ({scored}): {b['total']:>5d} total, "
            f"{b['matched']:>5d} matched, "
            f"{b['null']:>5d} null, "
            f"{not_union:>5d} not_union"
        )

    # --- Precision/recall at match_score thresholds ---
    union_preds = []
    for ex, true_fnum, result in zip(union_examples, true_fnums, union_results):
        if ex.get("reason_missing_fnum") == "potentially resolvable":
            continue
        if result["is_union"]:
            if true_fnum == -100:
                correct = not result["match_found"]
            else:
                correct = result["f_num"] == true_fnum
            union_preds.append((result["match_score"], correct))
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
        recall = tp / n_scored
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(
            f"{t:>10.2f}  {precision:>9.4f}  {recall:>9.4f}  {f1:>9.4f}  {tp + fp:>8d}  {tp:>7d}  {fp:>5d}"
        )

    # --- Save error details ---
    all_errors = union_errors + false_positive_errors
    if all_errors:
        pd.DataFrame(all_errors).to_csv(
            SCRIPT_DIR / "data/pipeline_errors.csv", index=False
        )
        print(
            f"\nAll {len(all_errors)} errors saved to training/data/pipeline_errors.csv"
        )


if __name__ == "__main__":
    main()
