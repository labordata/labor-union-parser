#!/usr/bin/env python3
"""End-to-end pipeline evaluation on labeled data.

Evaluates the full Extractor pipeline on the test split:
  - Union detection: false negatives and false positives
  - f_num matching: accuracy among detected unions with known f_num
  - Union name prediction: accuracy among detected unions
"""

import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from labor_union_parser import Extractor

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
BATCH_SIZE = 256


def _is_union(ex):
    return ex["records"] and ex.get("reason_missing_fnum") not in (
        "multi-union",
        "multi-local",
    )


def _load_test_data():
    with open(DATA_DIR / "training_examples.json") as f:
        all_examples = json.load(f)

    test = [
        ex
        for ex in all_examples
        if ex["split"] == "test" and ex.get("source", "labeled") == "labeled"
    ]
    union = [ex for ex in test if _is_union(ex)]
    non_union = [ex for ex in test if not _is_union(ex)]
    return union, non_union


def _run_predictions(extractor, union_examples, non_union_examples):
    all_texts = [ex["query"] for ex in union_examples + non_union_examples]
    results = []
    for batch in itertools.batched(all_texts, BATCH_SIZE):
        results.extend(extractor.extract_batch(list(batch)))
    n = len(union_examples)
    return results[:n], results[n:]


def compute_test_metrics():
    """Run extractor on test split and return metrics dict.

    Used by README.md cog blocks and main() below.
    """
    union_examples, non_union_examples = _load_test_data()
    extractor = Extractor()
    union_results, non_union_results = _run_predictions(
        extractor, union_examples, non_union_examples
    )
    known_fnums = set(extractor.idx_to_fnum.values())

    # --- Error classification ---
    errors = []
    for ex, result in zip(union_examples, union_results):
        true_fnum = ex["records"][0]["f_num"]
        if ex.get("reason_missing_fnum") == "potentially resolvable":
            continue
        if not result["is_union"]:
            errors.append({"type": "false_negative", "ex": ex, "result": result})
        elif true_fnum != -100 and result["f_num"] != true_fnum:
            errors.append({"type": "wrong_match", "ex": ex, "result": result})

    for ex, result in zip(non_union_examples, non_union_results):
        if result["is_union"]:
            errors.append({"type": "false_positive", "ex": ex, "result": result})

    error_counts = Counter(e["type"] for e in errors)

    # --- f_num accuracy (detected unions with known f_num) ---
    fnum_correct = fnum_total = 0
    fnum_invocab_correct = fnum_invocab_total = 0
    for ex, result in zip(union_examples, union_results):
        if not result["is_union"]:
            continue
        true_fnum = ex["records"][0]["f_num"]
        if true_fnum == -100:
            continue
        fnum_total += 1
        if result["f_num"] == true_fnum:
            fnum_correct += 1
        if true_fnum in known_fnums:
            fnum_invocab_total += 1
            if result["f_num"] == true_fnum:
                fnum_invocab_correct += 1

    # --- Union name accuracy (detected unions with known union_name) ---
    union_correct = union_total = 0
    for ex, result in zip(union_examples, union_results):
        if not result["is_union"]:
            continue
        true_union = ex["records"][0].get("union_name", "")
        if not true_union:
            continue
        union_total += 1
        if result["union_name"] == true_union:
            union_correct += 1

    # --- is_union accuracy (union detection on all scored examples) ---
    is_union_correct = is_union_total = 0
    for ex, result in zip(union_examples, union_results):
        if ex.get("reason_missing_fnum") == "potentially resolvable":
            continue
        is_union_total += 1
        if result["is_union"]:
            is_union_correct += 1
    for ex, result in zip(non_union_examples, non_union_results):
        is_union_total += 1
        if not result["is_union"]:
            is_union_correct += 1

    # --- Scoring ---
    n_potentially_resolvable = sum(
        1
        for ex in union_examples
        if ex.get("reason_missing_fnum") == "potentially resolvable"
    )
    n_scored = len(union_examples) + len(non_union_examples) - n_potentially_resolvable

    return {
        "n_scored": n_scored,
        "wrong_matches": error_counts["wrong_match"],
        "false_negatives": error_counts["false_negative"],
        "false_positives": error_counts["false_positive"],
        "is_union_accuracy": (
            is_union_correct / is_union_total if is_union_total else 0
        ),
        "is_union_correct": is_union_correct,
        "is_union_total": is_union_total,
        "fnum_accuracy": fnum_correct / fnum_total if fnum_total else 0,
        "fnum_correct": fnum_correct,
        "fnum_total": fnum_total,
        "fnum_invocab_accuracy": (
            fnum_invocab_correct / fnum_invocab_total if fnum_invocab_total else 0
        ),
        "union_accuracy": union_correct / union_total if union_total else 0,
        "union_correct": union_correct,
        "union_total": union_total,
        # Raw data for main() reporting
        "_errors": errors,
        "_union_examples": union_examples,
        "_union_results": union_results,
        "_non_union_results": non_union_results,
    }


def main():
    m = compute_test_metrics()
    errors = m["_errors"]
    union_examples = m["_union_examples"]
    union_results = m["_union_results"]
    non_union_results = m["_non_union_results"]
    n_scored = m["n_scored"]

    print(f"Test: {len(union_examples)} union, {len(non_union_results)} non-union")

    total_errors = len(errors)
    print(
        f"\nEnd-to-end: {n_scored - total_errors}/{n_scored} "
        f"= {(n_scored - total_errors) / n_scored:.4f} ({total_errors} errors)"
    )
    print(f"  False negatives (union, is_union=False): {m['false_negatives']}")
    print(f"  Wrong match (union, wrong f_num): {m['wrong_matches']}")
    print(f"  False positives (non-union, is_union=True): {m['false_positives']}")

    n_is_union = sum(1 for r in union_results if r["is_union"])
    print(f"\nAccuracy ({n_is_union} union examples with is_union=True):")
    print(
        f"  f_num:           {m['fnum_correct']}/{m['fnum_total']} "
        f"= {m['fnum_accuracy']:.4f}"
    )
    print(f"  f_num (in-vocab): {m['fnum_invocab_accuracy']:.4f}")
    print(
        f"  union_name:      {m['union_correct']}/{m['union_total']} "
        f"= {m['union_accuracy']:.4f}"
    )

    # --- No-fnum breakdown by reason ---
    nofnum_by_reason = {}
    for ex, result in zip(union_examples, union_results):
        true_fnum = ex["records"][0]["f_num"]
        if true_fnum != -100:
            continue
        reason = ex.get("reason_missing_fnum", "unknown")
        if reason not in nofnum_by_reason:
            nofnum_by_reason[reason] = {"total": 0, "is_union": 0}
        nofnum_by_reason[reason]["total"] += 1
        if result["is_union"]:
            nofnum_by_reason[reason]["is_union"] += 1

    if nofnum_by_reason:
        n_nofnum = sum(b["total"] for b in nofnum_by_reason.values())
        print(f"\nNo-fnum breakdown ({n_nofnum} examples):")
        for reason in sorted(
            nofnum_by_reason, key=lambda r: -nofnum_by_reason[r]["total"]
        ):
            b = nofnum_by_reason[reason]
            not_union = b["total"] - b["is_union"]
            scored = "excluded" if reason == "potentially resolvable" else "scored"
            print(
                f"  {reason:>25s} ({scored}): {b['total']:>5d} total, "
                f"{b['is_union']:>5d} is_union, "
                f"{not_union:>5d} not_union"
            )

    # --- Save error details ---
    if errors:
        out_path = DATA_DIR / "pipeline_errors.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "type",
                    "text",
                    "true_fnum",
                    "pred_fnum",
                    "is_union",
                    "is_union_score",
                    "match_score",
                    "pred_union_name",
                ],
            )
            writer.writeheader()
            for e in errors:
                ex, result = e["ex"], e["result"]
                true_fnum = ex["records"][0]["f_num"] if ex.get("records") else ""
                writer.writerow(
                    {
                        "type": e["type"],
                        "text": ex["query"][:80],
                        "true_fnum": true_fnum,
                        "pred_fnum": result["f_num"],
                        "is_union": result["is_union"],
                        "is_union_score": f"{result['is_union_score']:.4f}",
                        "match_score": f"{result['match_score']:.4f}",
                        "pred_union_name": result["union_name"],
                    }
                )
        print(f"\n{len(errors)} errors saved to {out_path}")


if __name__ == "__main__":
    main()
