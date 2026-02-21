#!/usr/bin/env python3
"""End-to-end pipeline evaluation on labeled data.

Uses the Extractor API and evaluates f_num accuracy as the primary metric.
"""

import itertools
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from labor_union_parser import Extractor

SCRIPT_DIR = Path(__file__).parent


def evaluate_pipeline(
    test_df: pd.DataFrame,
    extractor: Extractor | None = None,
    batch_size: int = 256,
    save_errors: bool = True,
) -> dict:
    """Evaluate the extraction pipeline on a test dataframe.

    Args:
        test_df: DataFrame with 'text' and 'f_num' columns
        extractor: Extractor instance (creates new one if None)
        batch_size: Batch size for extraction
        save_errors: Whether to save errors to CSV

    Returns:
        Dictionary with accuracy metrics
    """
    if extractor is None:
        extractor = Extractor()

    rows = list(test_df.itertuples(index=False))

    fnum_correct = 0
    union_detected = 0
    errors = []

    with tqdm(total=len(rows), desc="Evaluating") as pbar:
        for batch_rows in itertools.batched(rows, batch_size):
            texts = [row.text for row in batch_rows]
            results = extractor.extract_batch(texts)

            for row, result in zip(batch_rows, results):
                true_fnum = str(int(row.f_num))
                pred_fnum = result["f_num"]

                if result["is_union"]:
                    union_detected += 1

                if pred_fnum == true_fnum:
                    fnum_correct += 1
                else:
                    errors.append(
                        {
                            "text": row.text[:80],
                            "true_fnum": true_fnum,
                            "pred_fnum": pred_fnum,
                            "is_union": result["is_union"],
                            "union_score": f"{result['union_score']:.4f}",
                            "match_score": result["match_score"],
                            "pred_union_name": result["union_name"],
                        }
                    )

            pbar.update(len(batch_rows))

    total = len(test_df)

    if save_errors and errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(SCRIPT_DIR / "data/pipeline_errors.csv", index=False)

    return {
        "total": total,
        "fnum_correct": fnum_correct,
        "fnum_accuracy": fnum_correct / total if total else 0,
        "union_detected": union_detected,
        "union_detection_rate": union_detected / total if total else 0,
        "errors": errors,
    }


def print_results(metrics: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"\nf_num accuracy:     {metrics['fnum_correct']}/{metrics['total']} "
        f"= {metrics['fnum_accuracy']:.4f}"
    )
    print(
        f"Union detected:     {metrics['union_detected']}/{metrics['total']} "
        f"= {metrics['union_detection_rate']:.4f}"
    )

    errors = metrics.get("errors", [])
    if errors:
        not_detected = sum(1 for e in errors if not e["is_union"])
        wrong_match = sum(1 for e in errors if e["is_union"])

        print(f"\n{len(errors)} errors total:")
        print(f"  Not detected as union: {not_detected}")
        print(f"  Wrong gazetteer match: {wrong_match}")
        print("\nErrors saved to training/data/pipeline_errors.csv")


def main():
    df = pd.read_csv(SCRIPT_DIR / "data/labeled_data.csv")
    print(f"Total examples: {len(df)}")

    # Filter to test split with known f_num
    test_df = df[df["split"] == "test"].copy()
    test_df = test_df[test_df["f_num"].notna()].copy()
    print(f"Test split examples: {len(test_df)}")

    metrics = evaluate_pipeline(test_df)
    print_results(metrics)


if __name__ == "__main__":
    main()
