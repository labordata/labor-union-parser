"""Evaluate full pipeline on labeled data."""

import itertools
import sys
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
        test_df: DataFrame with 'text', 'aff_abbr', and 'desig_num' columns
        extractor: Extractor instance (creates new one if None)
        batch_size: Batch size for extraction
        save_errors: Whether to save errors to CSV

    Returns:
        Dictionary with accuracy metrics
    """
    if extractor is None:
        extractor = Extractor()

    # Filter to known affiliations
    known_df = test_df[test_df["aff_abbr"].isin(extractor.aff_list)].copy()

    # Counters
    aff_correct = 0
    aff_non_none_total = 0
    aff_non_none_correct = 0
    desig_correct = 0
    desig_correct_given_has_desig = 0
    desig_correct_given_no_desig = 0
    joint_correct = 0
    joint_non_none_correct = 0

    has_desig_total = 0
    no_desig_total = 0

    errors = []
    rows = list(known_df.itertuples(index=False))

    with tqdm(total=len(rows), desc="Evaluating", file=sys.stderr) as pbar:
        for batch_rows in itertools.batched(rows, batch_size):
            texts = [row.text for row in batch_rows]
            results = extractor.extract_batch(texts)

            for row, result in zip(batch_rows, results):
                true_aff = row.aff_abbr
                true_desig = (
                    str(row.desig_num).split(".")[0] if pd.notna(row.desig_num) else ""
                )
                true_desig = true_desig.lstrip("0") or ""

                pred_aff = result["affiliation"]
                pred_desig = (
                    result["designation"].lstrip("0") if result["designation"] else ""
                )

                aff_match = pred_aff == true_aff
                desig_match = pred_desig == true_desig

                if aff_match:
                    aff_correct += 1

                if pred_aff is not None:
                    aff_non_none_total += 1
                    if aff_match:
                        aff_non_none_correct += 1
                    if aff_match and desig_match:
                        joint_non_none_correct += 1

                if desig_match:
                    desig_correct += 1

                if true_desig:
                    has_desig_total += 1
                    if desig_match:
                        desig_correct_given_has_desig += 1
                else:
                    no_desig_total += 1
                    if desig_match:
                        desig_correct_given_no_desig += 1

                if aff_match and desig_match:
                    joint_correct += 1
                else:
                    errors.append(
                        {
                            "text": row.text[:80],
                            "true_aff": true_aff,
                            "pred_aff": pred_aff,
                            "aff_match": aff_match,
                            "true_desig": true_desig,
                            "pred_desig": pred_desig,
                            "desig_match": desig_match,
                        }
                    )

            pbar.update(len(batch_rows))

    total = len(known_df)

    # Save errors if requested
    if save_errors and errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(SCRIPT_DIR / "data/pipeline_errors.csv", index=False)

    return {
        "total": total,
        "aff_correct": aff_correct,
        "aff_accuracy": aff_correct / total if total else 0,
        "aff_non_none_total": aff_non_none_total,
        "aff_non_none_correct": aff_non_none_correct,
        "aff_non_none_accuracy": (
            aff_non_none_correct / aff_non_none_total if aff_non_none_total else 0
        ),
        "desig_correct": desig_correct,
        "desig_accuracy": desig_correct / total if total else 0,
        "has_desig_total": has_desig_total,
        "desig_correct_given_has_desig": desig_correct_given_has_desig,
        "desig_accuracy_with_desig": (
            desig_correct_given_has_desig / has_desig_total if has_desig_total else 0
        ),
        "no_desig_total": no_desig_total,
        "desig_correct_given_no_desig": desig_correct_given_no_desig,
        "desig_accuracy_without_desig": (
            desig_correct_given_no_desig / no_desig_total if no_desig_total else 0
        ),
        "joint_correct": joint_correct,
        "joint_accuracy": joint_correct / total if total else 0,
        "joint_non_none_correct": joint_non_none_correct,
        "joint_non_none_accuracy": (
            joint_non_none_correct / aff_non_none_total if aff_non_none_total else 0
        ),
        "errors": errors,
    }


def print_results(metrics: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"\nAffiliation accuracy:  {metrics['aff_correct']}/{metrics['total']} = {metrics['aff_accuracy']:.4f}"
    )
    print(
        f"  - Non-None preds:    {metrics['aff_non_none_correct']}/{metrics['aff_non_none_total']} = {metrics['aff_non_none_accuracy']:.4f}"
    )
    print(
        f"Designation accuracy:  {metrics['desig_correct']}/{metrics['total']} = {metrics['desig_accuracy']:.4f}"
    )
    print(
        f"  - With desig:        {metrics['desig_correct_given_has_desig']}/{metrics['has_desig_total']} = {metrics['desig_accuracy_with_desig']:.4f}"
    )
    print(
        f"  - Without desig:     {metrics['desig_correct_given_no_desig']}/{metrics['no_desig_total']} = {metrics['desig_accuracy_without_desig']:.4f}"
    )
    print(
        f"Joint accuracy:        {metrics['joint_correct']}/{metrics['total']} = {metrics['joint_accuracy']:.4f}"
    )
    print(
        f"  - Non-None preds:    {metrics['joint_non_none_correct']}/{metrics['aff_non_none_total']} = {metrics['joint_non_none_accuracy']:.4f}"
    )

    errors = metrics.get("errors", [])
    if errors:
        print(f"\nSaved {len(errors)} errors to training/data/pipeline_errors.csv")

        aff_only_errors = sum(
            1 for e in errors if not e["aff_match"] and e["desig_match"]
        )
        desig_only_errors = sum(
            1 for e in errors if e["aff_match"] and not e["desig_match"]
        )
        both_errors = sum(
            1 for e in errors if not e["aff_match"] and not e["desig_match"]
        )

        print("\nError breakdown:")
        print(f"  Affiliation only:    {aff_only_errors}")
        print(f"  Designation only:    {desig_only_errors}")
        print(f"  Both wrong:          {both_errors}")


def main():
    # Load data
    df = pd.read_csv(SCRIPT_DIR / "data/labeled_data.csv")
    print(f"Total examples: {len(df)}")

    # Filter to test split only
    test_df = df[df["split"] == "test"].copy()
    print(f"Test split examples: {len(test_df)}")

    # Run evaluation
    metrics = evaluate_pipeline(test_df)
    print_results(metrics)


if __name__ == "__main__":
    main()
