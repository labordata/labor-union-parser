#!/usr/bin/env python3
"""Evaluate the union detector on labeled data.

Loads the trained union detector via Extractor and reports classification
metrics: accuracy, precision, recall, F1, ROC-AUC, and non-union recall.
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from labor_union_parser import Extractor

DATA_DIR = Path(__file__).parent / "data"


@click.command()
@click.option("--batch-size", default=256)
def main(batch_size):
    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    nonunion_df = pd.read_csv(DATA_DIR / "nonunion_examples.csv")

    # Use test split for union examples
    if "split" in df.columns:
        union_df = df[df["split"] == "test"].copy()
    else:
        union_df = df.copy()

    union_texts = union_df["text"].tolist()
    nonunion_texts = nonunion_df["text"].tolist()

    all_texts = union_texts + nonunion_texts
    all_labels = np.array([1] * len(union_texts) + [0] * len(nonunion_texts))

    print(f"Union examples: {len(union_texts)}")
    print(f"Non-union examples: {len(nonunion_texts)}")
    print(f"Total: {len(all_texts)}")

    # Run extraction
    extractor = Extractor()
    results = extractor.extract_batch(all_texts, batch_size=batch_size)

    scores = np.array([r["union_score"] for r in results])
    preds = np.array([1 if r["is_union"] else 0 for r in results])

    # Metrics at current threshold
    print(f"\nThreshold: {extractor.union_threshold}")
    print(
        f"\n{classification_report(all_labels, preds, target_names=['non-union', 'union'])}"
    )

    accuracy = accuracy_score(all_labels, preds)
    roc_auc = roc_auc_score(all_labels, scores)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Non-union recall specifically
    nonunion_mask = all_labels == 0
    nonunion_recall = (preds[nonunion_mask] == 0).mean()
    print(
        f"Non-union recall: {nonunion_recall:.4f} ({(preds[nonunion_mask] == 0).sum()}/{nonunion_mask.sum()})"
    )

    # Threshold analysis
    fpr, tpr, thresholds = roc_curve(all_labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.4f}")

    optimal_preds = (scores > optimal_threshold).astype(int)
    optimal_acc = accuracy_score(all_labels, optimal_preds)
    optimal_nonunion_recall = (optimal_preds[nonunion_mask] == 0).mean()
    print(f"  Accuracy at optimal:     {optimal_acc:.4f}")
    print(f"  Non-union recall at opt: {optimal_nonunion_recall:.4f}")


if __name__ == "__main__":
    main()
