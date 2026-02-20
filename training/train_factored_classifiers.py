#!/usr/bin/env python3
"""
Train per-field text classifiers (query → field_value) and save as pickles.

These classifiers predict union_name and desig_name from query text using
char n-gram TF-IDF features + multinomial logistic regression. Their predicted
probabilities are then used as additional features in the LightGBM reranker.

Key insight: char n-grams capture abbreviation patterns like "IBT"→TEAMSTERS
or "SCDCL"→LABORERS that trigram Jaccard/TF-IDF cosine cannot learn.
"""

import json
import pickle
from pathlib import Path

import click
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"

CLASSIFIERS = {
    "union_name": {
        "path": DATA_DIR / "union_name_classifier.pkl",
        "vocab_key": "union_name_to_idx",
        "record_field": "union_name",
    },
    "desig_name": {
        "path": DATA_DIR / "desig_name_classifier.pkl",
        "vocab_key": "desig_name_to_idx",
        "record_field": "desig_name",
    },
    "prefix": {
        "path": DATA_DIR / "prefix_classifier.pkl",
        "vocab_key": None,  # built from data
        "record_field": "prefix",
    },
    "suffix": {
        "path": DATA_DIR / "suffix_classifier.pkl",
        "vocab_key": None,  # built from data
        "record_field": "suffix",
    },
    "union_unit": {
        "path": DATA_DIR / "union_unit_classifier.pkl",
        "vocab_key": None,  # composite label built from data
        "record_field": "union_unit",  # virtual field
    },
    "desig_num": {
        "path": DATA_DIR / "desig_num_classifier.pkl",
        "vocab_key": None,  # built from data
        "record_field": "desig_num",
    },
}


def top_k_accuracy(clf, X, y, k=5):
    """Compute top-k accuracy: correct if true label is in top-k predictions."""
    proba = clf.predict_proba(X)
    # argsort gives column indices; map back to class labels via clf.classes_
    top_k_col_indices = np.argsort(proba, axis=1)[:, -k:]
    top_k_classes = clf.classes_[top_k_col_indices]
    return np.mean([y[i] in top_k_classes[i] for i in range(len(y))])


def train_classifier(
    name,
    texts_train,
    labels_train,
    texts_val,
    labels_val,
    texts_test,
    labels_test,
    n_classes,
    null_label=None,
    null_weight=None,
):
    """Train a char n-gram TF-IDF + LogReg classifier."""
    print(f"\n{'='*60}")
    print(f"Training {name} classifier ({n_classes} classes)")
    print(
        f"  Train: {len(texts_train)}, Val: {len(labels_val)}, Test: {len(labels_test)}"
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=50000,
        sublinear_tf=True,
    )

    print("  Fitting TF-IDF vectorizer...")
    X_train = vectorizer.fit_transform(texts_train)
    X_val = vectorizer.transform(texts_val)
    X_test = vectorizer.transform(texts_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}, matrix: {X_train.shape}")

    # Build class_weight if null_weight is specified
    cw = None
    if null_weight is not None and null_label is not None:
        # Downweight the null class relative to all others
        unique_labels = set(labels_train)
        cw = {lab: 1.0 for lab in unique_labels}
        cw[null_label] = null_weight
        print(f"  class_weight: null({null_label})={null_weight}, others=1.0")

    print("  Training LogisticRegression...")
    clf = LogisticRegression(
        solver="saga",
        max_iter=100,
        tol=1e-3,
        class_weight=cw,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, labels_train)

    # Evaluate
    null_idx = null_label
    for split_name, X, y in [("Val", X_val, labels_val), ("Test", X_test, labels_test)]:
        preds = clf.predict(X)
        top1 = accuracy_score(y, preds)
        top5 = top_k_accuracy(clf, X, y, k=5)
        msg = f"  {split_name} top-1: {top1:.4f}, top-5: {top5:.4f}"
        # Non-null accuracy
        if null_idx is not None:
            non_null_mask = y != null_idx
            n_non_null = non_null_mask.sum()
            if n_non_null > 0:
                non_null_acc = accuracy_score(y[non_null_mask], preds[non_null_mask])
                msg += f", non-null top-1: {non_null_acc:.4f} ({n_non_null} examples)"
        print(msg)

    return vectorizer, clf


def _get_field_value(record, field):
    """Get a field value, handling composite fields like union_unit."""
    if field == "union_unit":
        uid = record.get("unit_id", "")
        if uid:
            return f"{record['union_name']}|{uid}"
        else:
            return record["union_name"]
    return record.get(field, "")


def build_value_to_idx(examples, field):
    """Build a value→idx mapping from training data for fields without a vocab entry."""
    values = set()
    for ex in examples:
        if ex["records"]:
            values.add(_get_field_value(ex["records"][0], field))
    return {v: i for i, v in enumerate(sorted(values, key=str))}


def extract_labels(examples, field, field_to_idx):
    """Extract (text, label) pairs for a given field."""
    texts = []
    labels = []
    for ex in examples:
        if not ex["records"]:
            continue
        value = _get_field_value(ex["records"][0], field)
        idx = field_to_idx.get(value)
        if idx is None:
            continue
        texts.append(ex["query"])
        labels.append(idx)
    return texts, np.array(labels)


@click.command()
@click.option(
    "--only",
    multiple=True,
    type=click.Choice(list(CLASSIFIERS.keys())),
    help="Train only these classifiers (can repeat). Default: all.",
)
@click.option(
    "--null-weight",
    type=float,
    default=None,
    help="Weight for the null class (0 or ''). Non-null classes get weight 1.0. "
    "E.g. --null-weight 0.1 downweights null 10x relative to non-null.",
)
def main(only, null_weight):
    print("Loading data...")
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    # Build per-split datasets
    splits = {"train": [], "val": [], "test": []}
    for ex in all_examples:
        splits[ex["split"]].append(ex)

    print(
        f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
    )

    to_train = list(only) if only else list(CLASSIFIERS.keys())
    print(f"Training: {to_train}")

    for name in to_train:
        cfg = CLASSIFIERS[name]

        # Get or build value→idx mapping
        if cfg["vocab_key"] is not None:
            value_to_idx = vocab[cfg["vocab_key"]]
        else:
            value_to_idx = build_value_to_idx(splits["train"], cfg["record_field"])

        idx_to_name = {v: k for k, v in value_to_idx.items()}

        # Extract labels
        train_texts, train_labels = extract_labels(
            splits["train"], cfg["record_field"], value_to_idx
        )
        val_texts, val_labels = extract_labels(
            splits["val"], cfg["record_field"], value_to_idx
        )
        test_texts, test_labels = extract_labels(
            splits["test"], cfg["record_field"], value_to_idx
        )

        # Determine null label index for non-null accuracy reporting
        null_values = {
            "prefix": 0,
            "suffix": "",
            "desig_name": "LU",
            "desig_num": 0,
        }
        null_val = null_values.get(name)
        null_label = value_to_idx.get(null_val) if null_val is not None else None

        vec, clf = train_classifier(
            name,
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            test_texts,
            test_labels,
            n_classes=len(value_to_idx),
            null_label=null_label,
            null_weight=null_weight,
        )

        # Save
        out_path = cfg["path"]
        print(f"\nSaving {name} classifier to {out_path}")
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": vec,
                    "classifier": clf,
                    "idx_to_name": idx_to_name,
                },
                f,
            )
        print(f"  Size: {out_path.stat().st_size / 1e6:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
