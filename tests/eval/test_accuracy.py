"""End-to-end accuracy regression tests.

Run with: pytest -m eval
Skip by default in normal test runs via pyproject.toml addopts.
"""

import itertools
import json
from pathlib import Path

import pytest

from labor_union_parser import Extractor

DATA_PATH = (
    Path(__file__).parent.parent.parent / "training" / "data" / "training_examples.json"
)

BATCH_SIZE = 256

# --- Thresholds (max allowed errors) ---
# Update these when model weights are intentionally changed.

# Union detection
MAX_FALSE_NEGATIVES = 5  # union text → is_union=False (currently 1)
MAX_FALSE_POSITIVES = 10  # non-union text → is_union=True (currently 7)

# Per-field accuracy (among union examples with is_union=True)
MAX_FIELD_ERRORS = {
    "union_name": 50,  # currently 41
    "desig_name": 35,  # currently 26
    "f_num": 110,  # currently 104
    "desig_num": 30,  # currently 20
    "prefix": 50,  # currently 40
    "suffix": 65,  # currently 55
}

# End-to-end (is_union correct AND f_num correct)
MAX_WRONG_MATCHES = 110  # currently 104


# --- Fixtures ---


@pytest.fixture(scope="module")
def eval_data():
    if not DATA_PATH.exists():
        pytest.skip(f"Eval data not found: {DATA_PATH}")

    with open(DATA_PATH) as f:
        all_examples = json.load(f)

    union = [ex for ex in all_examples if ex["split"] == "test" and ex["records"]]
    non_union = [
        ex for ex in all_examples if ex["split"] == "test" and not ex["records"]
    ]
    return union, non_union


@pytest.fixture(scope="module")
def extractor():
    return Extractor()


@pytest.fixture(scope="module")
def predictions(eval_data, extractor):
    union, non_union = eval_data
    all_texts = [ex["query"] for ex in union + non_union]

    results = []
    for batch in itertools.batched(all_texts, BATCH_SIZE):
        results.extend(extractor.extract_batch(list(batch)))

    n_union = len(union)
    return results[:n_union], results[n_union:]


# --- Union detection tests ---


@pytest.mark.eval
class TestUnionDetection:

    def test_false_negatives(self, eval_data, predictions):
        """Union texts misclassified as non-union."""
        union_results, _ = predictions
        fn = sum(1 for r in union_results if not r["is_union"])
        assert (
            fn <= MAX_FALSE_NEGATIVES
        ), f"False negatives: {fn} (threshold: {MAX_FALSE_NEGATIVES})"

    def test_false_positives(self, eval_data, predictions):
        """Non-union texts misclassified as union."""
        _, non_union_results = predictions
        fp = sum(1 for r in non_union_results if r["is_union"])
        assert (
            fp <= MAX_FALSE_POSITIVES
        ), f"False positives: {fp} (threshold: {MAX_FALSE_POSITIVES})"


# --- Per-field accuracy tests ---


def _field_truth(rec, field):
    """Extract ground truth value for a field, matching Extractor output format."""
    if field == "f_num":
        return str(rec.get("f_num", ""))
    if field == "desig_num":
        return str(rec.get("desig_num", 0) or "")
    if field == "prefix":
        return str(rec.get("prefix", 0) or "")
    if field == "suffix":
        return rec.get("suffix", "") or ""
    return rec.get(field, "")


@pytest.mark.eval
class TestFieldAccuracy:

    @pytest.mark.parametrize("field", MAX_FIELD_ERRORS.keys())
    def test_field_errors(self, field, eval_data, predictions):
        """Per-field error count on union examples where is_union=True."""
        union, _ = eval_data
        union_results, _ = predictions
        threshold = MAX_FIELD_ERRORS[field]

        errors = 0
        for ex, result in zip(union, union_results):
            if not result["is_union"]:
                continue
            if result[field] != _field_truth(ex["records"][0], field):
                errors += 1

        assert errors <= threshold, f"{field} errors: {errors} (threshold: {threshold})"


# --- End-to-end test ---


@pytest.mark.eval
class TestEndToEnd:

    def test_wrong_fnum_matches(self, eval_data, predictions):
        """Union texts where is_union=True but f_num is wrong."""
        union, _ = eval_data
        union_results, _ = predictions

        wrong = 0
        for ex, result in zip(union, union_results):
            if result["is_union"] and result["f_num"] != str(ex["records"][0]["f_num"]):
                wrong += 1

        assert (
            wrong <= MAX_WRONG_MATCHES
        ), f"Wrong f_num matches: {wrong} (threshold: {MAX_WRONG_MATCHES})"
