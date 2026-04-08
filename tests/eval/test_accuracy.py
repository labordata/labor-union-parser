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

# --- Thresholds (max allowed error rates) ---
# Update these when model weights are intentionally changed.
TOLERANCE = 0.05
ABSOLUTE_FLOOR = 0.002

# Union detection
MAX_FALSE_NEGATIVE_RATE = 0.0060  # among union examples with known f_num
MAX_FALSE_POSITIVE_RATE = 0.4028  # among non-union examples

# f_num matching (among union examples with is_union=True and known f_num)
MAX_FNUM_ERROR_RATE = 0.0400

# Union name prediction (among union examples with is_union=True)
MAX_UNION_NAME_ERROR_RATE = 0.0380

# End-to-end (is_union correct AND f_num correct, excludes potentially resolvable)
MAX_WRONG_MATCH_RATE = 0.0400


def _max_rate(baseline):
    """Compute threshold rate: baseline * (1 + TOLERANCE), with an absolute floor."""
    return max(baseline * (1 + TOLERANCE), baseline + ABSOLUTE_FLOOR)


# --- Fixtures ---


@pytest.fixture(scope="module")
def eval_data():
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
        """Union texts misclassified as non-union (only examples with known f_num)."""
        union, _ = eval_data
        union_results, _ = predictions
        n_with_fnum = sum(1 for ex in union if ex["records"][0]["f_num"] != -100)
        fn = sum(
            1
            for ex, r in zip(union, union_results)
            if not r["is_union"] and ex["records"][0]["f_num"] != -100
        )
        rate = fn / n_with_fnum if n_with_fnum else 0
        threshold = _max_rate(MAX_FALSE_NEGATIVE_RATE)
        assert rate <= threshold, (
            f"False negative rate: {rate:.4f} ({fn}/{n_with_fnum}), "
            f"threshold: {threshold:.4f}"
        )

    def test_false_positives(self, eval_data, predictions):
        """Non-union texts misclassified as union."""
        _, non_union = eval_data
        _, non_union_results = predictions
        n = len(non_union)
        if n == 0:
            return
        fp = sum(1 for r in non_union_results if r["is_union"])
        rate = fp / n
        threshold = _max_rate(MAX_FALSE_POSITIVE_RATE)
        assert rate <= threshold, (
            f"False positive rate: {rate:.4f} ({fp}/{n}), "
            f"threshold: {threshold:.4f}"
        )


# --- Accuracy tests ---


@pytest.mark.eval
class TestAccuracy:

    def test_fnum_accuracy(self, eval_data, predictions):
        """f_num error rate on union examples with is_union=True and known f_num."""
        union, _ = eval_data
        union_results, _ = predictions

        errors = 0
        n = 0
        for ex, result in zip(union, union_results):
            if not result["is_union"]:
                continue
            true_fnum = ex["records"][0]["f_num"]
            if true_fnum == -100:
                continue
            n += 1
            if result["f_num"] != true_fnum:
                errors += 1

        rate = errors / n if n else 0
        threshold = _max_rate(MAX_FNUM_ERROR_RATE)
        assert rate <= threshold, (
            f"f_num error rate: {rate:.4f} ({errors}/{n}), "
            f"threshold: {threshold:.4f}"
        )

    def test_union_name_accuracy(self, eval_data, predictions):
        """union_name error rate on union examples with is_union=True."""
        union, _ = eval_data
        union_results, _ = predictions

        errors = 0
        n = 0
        for ex, result in zip(union, union_results):
            if not result["is_union"]:
                continue
            true_union = ex["records"][0].get("union_name", "")
            if not true_union:
                continue
            n += 1
            if result["union_name"] != true_union:
                errors += 1

        rate = errors / n if n else 0
        threshold = _max_rate(MAX_UNION_NAME_ERROR_RATE)
        assert rate <= threshold, (
            f"union_name error rate: {rate:.4f} ({errors}/{n}), "
            f"threshold: {threshold:.4f}"
        )


# --- End-to-end test ---


@pytest.mark.eval
class TestEndToEnd:

    def test_wrong_fnum_matches(self, eval_data, predictions):
        """Union texts where is_union=True but f_num is wrong.

        Excludes 'potentially resolvable' examples (unknown true f_num).
        """
        union, _ = eval_data
        union_results, _ = predictions

        wrong = 0
        n = 0
        for ex, result in zip(union, union_results):
            if ex.get("reason_missing_fnum") == "potentially resolvable":
                continue
            true_fnum = ex["records"][0]["f_num"]
            if true_fnum == -100:
                continue  # Skip no-fnum examples (match_found threshold TBD)
            n += 1
            if result["is_union"] and result["f_num"] != true_fnum:
                wrong += 1

        rate = wrong / n if n else 0
        threshold = _max_rate(MAX_WRONG_MATCH_RATE)
        assert rate <= threshold, (
            f"Wrong match rate: {rate:.4f} ({wrong}/{n}), "
            f"threshold: {threshold:.4f}"
        )
