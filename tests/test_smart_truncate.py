"""Tests for smart_truncate_nonspace number recovery logic."""

from labor_union_parser.tokenizer import smart_truncate_nonspace


def _tokens(result):
    """Extract non-empty token strings from smart_truncate output."""
    return [d["token"] for d in result if d["token"]]


class TestBasicTruncation:
    """Basic tokenization and space dropping."""

    def test_short_input_no_truncation(self):
        result = smart_truncate_nonspace("IBEW Local 134", max_tokens=20)
        tokens = _tokens(result)
        assert tokens == ["ibew", "local", "134"]

    def test_spaces_dropped(self):
        result = smart_truncate_nonspace("IBEW Local 134", max_tokens=20)
        assert all(d["token"] != " " for d in result)

    def test_padded_to_max_tokens(self):
        result = smart_truncate_nonspace("IBEW", max_tokens=20)
        assert len(result) == 20
        assert result[-1]["token"] == ""
        assert result[-1]["token_type"] == 4

    def test_empty_input(self):
        result = smart_truncate_nonspace("", max_tokens=5)
        assert len(result) == 5
        assert all(d["token"] == "" for d in result)


class TestNumberRecovery:
    """Lost numbers beyond max_tokens should replace trailing non-number tokens."""

    def test_lost_number_recovered(self):
        # 5 words + number at end, truncate to 4 non-space tokens
        text = "alpha beta gamma delta 42"
        result = smart_truncate_nonspace(text, max_tokens=4)
        tokens = _tokens(result)
        assert "42" in tokens

    def test_lost_number_replaces_last_word(self):
        # With max_tokens=4, "alpha beta gamma delta 42" has 5 non-space tokens.
        # Token 4 ("delta") is beyond limit, token 5 ("42") is a lost number.
        # "42" should replace the last non-number token in the truncated list.
        text = "alpha beta gamma delta 42"
        result = smart_truncate_nonspace(text, max_tokens=4)
        tokens = _tokens(result)
        # "delta" should be replaced by "42"
        assert tokens == ["alpha", "beta", "gamma", "42"]

    def test_multiple_lost_numbers_order_preserved(self):
        # Two numbers beyond the truncation point
        text = "a b c 100 200"
        result = smart_truncate_nonspace(text, max_tokens=3)
        tokens = _tokens(result)
        # "a" is replaced by 100, "b" by 200 (backward scan, reversed)
        assert "100" in tokens
        assert "200" in tokens
        assert tokens.index("100") < tokens.index("200")

    def test_duplicate_number_not_recovered(self):
        # "42" already in truncated set, second "42" is not "lost"
        text = "42 alpha beta gamma 42"
        result = smart_truncate_nonspace(text, max_tokens=4)
        tokens = _tokens(result)
        # First 4 non-space: "42", "alpha", "beta", "gamma"
        # Second "42" has same token string, not counted as lost
        assert tokens == ["42", "alpha", "beta", "gamma"]

    def test_no_numbers_no_recovery(self):
        text = "alpha beta gamma delta epsilon"
        result = smart_truncate_nonspace(text, max_tokens=3)
        tokens = _tokens(result)
        assert tokens == ["alpha", "beta", "gamma"]

    def test_number_within_limit_not_recovered(self):
        text = "alpha 42 beta"
        result = smart_truncate_nonspace(text, max_tokens=5)
        tokens = _tokens(result)
        # All fit, nothing to recover
        assert tokens == ["alpha", "42", "beta"]

    def test_all_numbers_no_replacement_targets(self):
        # If truncated list is all numbers, can't replace any
        text = "1 2 3 4 5"
        result = smart_truncate_nonspace(text, max_tokens=3)
        tokens = _tokens(result)
        # First 3 numbers kept, "4" and "5" lost but no non-number to replace
        assert tokens == ["1", "2", "3"]

    def test_punctuation_can_be_replaced(self):
        # Punctuation tokens are non-number, so they can be replaced
        text = "alpha - beta - 42"
        result = smart_truncate_nonspace(text, max_tokens=4)
        tokens = _tokens(result)
        # First 4 non-space: "alpha", "-", "beta", "-"
        # "42" is lost, should replace last non-number (the second "-")
        assert tokens == ["alpha", "-", "beta", "42"]
