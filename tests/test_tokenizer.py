"""Tests for tokenizer period handling."""

from labor_union_parser.char_cnn import tokenize_to_chars


def get_tokens(text: str) -> list[str]:
    """Helper to get just the non-empty tokens."""
    _, tokens, _, _ = tokenize_to_chars(text, max_tokens=40)
    return [t for t in tokens if t]


class TestAcronymHandling:
    """Acronyms with periods should be merged into single tokens."""

    def test_ibew_with_periods(self):
        assert get_tokens("I.B.E.W.") == ["ibew"]

    def test_ibew_matches_no_periods(self):
        assert get_tokens("I.B.E.W.") == get_tokens("IBEW")

    def test_iam_with_periods(self):
        assert get_tokens("I.A.M.") == ["iam"]

    def test_iatse_with_periods(self):
        assert get_tokens("I.A.T.S.E.") == ["iatse"]

    def test_iuoe_with_periods(self):
        assert get_tokens("I.U.O.E.") == ["iuoe"]

    def test_iupat_with_periods(self):
        assert get_tokens("I.U.P.A.T.") == ["iupat"]

    def test_acronym_in_context(self):
        tokens = get_tokens("I.B.E.W. Local 134")
        assert tokens == ["ibew", " ", "local", " ", "134"]

    def test_two_letter_acronym(self):
        # U.S. should merge
        assert get_tokens("U.S. Steel") == ["us", " ", "steel"]


class TestSpacedAcronyms:
    """Acronyms with spaces between letters should be merged."""

    def test_afge_spaced(self):
        assert get_tokens("A F G E - 2138") == ["afge", " ", "-", " ", "2138"]

    def test_atu_spaced(self):
        assert get_tokens("A T U - 425") == ["atu", " ", "-", " ", "425"]

    def test_spaced_matches_no_spaces(self):
        assert get_tokens("A F G E") == get_tokens("AFGE")

    def test_spaced_matches_periods(self):
        assert get_tokens("I B E W") == get_tokens("I.B.E.W.")

    def test_spaced_in_context(self):
        tokens = get_tokens("Local A F G E 123")
        assert tokens == ["local", " ", "afge", " ", "123"]

    def test_two_letter_spaced(self):
        assert get_tokens("A W - 100") == ["aw", " ", "-", " ", "100"]

    def test_not_single_letter_word(self):
        # Single letter followed by a word should not merge
        tokens = get_tokens("I am here")
        assert tokens == ["i", " ", "am", " ", "here"]


class TestAbbreviations:
    """Trailing periods after abbreviations (followed by space) should be dropped."""

    def test_no_abbreviation_with_space(self):
        # "No. 123" - period followed by space, drop it
        tokens = get_tokens("Local No. 123")
        assert tokens == ["local", " ", "no", " ", "123"]

    def test_no_abbreviation_no_space(self):
        # "No.123" - period followed by number, keep it as signal
        tokens = get_tokens("Local No.123")
        assert tokens == ["local", " ", "no", ".", "123"]

    def test_intl_abbreviation(self):
        tokens = get_tokens("Intl. Brotherhood")
        assert tokens == ["intl", " ", "brotherhood"]

    def test_int_abbreviation(self):
        tokens = get_tokens("Int. Union")
        assert tokens == ["int", " ", "union"]


class TestDecimalNumbers:
    """Decimal local numbers keep the period between digits."""

    def test_decimal_local_number(self):
        tokens = get_tokens("AFSCME 140.11")
        assert tokens == ["afscme", " ", "140", ".", "11"]

    def test_decimal_in_context(self):
        tokens = get_tokens("Council 25 1640.12")
        assert tokens == ["council", " ", "25", " ", "1640", ".", "12"]


class TestOtherPunctuation:
    """Other punctuation should be preserved."""

    def test_ampersand(self):
        tokens = get_tokens("I.A.M. & A.W.")
        # Both acronyms should merge, ampersand preserved
        assert "iam" in tokens
        assert "&" in tokens
        assert "aw" in tokens

    def test_hyphen(self):
        tokens = get_tokens("IBEW-622")
        assert tokens == ["ibew", "-", "622"]

    def test_parentheses(self):
        tokens = get_tokens("SEIU (Local 1199)")
        assert "(" in tokens
        assert ")" in tokens
        assert "seiu" in tokens
        assert "1199" in tokens


class TestSpaceHandling:
    """Spaces should be normalized to single spaces."""

    def test_multiple_spaces(self):
        tokens = get_tokens("IBEW    Local   123")
        assert tokens == ["ibew", " ", "local", " ", "123"]

    def test_leading_trailing_spaces(self):
        tokens = get_tokens("  IBEW  ")
        assert tokens == [" ", "ibew", " "]


if __name__ == "__main__":
    # Run tests manually
    import traceback

    test_classes = [
        TestAcronymHandling,
        TestSpacedAcronyms,
        TestAbbreviations,
        TestDecimalNumbers,
        TestOtherPunctuation,
        TestSpaceHandling,
    ]

    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                try:
                    getattr(instance, name)()
                    print(f"  PASS: {cls.__name__}.{name}")
                    passed += 1
                except AssertionError:
                    print(f"  FAIL: {cls.__name__}.{name}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{passed} passed, {failed} failed")
