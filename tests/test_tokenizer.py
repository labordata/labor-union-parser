"""Tests for tokenizer period handling."""

from labor_union_parser.char_cnn import CHAR_VOCAB, chars_to_ids, tokenize_to_chars


def get_tokens(text: str) -> list[str]:
    """Helper to get just the non-empty tokens."""
    _, tokens, _, _ = tokenize_to_chars(text, max_tokens=40)
    return [t for t in tokens if t]


def get_tokens_and_types(text: str) -> list[tuple[str, int]]:
    """Helper to get (token, token_type) pairs for non-empty tokens."""
    _, tokens, _, token_types = tokenize_to_chars(text, max_tokens=40)
    return [(t, tt) for t, tt in zip(tokens, token_types) if t]


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


class TestCharsToIds:
    """chars_to_ids converts tokens to character ID sequences."""

    def test_known_characters(self):
        ids = chars_to_ids("abc")
        assert ids[0] == CHAR_VOCAB["a"]
        assert ids[1] == CHAR_VOCAB["b"]
        assert ids[2] == CHAR_VOCAB["c"]

    def test_unknown_character_maps_to_unk(self):
        ids = chars_to_ids("@")
        assert ids[0] == CHAR_VOCAB["<UNK>"]

    def test_unicode_maps_to_unk(self):
        ids = chars_to_ids("ñ")
        assert ids[0] == CHAR_VOCAB["<UNK>"]

    def test_case_folded(self):
        assert chars_to_ids("ABC") == chars_to_ids("abc")

    def test_empty_string_all_padding(self):
        ids = chars_to_ids("")
        assert all(c == 0 for c in ids)

    def test_padding_length(self):
        ids = chars_to_ids("hi")
        assert len(ids) == 20  # MAX_CHARS_PER_TOKEN
        assert ids[2:] == [0] * 18

    def test_truncation_at_max_chars(self):
        long_token = "a" * 30
        ids = chars_to_ids(long_token, max_chars=20)
        assert len(ids) == 20
        assert all(c == CHAR_VOCAB["a"] for c in ids)

    def test_digits(self):
        ids = chars_to_ids("123")
        assert ids[0] == CHAR_VOCAB["1"]
        assert ids[1] == CHAR_VOCAB["2"]
        assert ids[2] == CHAR_VOCAB["3"]

    def test_punctuation(self):
        for char in "-/&,.":
            ids = chars_to_ids(char)
            assert ids[0] == CHAR_VOCAB[char]


class TestTokenTypes:
    """token_type values: 0=word, 1=number, 2=space, 3=punct, 4=pad."""

    def test_word_type(self):
        pairs = get_tokens_and_types("hello")
        assert pairs[0] == ("hello", 0)

    def test_number_type(self):
        pairs = get_tokens_and_types("42")
        assert pairs[0] == ("42", 1)

    def test_space_type(self):
        # Use multi-char words to avoid letter-merging regex
        pairs = get_tokens_and_types("alpha beta")
        assert pairs[1] == (" ", 2)

    def test_punct_type(self):
        pairs = get_tokens_and_types("a-b")
        assert pairs[1] == ("-", 3)

    def test_pad_type(self):
        _, _, _, token_types = tokenize_to_chars("hi", max_tokens=5)
        # After real tokens, remaining should be pad (4)
        assert token_types[-1] == 4

    def test_acronym_is_word_type(self):
        pairs = get_tokens_and_types("I.B.E.W.")
        assert pairs[0] == ("ibew", 0)


class TestNumberNormalization:
    """Numbers should have leading zeros stripped for pointer matching."""

    def test_leading_zeros_stripped(self):
        tokens = get_tokens("Local 007")
        assert "7" in tokens

    def test_all_zeros_become_zero(self):
        tokens = get_tokens("Local 000")
        assert "0" in tokens

    def test_plain_number_unchanged(self):
        tokens = get_tokens("Local 42")
        assert "42" in tokens


class TestLetterMergingEdgeCases:
    """Letter-merging regex should not merge single letters adjacent to words."""

    def test_single_letter_before_word_no_merge(self):
        tokens = get_tokens("A big union")
        assert "a" in tokens
        assert "big" in tokens
        assert "union" in tokens

    def test_single_letter_article(self):
        tokens = get_tokens("I am here")
        assert tokens == ["i", " ", "am", " ", "here"]

    def test_tab_separated_letters_merge(self):
        tokens = get_tokens("A\tF\tG\tE")
        assert "afge" in tokens

    def test_spaced_letters_near_words(self):
        # "Local A F G E 123" - the letters should merge
        tokens = get_tokens("Local A F G E 123")
        assert "afge" in tokens
        assert "local" in tokens
        assert "123" in tokens


class TestTrailingPeriod:
    """Trailing periods (not followed by anything) are dropped."""

    def test_trailing_period_dropped(self):
        tokens = get_tokens("IBEW.")
        assert "." not in tokens
        assert "ibew" in tokens

    def test_period_before_space_dropped(self):
        tokens = get_tokens("Inc. of")
        assert "." not in tokens


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
