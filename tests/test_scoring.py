"""Tests for scoring module: normalization, pointer lookup, and gazetteer matrix."""

from labor_union_parser.scoring import (
    _get_field_value,
    _normalize_pointer_value,
    build_gazetteer_matrix,
    build_pointer_lookup,
)


class TestNormalizePointerValue:
    """_normalize_pointer_value normalizes record values for token matching."""

    def test_plain_number(self):
        assert _normalize_pointer_value(123) == "123"

    def test_string_number(self):
        assert _normalize_pointer_value("42") == "42"

    def test_leading_zeros_stripped(self):
        assert _normalize_pointer_value("007") == "7"

    def test_all_zeros_is_none(self):
        # "000" is the same as "0" — means absent
        assert _normalize_pointer_value("000") is None

    def test_zero_int_is_none(self):
        assert _normalize_pointer_value(0) is None

    def test_zero_string_is_none(self):
        assert _normalize_pointer_value("0") is None

    def test_empty_string_is_none(self):
        assert _normalize_pointer_value("") is None

    def test_whitespace_only_is_none(self):
        assert _normalize_pointer_value("  ") is None

    def test_none_is_none(self):
        assert _normalize_pointer_value(None) is None

    def test_uppercase_lowered(self):
        assert _normalize_pointer_value("A") == "a"

    def test_mixed_case_lowered(self):
        assert _normalize_pointer_value("NorthEast") == "northeast"

    def test_alphanumeric_lowered(self):
        # Not pure digits, so .lower() path is taken (no zero-stripping)
        assert _normalize_pointer_value("12A") == "12a"

    def test_whitespace_stripped(self):
        assert _normalize_pointer_value("  42  ") == "42"

    def test_whitespace_stripped_string(self):
        assert _normalize_pointer_value("  abc  ") == "abc"


class TestGetFieldValue:
    """_get_field_value extracts field values with correct defaults."""

    def test_fnum_present(self):
        assert _get_field_value({"f_num": 500000}, "f_num") == 500000

    def test_fnum_missing_raises(self):
        try:
            _get_field_value({}, "f_num")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_desig_num_present(self):
        assert _get_field_value({"desig_num": 42}, "desig_num") == 42

    def test_desig_num_missing_defaults_zero(self):
        assert _get_field_value({}, "desig_num") == 0

    def test_union_name_present(self):
        assert _get_field_value({"union_name": "SEIU"}, "union_name") == "SEIU"

    def test_union_name_missing_defaults_empty(self):
        assert _get_field_value({}, "union_name") == ""

    def test_suffix_missing_defaults_empty(self):
        assert _get_field_value({}, "suffix") == ""


class TestBuildPointerLookup:
    """build_pointer_lookup maps normalized values to record indices."""

    def test_basic_mapping(self):
        records = [
            {"desig_num": 100},
            {"desig_num": 200},
            {"desig_num": 100},
        ]
        val_to_idx, none_idx = build_pointer_lookup(records, "desig_num")
        assert set(val_to_idx.keys()) == {"100", "200"}
        assert val_to_idx["100"].tolist() == [0, 2]
        assert val_to_idx["200"].tolist() == [1]
        assert len(none_idx) == 0

    def test_zero_goes_to_none(self):
        records = [
            {"desig_num": 0},
            {"desig_num": 42},
        ]
        val_to_idx, none_idx = build_pointer_lookup(records, "desig_num")
        assert none_idx.tolist() == [0]
        assert val_to_idx["42"].tolist() == [1]

    def test_missing_field_goes_to_none(self):
        records = [
            {},  # desig_num defaults to 0 via _get_field_value
            {"desig_num": 7},
        ]
        val_to_idx, none_idx = build_pointer_lookup(records, "desig_num")
        assert none_idx.tolist() == [0]

    def test_leading_zeros_normalized(self):
        records = [
            {"desig_num": "007"},
            {"desig_num": 7},
        ]
        val_to_idx, none_idx = build_pointer_lookup(records, "desig_num")
        # Both should map to "7"
        assert "7" in val_to_idx
        assert val_to_idx["7"].tolist() == [0, 1]
        assert "007" not in val_to_idx

    def test_suffix_case_normalized(self):
        records = [
            {"suffix": "A"},
            {"suffix": "a"},
            {"suffix": "B"},
        ]
        val_to_idx, none_idx = build_pointer_lookup(records, "suffix")
        assert val_to_idx["a"].tolist() == [0, 1]
        assert val_to_idx["b"].tolist() == [2]

    def test_empty_suffix_goes_to_none(self):
        records = [
            {"suffix": ""},
            {"suffix": "A"},
        ]
        val_to_idx, none_idx = build_pointer_lookup(records, "suffix")
        assert none_idx.tolist() == [0]

    def test_empty_records(self):
        val_to_idx, none_idx = build_pointer_lookup([], "desig_num")
        assert val_to_idx == {}
        assert len(none_idx) == 0


class TestBuildGazetteerMatrix:
    """build_gazetteer_matrix builds field index tensors for classification fields."""

    def _make_vocabs(self):
        return {
            "union_name": {"SEIU": 0, "IBEW": 1, "AFSCME": 2},
            "desig_name": {"LU": 0, "JC": 1},
            "f_num": {500000: 0, 500001: 1},
            # Pointer fields need entries but are skipped by build_gazetteer_matrix
            "desig_num": {},
            "prefix": {},
            "suffix": {},
        }

    def test_known_values_get_correct_indices(self):
        fnum_to_records = {
            "500000": [
                {"union_name": "SEIU", "desig_name": "LU", "f_num": 500000},
            ],
        }
        field_indices, field_known, record_fnums, records = build_gazetteer_matrix(
            fnum_to_records, self._make_vocabs()
        )
        assert field_indices["union_name"].tolist() == [0]  # SEIU -> 0
        assert field_indices["desig_name"].tolist() == [0]  # LU -> 0
        assert field_indices["f_num"].tolist() == [0]  # 500000 -> 0
        assert field_known["union_name"].tolist() == [True]
        assert field_known["desig_name"].tolist() == [True]
        assert field_known["f_num"].tolist() == [True]

    def test_unknown_value_gets_index_zero_and_known_false(self):
        fnum_to_records = {
            "999999": [
                {"union_name": "UAW", "desig_name": "LU", "f_num": 999999},
            ],
        }
        field_indices, field_known, _, _ = build_gazetteer_matrix(
            fnum_to_records, self._make_vocabs()
        )
        # UAW not in vocab -> index 0, known=False
        assert field_indices["union_name"].tolist() == [0]
        assert field_known["union_name"].tolist() == [False]
        # LU is in vocab -> known=True
        assert field_known["desig_name"].tolist() == [True]

    def test_known_false_distinguishes_from_index_zero(self):
        """Index 0 is used for both 'SEIU' (known) and unknown values.
        field_known must distinguish them."""
        fnum_to_records = {
            "500000": [
                {"union_name": "SEIU", "desig_name": "LU", "f_num": 500000},
            ],
            "999999": [
                {"union_name": "UAW", "desig_name": "LU", "f_num": 999999},
            ],
        }
        field_indices, field_known, _, _ = build_gazetteer_matrix(
            fnum_to_records, self._make_vocabs()
        )
        # Both get index 0 for union_name
        assert field_indices["union_name"].tolist() == [0, 0]
        # But only first is truly known
        assert field_known["union_name"].tolist() == [True, False]

    def test_pointer_fields_skipped(self):
        fnum_to_records = {
            "500000": [
                {"union_name": "SEIU", "desig_name": "LU", "f_num": 500000},
            ],
        }
        field_indices, field_known, _, _ = build_gazetteer_matrix(
            fnum_to_records, self._make_vocabs()
        )
        assert "desig_num" not in field_indices
        assert "prefix" not in field_indices
        assert "suffix" not in field_indices

    def test_record_fnums_are_ints(self):
        fnum_to_records = {
            "500000": [
                {"union_name": "SEIU", "desig_name": "LU", "f_num": 500000},
            ],
        }
        _, _, record_fnums, _ = build_gazetteer_matrix(
            fnum_to_records, self._make_vocabs()
        )
        assert record_fnums == [500000]

    def test_multiple_records_per_fnum(self):
        fnum_to_records = {
            "500000": [
                {"union_name": "SEIU", "desig_name": "LU", "f_num": 500000},
                {"union_name": "SEIU", "desig_name": "JC", "f_num": 500000},
            ],
        }
        field_indices, _, record_fnums, records = build_gazetteer_matrix(
            fnum_to_records, self._make_vocabs()
        )
        assert len(records) == 2
        assert record_fnums == [500000, 500000]
        assert field_indices["desig_name"].tolist() == [0, 1]  # LU=0, JC=1

    def test_empty_gazetteer(self):
        field_indices, field_known, record_fnums, records = build_gazetteer_matrix(
            {}, self._make_vocabs()
        )
        assert records == []
        assert record_fnums == []
        for f in ("union_name", "desig_name", "f_num"):
            assert len(field_indices[f]) == 0
            assert len(field_known[f]) == 0
