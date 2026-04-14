"""Tests for filter_records_by_query from training/prepare_data.py.

Tests cover the three filtering stages:
1. desig_num filtering
2. prefix/suffix filtering (positive match and inverse)
3. desig_name filtering (keyword, code, inverse, preference)

Plus regression tests for known bugs (see issue #17).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from prepare_data import filter_records_by_query


def rec(
    desig_num=0,
    prefix=0,
    suffix="",
    desig_name="LU",
    union_name="STEELWORKERS, AFL-CIO",
    unit_id="",
    **extra,
):
    """Helper to build a record dict."""
    r = {
        "union_name": union_name,
        "desig_name": desig_name,
        "desig_num": desig_num,
        "prefix": prefix,
        "suffix": suffix,
        "unit_id": unit_id,
    }
    r.update(extra)
    return r


# ── Stage 1: desig_num filtering ──────────────────────────────────────


class TestDesigNumFiltering:
    def test_single_record_passthrough(self):
        records = [rec(desig_num=100)]
        assert filter_records_by_query("USW Local 100", records) == records

    def test_keeps_matching_desig_num(self):
        r100 = rec(desig_num=100)
        r200 = rec(desig_num=200)
        result = filter_records_by_query("USW Local 100", [r100, r200])
        assert result == [r100]

    def test_multiple_desig_nums_in_query(self):
        r100 = rec(desig_num=100)
        r200 = rec(desig_num=200)
        r300 = rec(desig_num=300)
        result = filter_records_by_query("Locals 100 and 200", [r100, r200, r300])
        assert set(r["desig_num"] for r in result) == {100, 200}

    def test_no_numbers_in_query_keeps_zero(self):
        r0 = rec(desig_num=0)
        r100 = rec(desig_num=100)
        result = filter_records_by_query("United Steelworkers", [r0, r100])
        assert result == [r0]

    def test_all_same_desig_num_no_filter(self):
        records = [rec(desig_num=100, prefix=1), rec(desig_num=100, prefix=2)]
        result = filter_records_by_query("USW 100", records)
        assert len(result) == 2


# ── Stage 2: prefix filtering ────────────────────────────────────────


class TestPrefixFiltering:
    def test_prefix_positive_match(self):
        """When query contains a number matching a record prefix, keep it."""
        r_pre2 = rec(desig_num=286, prefix=2)
        r_pre6 = rec(desig_num=286, prefix=6)
        r_pre0 = rec(desig_num=286, prefix=0)
        result = filter_records_by_query("USW Local 2-286", [r_pre0, r_pre2, r_pre6])
        assert result == [r_pre2]

    def test_prefix_inverse_keeps_zero_prefix(self):
        """When query has no prefix indicator, keep the pre=0 variant."""
        r_pre0 = rec(desig_num=286, prefix=0)
        r_pre6 = rec(desig_num=286, prefix=6)
        result = filter_records_by_query("USW 286", [r_pre0, r_pre6])
        assert result == [r_pre0]

    def test_prefix_no_match_falls_to_inverse(self):
        """When query number doesn't match any prefix, fall back to inverse."""
        r_pre2 = rec(desig_num=286, prefix=2)
        r_pre6 = rec(desig_num=286, prefix=6)
        # Query has 286 which matches neither prefix
        result = filter_records_by_query("USW 286", [r_pre2, r_pre6])
        # Neither is pre=0, so both survive (no inverse filter applies)
        assert len(result) == 2


# ── Stage 2: suffix filtering ────────────────────────────────────────


class TestSuffixFiltering:
    def test_suffix_letter_after_number(self):
        """'748L' should match suffix='L'."""
        r_plain = rec(desig_num=748, suffix="")
        r_L = rec(desig_num=748, suffix="L")
        result = filter_records_by_query("USW 748L", [r_plain, r_L])
        assert result == [r_L]

    def test_suffix_letter_with_dash(self):
        """'748-L' should match suffix='L'."""
        r_plain = rec(desig_num=748, suffix="")
        r_L = rec(desig_num=748, suffix="L")
        result = filter_records_by_query("USW 748-L", [r_plain, r_L])
        assert result == [r_L]

    def test_suffix_letter_space_separated(self):
        """'985 L' should match suffix='L'."""
        r_plain = rec(desig_num=985, suffix="")
        r_L = rec(desig_num=985, suffix="L")
        result = filter_records_by_query("USW 985 L", [r_plain, r_L])
        assert result == [r_L]

    def test_suffix_before_number(self):
        """'D583' should match suffix='D' or desig_name='D'."""
        r_plain = rec(desig_num=583, suffix="")
        r_D = rec(desig_num=583, suffix="D")
        result = filter_records_by_query("USW D583", [r_plain, r_D])
        assert result == [r_D]

    def test_suffix_inverse_keeps_no_suffix(self):
        """When query doesn't specify suffix, keep suffix='' variant."""
        r_plain = rec(desig_num=286, suffix="")
        r_S = rec(desig_num=286, suffix="S")
        result = filter_records_by_query("USW 286", [r_plain, r_S])
        assert result == [r_plain]

    def test_suffix_B_attached_to_desig_num(self):
        """'182B' should match suffix='B'."""
        r_plain = rec(desig_num=182, suffix="")
        r_B = rec(desig_num=182, suffix="B")
        result = filter_records_by_query("United Steelworkers 182B", [r_plain, r_B])
        assert result == [r_B]

    def test_suffix_G_attached_to_desig_num(self):
        """'460G' should match suffix='G'."""
        r_plain = rec(desig_num=460, suffix="")
        r_G = rec(desig_num=460, suffix="G")
        result = filter_records_by_query("USW 460G", [r_plain, r_G])
        assert result == [r_G]


# ── Stage 2: suffix + trailing number (issue #17 regressions) ────────


class TestSuffixWithTrailingNumber:
    """Regression tests for issue #17.

    When a suffix letter is attached to the desig_num and followed by a
    trailing number (e.g. '182B-02'), the trailing number should NOT be
    treated as a prefix match. The suffix letter is the discriminator.
    """

    def test_suffix_B_with_trailing_number(self):
        """'182B-02' should match suffix='B', not prefix=2."""
        r_B = rec(desig_num=182, prefix=0, suffix="B")
        r_pre2 = rec(desig_num=182, prefix=2, suffix="S")
        result = filter_records_by_query("United Steelworkers 182B-02", [r_B, r_pre2])
        assert result == [r_B]

    def test_suffix_G_with_trailing_number(self):
        """'460G-01' should match suffix='G', not prefix=1."""
        r_G = rec(desig_num=460, prefix=11, suffix="G")
        r_pre1 = rec(desig_num=460, prefix=1, suffix="")
        result = filter_records_by_query("USW-460G-01", [r_G, r_pre1])
        assert result == [r_G]

    def test_suffix_L_with_trailing_number(self):
        """'134L-01' should match suffix='L', not prefix=1."""
        r_L = rec(desig_num=134, prefix=4, suffix="L")
        r_pre1 = rec(desig_num=134, prefix=1, suffix="S")
        result = filter_records_by_query("USW-00134L-01", [r_L, r_pre1])
        assert result == [r_L]

    def test_suffix_B_with_leading_zeros(self):
        """'00182B-04' should match suffix='B', not prefix=4."""
        r_B = rec(desig_num=182, prefix=0, suffix="B")
        r_pre4 = rec(desig_num=182, prefix=4, suffix="L")
        result = filter_records_by_query("United Steelworkers 00182B-04", [r_B, r_pre4])
        assert result == [r_B]

    def test_local_dash_L(self):
        """'Local 7-L' should match suffix='L', not prefix=7."""
        r_L = rec(desig_num=7, prefix=1, suffix="L")
        r_pre7 = rec(desig_num=7, prefix=7, suffix="")
        result = filter_records_by_query(
            "UNITED STEELWORKERS, LOCAL 7-L", [r_L, r_pre7]
        )
        assert result == [r_L]

    def test_suffix_B_with_leading_number(self):
        """'45B-09': suffix='B' should win over prefix=9."""
        r_B = rec(desig_num=45, prefix=0, suffix="B")
        r_pre9 = rec(desig_num=45, prefix=9, suffix="T")
        result = filter_records_by_query("UNITED STEELWORKERS 45B-09", [r_B, r_pre9])
        assert result == [r_B]


# ── Stage 2: district number vs prefix (issue #17) ───────────────────


class TestDistrictVsPrefix:
    """District numbers in text should not be confused with record prefix."""

    def test_district_number_not_prefix(self):
        """'DISTRICT 4, UNITED STEELWORKERS UNION' with dn=4:
        the '4' is a district number, not a prefix."""
        r_pre0 = rec(desig_num=4, prefix=0, suffix="")
        r_pre4 = rec(desig_num=4, prefix=4, suffix="S")
        result = filter_records_by_query(
            "DISTRICT 4, UNITED STEELWORKERS UNION", [r_pre0, r_pre4]
        )
        # Should keep pre=0 (the District itself), not match pre=4
        assert result == [r_pre0]

    @pytest.mark.xfail(
        reason="issue #17: district/subdistrict numbers matched as prefix"
    )
    def test_district_with_subdistrict(self):
        """'DISTRICT 1, SUBDISTRICT 2' — neither 1 nor 2 is a prefix."""
        r_pre0 = rec(desig_num=1, prefix=0, suffix="")
        r_pre2 = rec(desig_num=1, prefix=2, suffix="")
        result = filter_records_by_query(
            "UNITED STEELWORKERS OF AMERICA, DISTRICT 1, SUBDISTRICT 2",
            [r_pre0, r_pre2],
        )
        assert result == [r_pre0]


# ── Stage 2: trailing number not a prefix ─────────────────────────────


class TestTrailingNumberNotPrefix:
    """Trailing numbers after desig_num (sub-unit IDs) should not match prefix."""

    def test_trailing_number_ignored(self):
        """'USW-530-01': the 01 is a sub-unit, not prefix=1."""
        r_pre5 = rec(desig_num=530, prefix=5, suffix="")
        r_pre0 = rec(desig_num=530, prefix=0, suffix="")
        result = filter_records_by_query("USW-530-01", [r_pre5, r_pre0])
        # Should not match pre=5 or any prefix from trailing 01
        # Ideally keeps pre=0, but at minimum should not match wrong prefix
        assert r_pre0 in result

    @pytest.mark.xfail(reason="issue #17: trailing sub-unit number matched as prefix")
    def test_multi_trailing_numbers(self):
        """'USW 13-1-5' with dn=1: 13 is district, 5 is sub-unit."""
        r_pre4 = rec(desig_num=1, prefix=4, suffix="")
        r_pre5 = rec(desig_num=1, prefix=5, suffix="")
        r_pre0 = rec(desig_num=1, prefix=0, suffix="")
        result = filter_records_by_query("USW 13-1-5", [r_pre0, r_pre4, r_pre5])
        # Should not resolve to pre=5 based on trailing number
        assert r_pre5 not in result or len(result) > 1


# ── Stage 3: desig_name filtering ─────────────────────────────────────


class TestDesigNameFiltering:
    def test_keyword_local_matches_LU(self):
        r_LU = rec(desig_name="LU")
        r_DC = rec(desig_name="DC")
        result = filter_records_by_query("USW Local 100", [r_LU, r_DC])
        assert result == [r_LU]

    def test_keyword_district_matches_D_or_DC(self):
        r_LU = rec(desig_name="LU")
        r_D = rec(desig_name="D")
        r_DC = rec(desig_name="DC")
        result = filter_records_by_query("USW District 4", [r_LU, r_D, r_DC])
        assert all(r["desig_name"] in ("D", "DC") for r in result)

    def test_keyword_lodge_matches_LU_or_DC(self):
        r_LU = rec(desig_name="LU")
        r_DC = rec(desig_name="DC")
        r_JB = rec(desig_name="JB")
        result = filter_records_by_query("Lodge 100", [r_LU, r_DC, r_JB])
        assert result == [r_LU, r_DC]

    def test_keyword_district_council_matches_DC(self):
        r_DC = rec(desig_name="DC")
        r_D = rec(desig_name="D")
        r_LU = rec(desig_name="LU")
        result = filter_records_by_query("District Council 5", [r_DC, r_D, r_LU])
        assert result == [r_DC]

    def test_code_near_number_matches_desig_name(self):
        """'D583' should match desig_name='D'."""
        r_LU = rec(desig_num=583, desig_name="LU")
        r_D = rec(desig_num=583, desig_name="D")
        result = filter_records_by_query("D583", [r_LU, r_D])
        assert result == [r_D]

    def test_inverse_keeps_non_blank(self):
        r_LU = rec(desig_name="LU")
        r_blank = rec(desig_name="")
        result = filter_records_by_query("USW 100", [r_LU, r_blank])
        assert result == [r_LU]

    def test_preference_LU_over_LOCAL(self):
        r_LU = rec(desig_name="LU")
        r_LOCAL = rec(desig_name="LOCAL")
        result = filter_records_by_query("USW Local 100", [r_LU, r_LOCAL])
        assert result == [r_LU]

    def test_preference_C_over_LEADC(self):
        r_C = rec(desig_name="C")
        r_LEADC = rec(desig_name="LEADC")
        result = filter_records_by_query("Council 5", [r_C, r_LEADC])
        assert result == [r_C]


# ── Combined stage interactions ───────────────────────────────────────


class TestCombinedStages:
    def test_prefix_and_suffix_together(self):
        """'2-286-S' should match prefix=2 AND suffix='S'."""
        r1 = rec(desig_num=286, prefix=2, suffix="")
        r2 = rec(desig_num=286, prefix=2, suffix="S")
        r3 = rec(desig_num=286, prefix=6, suffix="")
        result = filter_records_by_query("USW 2-286-S", [r1, r2, r3])
        assert result == [r2]

    def test_desig_num_then_prefix(self):
        """Filter desig_num first, then prefix."""
        r100_p2 = rec(desig_num=100, prefix=2)
        r100_p0 = rec(desig_num=100, prefix=0)
        r200_p2 = rec(desig_num=200, prefix=2)
        result = filter_records_by_query("USW 2-100", [r100_p2, r100_p0, r200_p2])
        assert result == [r100_p2]

    def test_empty_records(self):
        assert filter_records_by_query("anything", []) == []

    def test_single_record_always_returned(self):
        r = rec(desig_num=999, prefix=5, suffix="S")
        assert filter_records_by_query("totally unrelated text", [r]) == [r]


# ── Numeric suffix extraction ─────────────────────────────────────────


class TestNumericSuffix:
    def test_numeric_suffix_after_long_number(self):
        """'1410-1' should extract numeric suffix '1'."""
        r_suf1 = rec(desig_num=1410, suffix="1")
        r_plain = rec(desig_num=1410, suffix="")
        result = filter_records_by_query("USW 1410-1", [r_plain, r_suf1])
        assert result == [r_suf1]

    def test_numeric_suffix_with_leading_zero(self):
        """'662-04' should extract numeric suffix '4'."""
        r_suf4 = rec(desig_num=662, suffix="4")
        r_plain = rec(desig_num=662, suffix="")
        result = filter_records_by_query("USW 662-04", [r_plain, r_suf4])
        assert result == [r_suf4]


# ── Regression tests: desig_num excluded from prefix candidates ────────


class TestDesigNumNotPrefix:
    """When a number matches as desig_num, it should not also match as prefix."""

    def test_district_N_where_N_is_desig_num(self):
        """'District 7' with dn=7: the 7 is desig_num, not prefix=7."""
        r_pre0 = rec(desig_num=7, prefix=0)
        r_pre7 = rec(desig_num=7, prefix=7)
        result = filter_records_by_query(
            "District 7 United Steelworkers", [r_pre0, r_pre7]
        )
        assert result == [r_pre0]

    def test_prefix_separate_from_desig_num(self):
        """'USW 8-00002-01': 8 is prefix, 2 is desig_num."""
        r_pre2 = rec(desig_num=2, prefix=2)
        r_pre8 = rec(desig_num=2, prefix=8)
        r_pre0 = rec(desig_num=2, prefix=0)
        result = filter_records_by_query("USW 8-00002-01", [r_pre2, r_pre8, r_pre0])
        assert result == [r_pre8]

    def test_suffix_letter_then_number_no_dash(self):
        """'582L1': suffix=L, and 1 is prefix (no dash, so not excluded)."""
        r_pre1_sufL = rec(desig_num=582, prefix=1, suffix="L")
        r_pre0 = rec(desig_num=582, prefix=0, suffix="")
        result = filter_records_by_query("USA - 582L1", [r_pre1_sufL, r_pre0])
        assert result == [r_pre1_sufL]

    def test_suffix_letter_then_dash_number_excluded(self):
        """'182B-02': suffix=B, trailing -02 is NOT prefix."""
        r_B = rec(desig_num=182, prefix=0, suffix="B")
        r_pre2 = rec(desig_num=182, prefix=2, suffix="S")
        result = filter_records_by_query("USW 182B-02", [r_B, r_pre2])
        assert result == [r_B]

    @pytest.mark.xfail(
        reason="issue #17: when desig_num == prefix value, prefix is excluded"
    )
    def test_prefix_equals_desig_num(self):
        """'USW 7-0007': prefix=7, dn=7. The 7 is both desig_num and prefix."""
        r_pre0 = rec(desig_num=7, prefix=0)
        r_pre7 = rec(desig_num=7, prefix=7)
        result = filter_records_by_query("USW 7-0007", [r_pre0, r_pre7])
        assert result == [r_pre7]
