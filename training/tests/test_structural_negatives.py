"""Tests for add_structural_negatives in prepare_data.py."""

import sys
from pathlib import Path

# Allow importing prepare_data from the training directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_data import add_example_categories, add_structural_negatives


def make_record(union_name, desig_name, desig_num, prefix, suffix, unit_id, f_num):
    return {
        "union_name": union_name,
        "desig_name": desig_name,
        "desig_num": desig_num,
        "prefix": prefix,
        "suffix": suffix,
        "unit_id": unit_id,
        "f_num": f_num,
    }


def test_same_conflict_group_different_prefix():
    """USW Local 555 case: same conflict group but different prefixes.

    f_nums 9935, 68176, 68853, 516405 all share (STEELWORKERS, 555)
    and are in the same conflict group (149), but have different prefixes.
    The prefix IS in the query, so the model can disambiguate — these
    should be structural negatives.
    """
    rec_9935 = make_record("STEELWORKERS, AFL-CIO", "LU", 555, 6, "", "A", 9935)
    rec_68176 = make_record("STEELWORKERS, AFL-CIO", "LU", 555, 10, "T", "B", 68176)
    rec_68853 = make_record("STEELWORKERS, AFL-CIO", "LU", 555, 13, "", "C", 68853)
    rec_516405 = make_record("STEELWORKERS, AFL-CIO", "LU", 555, 12, "", "D", 516405)

    fnum_to_records = {
        9935: [rec_9935],
        68176: [rec_68176],
        68853: [rec_68853],
        516405: [rec_516405],
    }

    # All in the same conflict group
    fnum_to_component = {9935: 149, 68176: 149, 68853: 149, 516405: 149}

    examples = [
        {
            "query": "United Steelworkers 13-555-01",
            "f_num": 68853,
            "records": [rec_68853],
        }
    ]

    add_structural_negatives(examples, fnum_to_records, fnum_to_component)

    neg_fnums = {s["f_num"] for s in examples[0].get("structural_negatives", [])}
    assert neg_fnums == {
        9935,
        68176,
        516405,
    }, f"Expected structural negatives from f_nums {{9935, 68176, 516405}}, got {neg_fnums}"


def test_structural_negatives_key_off_query_desig_num():
    """USW 2002 case: f_num 7040 has desig_num=61 AND desig_num=2002.

    The query "United Steelworkers 2002-22" is about desig_num=2002.
    Structural negatives should match on desig_num=2002 (from the query),
    not desig_num=61 (another variant of the same f_num).

    Add a second f_num (99999) with desig_num=2002 and different prefix
    to verify it gets picked up as a structural negative.
    """
    rec_7040_61 = make_record("STEELWORKERS, AFL-CIO", "LU", 61, 11, "U", "", 7040)
    rec_7040_2002 = make_record("STEELWORKERS, AFL-CIO", "LU", 2002, 11, "", "", 7040)
    rec_99999 = make_record("STEELWORKERS, AFL-CIO", "LU", 2002, 5, "", "", 99999)

    fnum_to_records = {
        7040: [rec_7040_61, rec_7040_2002],
        99999: [rec_99999],
    }

    fnum_to_component = {}

    examples = [
        {
            "query": "United Steelworkers 2002-22",
            "f_num": 7040,
            # records filtered to match query — only desig_num=2002
            "records": [rec_7040_2002],
        }
    ]

    add_structural_negatives(examples, fnum_to_records, fnum_to_component)

    neg_fnums = {s["f_num"] for s in examples[0].get("structural_negatives", [])}
    assert neg_fnums == {
        99999
    }, f"Expected structural negative f_num 99999 (shares desig_num=2002), got {neg_fnums}"


def test_category_is_query_level_not_fnum_level():
    """USW 2002 case: f_num 7040 has desig_num=61 (shared) and desig_num=2002 (unique).

    The query "United Steelworkers 2002-22" is about desig_num=2002, which
    is unique. It should be categorized as "unique", not "ambiguous".

    Currently classify_fnum_discrimination marks the entire f_num as ambiguous
    because desig_num=61 is shared with other f_nums.
    """
    rec_7040_61 = make_record("STEELWORKERS, AFL-CIO", "LU", 61, 11, "U", "", 7040)
    rec_7040_2002 = make_record("STEELWORKERS, AFL-CIO", "LU", 2002, 11, "", "", 7040)
    rec_8888 = make_record("STEELWORKERS, AFL-CIO", "LU", 61, 0, "", "", 8888)

    fnum_to_records = {
        7040: [rec_7040_61, rec_7040_2002],
        8888: [rec_8888],
    }

    examples = [
        {
            "query": "United Steelworkers 2002-22",
            "f_num": 7040,
            # records filtered to match query — only desig_num=2002
            "records": [rec_7040_2002],
            "split": "train",
        }
    ]

    add_example_categories(examples, fnum_to_records)

    assert examples[0]["category"] == "unique", (
        f"Query about desig_num=2002 (unique) should be 'unique', "
        f"got '{examples[0]['category']}'"
    )


def test_same_shape_different_unit_id_are_structural_negatives():
    """Iron Workers Local 46 case: multiple f_nums share the same
    (union_name, desig_num, desig_name, prefix, suffix) but differ by unit_id.

    The unit_id is a field the model sees in the record embedding, so
    these are distinguishable and should be structural negatives.
    """
    rec_8438 = make_record("IRON WORKERS AFL-CIO", "LU", 46, 0, "", "A", 8438)
    rec_9000 = make_record("IRON WORKERS AFL-CIO", "LU", 46, 0, "", "B", 9000)
    rec_9001 = make_record("IRON WORKERS AFL-CIO", "LU", 46, 0, "", "C", 9001)

    fnum_to_records = {
        8438: [rec_8438],
        9000: [rec_9000],
        9001: [rec_9001],
    }

    fnum_to_component = {}

    examples = [
        {
            "query": "Metallic Lathers Union and Reinforcing Ironworkers of NY 46",
            "f_num": 8438,
            "records": [rec_8438],
        }
    ]

    add_structural_negatives(examples, fnum_to_records, fnum_to_component)

    neg_fnums = {s["f_num"] for s in examples[0].get("structural_negatives", [])}
    assert neg_fnums == {9000, 9001}, (
        f"Expected structural negatives from f_nums {{9000, 9001}} "
        f"(same shape, different unit_id), got {neg_fnums}"
    )
