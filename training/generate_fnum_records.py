#!/usr/bin/env python3
"""
Generate fnum_to_records.json: for each f_num, collect all distinct
record representations from opdr.db and attach unit identifiers.

Inputs:
    opdr.db                          — OLMS filing database
    training/fnum_to_unit_identifier.csv — unit_id per f_num

Output:
    training/data/fnum_to_records.json
"""

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "opdr.db"
UNIT_ID_CSV = Path(__file__).parent / "fnum_to_unit_identifier.csv"
OUTPUT_PATH = Path(__file__).parent / "data" / "fnum_to_records.json"


def load_unit_identifiers():
    """Load f_num -> unit_identifier from CSV."""
    fnum_to_uid = {}
    with open(UNIT_ID_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fnum_to_uid[int(row["f_num"])] = row["unit_identifier"]
    print(f"Loaded {len(fnum_to_uid)} unit identifiers")
    return fnum_to_uid


DESIG_NAME_COLLAPSE = {
    "LG": "LU",
    "LLG": "LU",
    "Br": "BR",
    "DLG": "DC",
}


def normalize_designation(s):
    """Normalize designation suffix strings — strip leading zeros from numeric values."""
    if not s:
        return ""
    s = s.strip()
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


def generate_records():
    """Query opdr.db for all distinct representations per f_num."""
    fnum_to_uid = load_unit_identifiers()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT
            f_num,
            union_name,
            COALESCE(desig_name, '') as desig_name,
            desig_num,
            COALESCE(desiq_pre, '') as desig_pre,
            COALESCE(desig_suf, '') as desig_suf
        FROM lm_data
        WHERE f_num IS NOT NULL
          AND union_name IS NOT NULL
        """
    )

    fnum_to_records = defaultdict(list)
    seen = set()

    for row in cur.fetchall():
        f_num, union_name, desig_name, desig_num, prefix_raw, suffix_raw = row

        # Collapse desig_name variants
        desig_name = DESIG_NAME_COLLAPSE.get(desig_name, desig_name)

        # Normalize prefix: strip non-alphanumeric, keep if numeric
        prefix_clean = "".join(c for c in (prefix_raw or "") if c.isalnum())
        if prefix_clean.isdigit() and prefix_clean:
            prefix = int(prefix_clean)
        else:
            prefix = 0

        suffix = normalize_designation(suffix_raw)

        # Deduplicate records per f_num — keep NULL vs 0 desig_num distinct
        key = (f_num, union_name, desig_name, desig_num, prefix, suffix)
        desig_num = int(desig_num) if desig_num is not None else 0
        if key in seen:
            continue
        seen.add(key)

        unit_id = fnum_to_uid.get(f_num, "")

        record = {
            "union_name": union_name,
            "desig_name": desig_name,
            "desig_num": desig_num,
            "prefix": prefix,
            "suffix": suffix,
            "unit_id": unit_id,
            "f_num": f_num,
        }
        fnum_to_records[f_num].append(record)

    conn.close()

    total_records = sum(len(recs) for recs in fnum_to_records.values())
    print(f"{len(fnum_to_records)} unique f_nums, {total_records} records")

    # Distribution of records per f_num
    counts = defaultdict(int)
    for recs in fnum_to_records.values():
        counts[len(recs)] += 1
    print("\nRecords per f_num:")
    for n in sorted(counts.keys())[:10]:
        print(f"  {n}: {counts[n]} f_nums")
    if len(counts) > 10:
        print(f"  ... and {len(counts) - 10} more")

    return fnum_to_records


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fnum_to_records = generate_records()

    # JSON keys must be strings
    output = {str(k): v for k, v in fnum_to_records.items()}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
