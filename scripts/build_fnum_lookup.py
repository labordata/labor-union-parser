#!/usr/bin/env python3
"""Build fnum_lookup.json from opdr.db database."""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "opdr.db"
OUTPUT_PATH = (
    PROJECT_ROOT / "src" / "labor_union_parser" / "weights" / "fnum_lookup.json"
)


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print(f"Reading from: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Query all unique (aff_abbr, desig_num, f_num) combinations
    cursor.execute(
        """
        SELECT DISTINCT aff_abbr, desig_num, f_num
        FROM lm_data
        WHERE aff_abbr IS NOT NULL
          AND aff_abbr != ''
          AND desig_num IS NOT NULL
          AND f_num IS NOT NULL
        ORDER BY aff_abbr, desig_num
    """
    )

    # Build lookup: {aff_abbr|desig_num: [fnum1, fnum2, ...]}
    lookup = defaultdict(list)
    for aff_abbr, desig_num, f_num in cursor:
        key = f"{aff_abbr}|{desig_num}"
        if f_num not in lookup[key]:
            lookup[key].append(f_num)

    conn.close()

    # Convert to regular dict and sort fnums
    lookup = {k: sorted(v) for k, v in lookup.items()}

    print(f"Found {len(lookup)} unique (affiliation, designation) pairs")

    # Write to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(lookup, f)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Written to: {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
