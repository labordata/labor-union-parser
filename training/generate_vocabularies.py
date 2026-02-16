#!/usr/bin/env python3
"""
Generate vocabularies.json: build index mappings for all categorical
fields used by the model.

Index conventions match the original training code:
  - union_name:  no empty-string entry, indices start at 0
  - desig_name:  indices start at 1, 0 reserved for padding/unknown
  - suffix:      empty-string at 0, normalized (leading-zero stripped)
  - unit_id:     no empty-string entry, indices start at 0

Inputs:
    training/data/fnum_to_records.json — all record variants per f_num

Output:
    training/data/vocabularies.json
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FNUM_RECORDS_PATH = DATA_DIR / "fnum_to_records.json"
UNIT_ID_CSV = Path(__file__).parent / "fnum_to_unit_identifier.csv"
OUTPUT_PATH = DATA_DIR / "vocabularies.json"


def normalize_designation(s: str) -> str:
    """Normalize designation strings — strip leading zeros from numeric values."""
    if not s:
        return s
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


def main():
    print(f"Loading fnum_to_records from {FNUM_RECORDS_PATH}...")
    with open(FNUM_RECORDS_PATH) as f:
        raw = json.load(f)
    fnum_to_records = {int(k): v for k, v in raw.items()}

    total_records = sum(len(recs) for recs in fnum_to_records.values())
    print(f"  {len(fnum_to_records)} f_nums, {total_records} records")

    # Collect all unique values for each field
    union_names = set()
    desig_names = set()
    suffixes = set()

    for records in fnum_to_records.values():
        for rec in records:
            union_names.add(rec["union_name"])
            desig_names.add(rec.get("desig_name", ""))
            raw_suffix = rec.get("suffix", "") or ""
            suffixes.add(normalize_designation(raw_suffix))

    # Union names: no empty-string entry, indices start at 0
    union_name_to_idx = {name: i for i, name in enumerate(sorted(union_names))}

    # Desig names: indices start at 1, 0 reserved for padding/unknown
    desig_names.discard("")
    desig_name_to_idx = {name: i + 1 for i, name in enumerate(sorted(desig_names))}

    # Suffixes: empty-string at 0, rest sorted starting at 1
    suffixes.discard("")
    suffix_to_idx = {"": 0}
    for i, s in enumerate(sorted(suffixes)):
        suffix_to_idx[s] = i + 1

    # Unit IDs: built from CSV in insertion order (empty string at 0)
    unit_id_to_idx = {}
    idx = 0
    with open(UNIT_ID_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["unit_identifier"]
            if uid not in unit_id_to_idx:
                unit_id_to_idx[uid] = idx
                idx += 1

    print(f"  {len(union_name_to_idx)} union names")
    print(f"  {len(desig_name_to_idx)} desig names")
    print(f"  {len(suffix_to_idx)} suffixes")
    print(f"  {len(unit_id_to_idx)} unit ids")

    vocab = {
        "union_name_to_idx": union_name_to_idx,
        "desig_name_to_idx": desig_name_to_idx,
        "suffix_to_idx": suffix_to_idx,
        "unit_id_to_idx": unit_id_to_idx,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(vocab, f)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
