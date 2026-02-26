"""Gazetteer scoring logic for the structured classifier.

Scores all gazetteer records against per-field log-probabilities
from the structured classifier to find the best match.
"""

import math
from collections import defaultdict

import torch

from .classifier import FIELDS, POINTER_FIELDS

# Per-field fallback log-prob when record value not found in query text.
POINTER_NOT_FOUND_LOG_PROB = {
    "desig_num": math.log(1e-10),  # effectively rules out record
    "prefix": math.log(0.329),  # 32.9% absent — mild
    "suffix": math.log(0.444),  # 44.4% absent — mild
}


def _get_field_value(record, field):
    """Get field value from record."""
    if field == "f_num":
        return record["f_num"]
    if field == "desig_num":
        return record.get("desig_num", 0)
    return record.get(field, "")


def build_field_vocabs(train_examples):
    """Build value->idx mappings for each field from training data."""
    vocabs = {}
    for f in FIELDS:
        values = set()
        for ex in train_examples:
            if ex["records"]:
                values.add(_get_field_value(ex["records"][0], f))
        vocabs[f] = {v: i for i, v in enumerate(sorted(values, key=str))}
    return vocabs


def _normalize_pointer_value(val):
    """Normalize a pointer field value for matching against query tokens."""
    if val is None:
        return None
    val = str(val).strip()
    if not val or val == "0":
        return None
    if val.isdigit():
        val = val.lstrip("0") or "0"
        return None if val == "0" else val
    return val.lower()


def build_gazetteer_matrix(fnum_to_records, field_vocabs):
    """Build a matrix of field indices for all gazetteer records.

    Returns:
        field_indices: dict of field -> tensor (n_records,) [classification only]
        field_known: dict of field -> bool tensor (n_records,)
        record_fnums: list of f_num for each row
        records_list: list of raw record dicts
    """
    records = []
    record_fnums = []
    for fnum, recs in fnum_to_records.items():
        for rec in recs:
            records.append(rec)
            record_fnums.append(int(fnum))

    field_indices = {}
    field_known = {}

    for f in FIELDS:
        if f in POINTER_FIELDS:
            continue
        indices = []
        known = []
        for rec in records:
            val = _get_field_value(rec, f)
            idx = field_vocabs[f].get(val)
            if idx is not None:
                indices.append(idx)
                known.append(True)
            else:
                indices.append(0)
                known.append(False)
        field_indices[f] = torch.tensor(indices, dtype=torch.long)
        field_known[f] = torch.tensor(known, dtype=torch.bool)

    return field_indices, field_known, record_fnums, records


def build_pointer_lookup(records_list, field):
    """Build mapping from normalized values to record indices for a pointer field.

    Returns:
        value_to_indices: dict of str -> tensor of record indices
        none_indices: tensor of record indices where value is NONE
    """
    val_to_idx = defaultdict(list)
    none_indices = []
    for i, rec in enumerate(records_list):
        normalized = _normalize_pointer_value(_get_field_value(rec, field))
        if normalized is None:
            none_indices.append(i)
        else:
            val_to_idx[normalized].append(i)

    value_to_indices = {
        v: torch.tensor(idxs, dtype=torch.long) for v, idxs in val_to_idx.items()
    }
    none_indices = torch.tensor(none_indices, dtype=torch.long)
    return value_to_indices, none_indices
