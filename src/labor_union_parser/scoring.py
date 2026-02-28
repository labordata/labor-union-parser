"""Gazetteer scoring logic for the structured classifier.

Scores all gazetteer records against per-field log-probabilities
from the structured classifier to find the best match.
"""

import math
from collections import defaultdict

import torch

from .classifier import FIELDS, MAX_TOKENS, POINTER_FIELDS

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
                val = _get_field_value(ex["records"][0], f)
                if val != -100:
                    values.add(val)
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


def compute_record_features(
    log_probs,
    token_strings_batch,
    field_indices,
    field_known,
    pointer_val_to_indices,
    pointer_none_indices,
    n_records,
):
    """Compute per-(query, record) feature vectors for gazetteer scoring.

    Args:
        log_probs: dict of field -> (B, n_classes) temperature-scaled log-prob tensors
        token_strings_batch: list of list of str, query token strings per batch item
        field_indices: dict of classification field -> (R,) index tensor
        field_known: dict of classification field -> (R,) bool tensor
        pointer_val_to_indices: dict of pointer field -> {str: tensor of record indices}
        pointer_none_indices: dict of pointer field -> tensor of record indices
        n_records: int, number of gazetteer records (R)

    Returns:
        features: (B, R, 12) tensor with layout:
            [lp_union, lp_desig, lp_fnum, lp_designum, lp_prefix, lp_suffix,
             unk_union, unk_desig, unk_fnum, nf_designum, nf_prefix, nf_suffix]
    """
    device = log_probs["union_name"].device
    B = len(token_strings_batch)
    R = n_records

    # Classification fields: lp where known, 0 where unknown
    cls_fields = ("union_name", "desig_name", "f_num")
    lp_cls = []  # features 0, 1, 2
    unk_cls = []  # features 6, 7, 8
    for f in cls_fields:
        field_lp = log_probs[f][:, field_indices[f]]  # (B, R)
        known = field_known[f]  # (R,)
        lp_cls.append(torch.where(known, field_lp, torch.zeros_like(field_lp)))
        unk_cls.append((~known).float().expand(B, -1))

    # Pointer fields: lp where found, 0 where not-found
    ptr_fields = ("desig_num", "prefix", "suffix")
    lp_ptr = []  # features 3, 4, 5
    nf_ptr = []  # features 9, 10, 11

    for f in ptr_fields:
        field_scores = torch.zeros(B, R, device=device)
        is_not_found = torch.ones(B, R, device=device)

        none_idx = pointer_none_indices[f]
        if len(none_idx) > 0:
            field_scores[:, none_idx] = log_probs[f][:, MAX_TOKENS].unsqueeze(1)
            is_not_found[:, none_idx] = 0.0

        val_to_idx = pointer_val_to_indices[f]
        all_batch_idx = []
        all_rec_idx = []
        all_positions = []

        for i, query_toks in enumerate(token_strings_batch):
            tok_to_pos = {}
            for pos, tok in enumerate(query_toks):
                if tok and tok not in tok_to_pos:
                    tok_to_pos[tok] = pos
            for tok, pos in tok_to_pos.items():
                rec_indices = val_to_idx.get(tok)
                if rec_indices is not None:
                    n = len(rec_indices)
                    all_batch_idx.append(torch.full((n,), i, dtype=torch.long))
                    all_rec_idx.append(rec_indices)
                    all_positions.append((i, pos, n))

        if all_batch_idx:
            bi = torch.cat(all_batch_idx)
            ri = torch.cat(all_rec_idx).to(device)
            vals = torch.empty(bi.shape[0], device=device)
            offset = 0
            for i, pos, n in all_positions:
                vals[offset : offset + n] = log_probs[f][i, pos]
                offset += n
            field_scores[bi, ri] = vals
            is_not_found[bi, ri] = 0.0

        lp_ptr.append(field_scores)
        nf_ptr.append(is_not_found)

    # Stack into (B, R, 12)
    all_features = lp_cls + lp_ptr + unk_cls + nf_ptr
    features = torch.stack(all_features, dim=2)  # (B, R, 12)
    return features
