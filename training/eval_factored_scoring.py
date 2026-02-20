#!/usr/bin/env python3
"""
Evaluate factored scoring: use the structured classifier's per-field
probability distributions to score all gazetteer records and find the
best match for each test query.

No retrieval step — exhaustive scoring over all records.

Supports pointer heads: for pointer fields (e.g. desig_num), scoring
checks if the record's value appears as a token in the query and uses
the pointer probability for that position. If not found, the score is
a configurable fallback (default: log(0.01 / n_unseen), effectively -inf).
"""

import json
import math
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_structured_classifier import (
    COLLATE_FNS,
    DEVICE,
    FIELDS,
    MAX_TOKENS,
    POINTER_FIELDS,
    StructuredClassifier,
    StructuredDataset,
    _get_field_value,
    build_field_vocabs,
    model_path,
    smart_truncate_nonspace,
)

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
GAZETTEER_PATH = DATA_DIR / "fnum_to_records.json"

# Per-field fallback log-prob when record value not found in query text.
# desig_num: almost always in text, so missing is strong negative evidence.
# prefix/suffix: often absent even for correct records, use empirical absent rates.
POINTER_NOT_FOUND_LOG_PROB = {
    "desig_num": math.log(1e-10),  # -23.03 — effectively rules out record
    "prefix": math.log(0.329),  # 32.9% absent — mild
    "suffix": math.log(0.444),  # 44.4% absent — mild
}


def load_model(ckpt):
    """Load trained structured classifier from checkpoint."""
    encoder = ckpt.get("encoder", "char")
    field_sizes = ckpt["field_sizes"]
    model = StructuredClassifier(
        field_sizes=field_sizes,
        encoder=encoder,
        d_model=ckpt["d_model"],
        n_heads=4,
        n_layers=ckpt["n_layers"],
        ff_dim=ckpt["d_model"] * 2,
        dropout=0.0,  # no dropout at inference
        token_vocab_size=len(ckpt["token_vocab"]) if "token_vocab" in ckpt else None,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def build_gazetteer_matrix(fnum_to_records, field_vocabs):
    """Build a matrix of field indices for all gazetteer records.

    For classification fields: field_indices[f] = tensor of vocab indices
    For pointer fields: skipped (handled separately via raw values)

    Returns:
        field_indices: dict of field -> tensor of shape (n_records,) [classification only]
        record_fnums: list of f_num for each row
        records_list: list of raw record dicts (for pointer field lookups)
    """
    records = []
    record_fnums = []
    for fnum, recs in fnum_to_records.items():
        for rec in recs:
            records.append(rec)
            record_fnums.append(int(fnum))

    field_indices = {}
    field_known = {}  # bool mask: True where value is in vocab

    for f in FIELDS:
        if f in POINTER_FIELDS:
            continue  # pointer fields handled separately
        indices = []
        known = []
        for rec in records:
            val = _get_field_value(rec, f)
            idx = field_vocabs[f].get(val)
            if idx is not None:
                indices.append(idx)
                known.append(True)
            else:
                indices.append(0)  # placeholder
                known.append(False)
        field_indices[f] = torch.tensor(indices, dtype=torch.long)
        field_known[f] = torch.tensor(known, dtype=torch.bool)

    n_known_all = sum(
        all(field_known[f][i] for f in field_known) for i in range(len(records))
    )
    print(f"Gazetteer: {len(records)} records, {n_known_all} with all fields in vocab")

    return field_indices, field_known, record_fnums, records


def _normalize_pointer_value(val):
    """Normalize a pointer field value for matching against query tokens.

    Numbers: strip leading zeros (matching tokenizer behavior).
    Non-numbers: lowercase (tokens are lowercased by tokenizer).
    Empty/zero: returns None (means NONE/absent).
    """
    val = str(val).strip()
    if not val or val == "0":
        return None
    if val.isdigit():
        return val.lstrip("0") or "0"
    return val.lower()


def build_pointer_lookup(records_list, field):
    """Build a mapping from normalized string values to record indices.

    Returns:
        value_to_indices: dict of str -> tensor of record indices
        none_indices: tensor of record indices where value is NONE
    """
    from collections import defaultdict

    val_to_idx = defaultdict(list)
    none_indices = []
    for i, rec in enumerate(records_list):
        normalized = _normalize_pointer_value(_get_field_value(rec, field))
        if normalized is None:
            none_indices.append(i)
        else:
            val_to_idx[normalized].append(i)

    # Convert to tensors
    value_to_indices = {
        v: torch.tensor(idxs, dtype=torch.long) for v, idxs in val_to_idx.items()
    }
    none_indices = torch.tensor(none_indices, dtype=torch.long)
    return value_to_indices, none_indices


def get_model_input(inputs, encoder, device):
    """Extract the right tensor from inputs dict based on encoder type."""
    mask = inputs["mask"].to(device)
    if encoder == "token-embed":
        return inputs["token_ids"].to(device), mask
    else:
        return inputs["char_ids"].to(device), mask


@click.command()
@click.option("--batch-size", default=256)
@click.option("--split", default="test", type=click.Choice(["val", "test"]))
@click.option(
    "--encoder",
    default="char",
    type=click.Choice(["char", "token-charcnn", "token-embed"]),
    help="Which encoder checkpoint to load",
)
@click.option(
    "--fnum-weight",
    default=0.0,
    type=float,
    help="Weight for f_num head log-prob (0=factored only, 1=f_num only)",
)
def main(batch_size, split, encoder, fnum_weight):
    print(f"Device: {DEVICE}")
    print(f"Encoder: {encoder}")
    if fnum_weight > 0:
        print(f"f_num blending weight: {fnum_weight}")
    print("Loading data...")

    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)
    with open(GAZETTEER_PATH) as f:
        fnum_to_records = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for ex in all_examples:
        splits[ex["split"]].append(ex)

    # Load checkpoint and set pointer fields
    print("Loading model...")
    ckpt = torch.load(model_path(encoder), weights_only=False, map_location=DEVICE)
    token_vocab = ckpt.get("token_vocab")

    # Restore pointer fields from checkpoint
    global POINTER_FIELDS
    saved_pointer_fields = ckpt.get("pointer_fields", [])
    POINTER_FIELDS.clear()
    POINTER_FIELDS.update(saved_pointer_fields)
    if POINTER_FIELDS:
        print(f"Pointer fields: {sorted(POINTER_FIELDS)}")

    # Build vocabs from training data
    field_vocabs = build_field_vocabs(splits["train"])

    # Build gazetteer scoring matrix
    field_indices, field_known, record_fnums, records_list = build_gazetteer_matrix(
        fnum_to_records, field_vocabs
    )
    record_fnums_array = np.array(record_fnums)
    n_records = len(record_fnums)
    print(f"Scoring against {n_records} gazetteer records")

    # Build pointer lookup tables for pointer fields
    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    # Move field indices and known masks to device
    field_indices = {f: t.to(DEVICE) for f, t in field_indices.items()}
    field_known = {f: t.to(DEVICE) for f, t in field_known.items()}

    # Load model
    model = load_model(ckpt)

    # Evaluate on split
    eval_examples = splits[split]
    print(f"\nEvaluating on {len(eval_examples)} {split} examples")

    # Build dataset
    eval_ds = StructuredDataset(eval_examples, field_vocabs, encoder, token_vocab)
    collate_fn = COLLATE_FNS[encoder]
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # We need the target f_nums, queries, and token strings aligned with dataset
    target_fnums = []
    query_texts = []
    query_token_strings = []  # for pointer field scoring
    for ex in eval_examples:
        if not ex["records"]:
            continue
        rec = ex["records"][0]
        target_fnums.append(rec["f_num"])
        query_texts.append(ex["query"])
        if POINTER_FIELDS and encoder != "char":
            tokens = smart_truncate_nonspace(ex["query"])
            query_token_strings.append([t["token"] for t in tokens])

    # Build f_num to description for error reporting
    def describe_record(rec):
        parts = [rec["union_name"]]
        if rec.get("desig_name"):
            parts.append(rec["desig_name"])
        if rec.get("desig_num"):
            parts.append(f"num={rec['desig_num']}")
        if rec.get("prefix"):
            parts.append(f"pre={rec['prefix']}")
        if rec.get("suffix"):
            parts.append(f"suf={rec['suffix']}")
        if rec.get("unit_id"):
            parts.append(f"uid={rec['unit_id']}")
        return " / ".join(parts)

    fnum_to_desc = {}
    for fnum, recs in fnum_to_records.items():
        descs = [describe_record(r) for r in recs]
        fnum_to_desc[int(fnum)] = " | ".join(descs)

    correct = 0
    total = 0
    correct_top5 = 0
    errors = []

    with torch.no_grad():
        example_idx = 0
        for inputs, labels in eval_loader:
            model_input, mask = get_model_input(inputs, encoder, DEVICE)
            logits = model(model_input, mask)

            # Get log-probabilities for each field
            log_probs = {f: F.log_softmax(logits[f], dim=-1) for f in FIELDS}

            # Score all gazetteer records for each query in batch
            batch_size_actual = model_input.shape[0]
            for i in range(batch_size_actual):
                # Sum log-probs across structured fields (excluding f_num)
                scores = torch.zeros(n_records, device=DEVICE)

                for f in FIELDS:
                    if f == "f_num":
                        continue  # handled separately for blending
                    if f not in POINTER_FIELDS:
                        # Classification field: lookup with floor for unknown values
                        field_lp = log_probs[f][i][field_indices[f]]
                        # Unknown values get floor log-prob (uniform over vocab)
                        vocab_size = log_probs[f].shape[-1]
                        floor_lp = -math.log(vocab_size)
                        field_lp = torch.where(field_known[f], field_lp, floor_lp)
                        scores += field_lp
                    else:
                        # Pointer field: vectorized scoring
                        query_toks = query_token_strings[example_idx]
                        lp = log_probs[f][i]  # (MAX_TOKENS + 1,)

                        # Build position lookup: token_string -> position index
                        tok_to_pos = {}
                        for pos, tok in enumerate(query_toks):
                            if tok and tok not in tok_to_pos:
                                tok_to_pos[tok] = pos

                        # Start with fallback for all records
                        field_scores = torch.full(
                            (n_records,), POINTER_NOT_FOUND_LOG_PROB[f], device=DEVICE
                        )

                        # NONE records get the NONE log-prob
                        none_idx = pointer_none_indices[f]
                        if len(none_idx) > 0:
                            field_scores[none_idx] = lp[MAX_TOKENS]

                        # For each unique value that appears in query tokens,
                        # scatter the pointer log-prob to all matching records
                        val_to_idx = pointer_val_to_indices[f]
                        for tok, pos in tok_to_pos.items():
                            rec_indices = val_to_idx.get(tok)
                            if rec_indices is not None:
                                field_scores[rec_indices] = lp[pos]

                        scores += field_scores

                # Blend with f_num head if available and weighted.
                #
                # The f_num head is a 13K-class classifier that predicts f_num
                # directly from the query text. Its accuracy correlates strongly
                # with how many training examples exist for each f_num:
                #   n=0: 0%, n=1: 83%, n=4-7: 98%, n=16+: 99.6%
                #
                # For records whose f_num was never seen in training (n=0), the
                # model assigns near-zero probability (~1/13K) to the correct
                # class. It can't predict what it's never seen. But it may
                # assign high confidence (~44%) to some *wrong* f_num that
                # covers similar unions/locals.
                #
                # Per-record weighting addresses this by scaling the f_num
                # contribution based on how many training examples exist for
                # each record's f_num:
                #
                #   w = floor_w + (base_w - floor_w) * min(1, log1p(n) / log1p(sat))
                #
                # Critically, floor_w should be small but positive (e.g. 0.1),
                # not zero. For unseen f_nums, the score is floor_w * log(1/V),
                # a constant penalty applied equally to ALL unseen-f_num records.
                # This doesn't change their ranking relative to each other (it's
                # a constant offset), but it pushes them all down relative to
                # seen-f_num records. This encodes a useful prior: "if the model
                # recognizes a record's f_num, prefer it over one it doesn't
                # recognize, all else being equal." Zeroing out the f_num weight
                # for unseen records throws away this signal entirely, which
                # empirically increases errors.
                if (
                    "f_num" in log_probs
                    and "f_num" in field_indices
                    and fnum_weight > 0
                ):
                    fnum_lp = log_probs["f_num"][i][field_indices["f_num"]]
                    # Unknown f_nums get floor log-prob
                    fnum_vocab_size = log_probs["f_num"].shape[-1]
                    fnum_floor = -math.log(fnum_vocab_size)
                    fnum_lp = torch.where(field_known["f_num"], fnum_lp, fnum_floor)
                    scores = (1 - fnum_weight) * scores + fnum_weight * fnum_lp

                # Top-1 and top-5
                top5_indices = scores.topk(5).indices.cpu().numpy()
                pred_fnum = record_fnums_array[top5_indices[0]]
                top5_fnums = set(record_fnums_array[top5_indices])

                target_fnum = target_fnums[example_idx]
                if pred_fnum == target_fnum:
                    correct += 1
                if target_fnum in top5_fnums:
                    correct_top5 += 1

                if pred_fnum != target_fnum:
                    # Find target rank
                    target_in_gaz = np.where(record_fnums_array == target_fnum)[0]
                    if len(target_in_gaz) > 0:
                        target_score = scores[target_in_gaz[0]].item()
                        target_rank = (
                            scores > scores[target_in_gaz[0]]
                        ).sum().item() + 1
                    else:
                        target_score = float("-inf")
                        target_rank = -1

                    errors.append(
                        {
                            "query": query_texts[example_idx],
                            "target_fnum": target_fnum,
                            "target_desc": fnum_to_desc.get(target_fnum, "?"),
                            "target_score": target_score,
                            "target_rank": int(target_rank),
                            "pred_fnum": int(pred_fnum),
                            "pred_desc": fnum_to_desc.get(int(pred_fnum), "?"),
                            "pred_score": scores[top5_indices[0]].item(),
                        }
                    )
                total += 1
                example_idx += 1

            if total % 1000 < batch_size:
                print(
                    f"  {total}/{len(target_fnums)}: top-1={correct/total:.4f}, top-5={correct_top5/total:.4f}"
                )

    print(f"\nFinal {split} results:")
    print(f"  Top-1: {correct}/{total} = {correct/total:.4f} ({total-correct} errors)")
    print(f"  Top-5: {correct_top5}/{total} = {correct_top5/total:.4f}")

    # Also report: how many errors are due to target not in gazetteer?
    target_fnum_set = set(record_fnums)
    missing = sum(1 for fn in target_fnums if fn not in target_fnum_set)
    if missing:
        print(f"  ({missing} target f_nums not in gazetteer — impossible to get right)")

    # Write all errors to CSV
    errors.sort(key=lambda e: e["target_rank"])
    errors_path = DATA_DIR / f"factored_scoring_errors_{split}.csv"
    with open(errors_path, "w", newline="") as f:
        import csv

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "target_fnum",
                "target_desc",
                "target_score",
                "target_rank",
                "pred_fnum",
                "pred_desc",
                "pred_score",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)
    print(f"\nErrors written to {errors_path}")

    # Print summary to stdout
    print(f"\n{'='*80}")
    print(f"Errors ({len(errors)} total, showing first 20):")
    print(f"{'='*80}")
    for e in errors[:20]:
        print(f"\n  Query: {e['query']}")
        print(f"  Target (rank {e['target_rank']}): {e['target_desc']}")
        print(f"  Pred:   {e['pred_desc']}")
        print(
            f"  Scores: pred={e['pred_score']:.2f}, target={e['target_score']:.2f}, gap={e['pred_score']-e['target_score']:.2f}"
        )


if __name__ == "__main__":
    main()
