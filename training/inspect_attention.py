#!/usr/bin/env python3
"""
Diagnostic: trace cross-attention weights for error cases.

For each query/record pair, shows:
- Query tokens (what the CharCNN/frozen_num sees)
- Record fields (union, desig, prefix, number, suffix, unit_id)
- Cross-attention weights from each query token to each field
- Which fields are masked
"""

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import (
    encode_query_batch,
    encode_record_batch,
)
from prepare_data import filter_records_by_query
from train_dual_task import DEVICE, VOCAB_PATH

from labor_union_parser.char_cnn import tokenize_to_chars
from labor_union_parser.model import DualTaskModel

FIELD_NAMES = ["union", "desig", "prefix", "number", "suffix", "unit_id"]


def load_model(checkpoint_path):
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    model = DualTaskModel(
        num_union_names=len(vocab["union_name_to_idx"]),
        num_desig_names=len(vocab["desig_name_to_idx"]),
        num_suffixes=len(vocab["suffix_to_idx"]),
        num_unit_ids=len(vocab["unit_id_to_idx"]),
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "state_dict" in ckpt:
        state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    else:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, vocab


def get_query_tokens(query, max_len=20):
    """Get human-readable token list for a query."""
    _, tokens, is_num, token_types, _ = tokenize_to_chars(query, 999)

    # Filter to non-space tokens (matching smart_truncate_nonspace)
    nonspace = []
    for i, tt in enumerate(token_types):
        if tokens[i] and tt != 2:  # not space
            nonspace.append((tokens[i], bool(is_num[i])))

    return nonspace[:max_len]


def get_cross_attention_weights(model, query, record, vocab):
    """
    Run cross-attention and extract attention weights from both layers.

    Returns:
        token_labels: list of (token_str, is_number) for query tokens
        field_values: dict of field name -> value
        attn_weights_1: [num_heads, seq_len, 6] from layer 1
        attn_weights_2: [num_heads, seq_len, 6] from layer 2
        score: scalar reranking score
    """
    q_batch = encode_query_batch([query])
    r_batch = encode_record_batch([record], vocab)
    tensors = {**q_batch, **r_batch}
    tensors = {k: v.to(DEVICE) for k, v in tensors.items()}

    with torch.no_grad():
        score, attn1, attn2 = model.forward_reranking_with_attention(
            tensors["char_ids"],
            tensors["is_number"],
            tensors["numeric_ids"],
            tensors["union_idx"],
            tensors["desig_idx"],
            tensors["prefix_hash"],
            tensors["num_hash"],
            tensors["suffix_idx"],
            tensors["unit_id_idx"],
        )

    return (
        get_query_tokens(query),
        _get_field_values(record),
        attn1[0],  # remove batch dim
        attn2[0],
        score.item(),
    )


def _get_field_values(record):
    """Get human-readable field values."""
    return {
        "union": record.get("union_name", ""),
        "desig": record.get("desig_name", ""),
        "prefix": record.get("prefix", 0),
        "number": record.get("desig_num", 0),
        "suffix": record.get("suffix", ""),
        "unit_id": record.get("unit_id", ""),
    }


def print_attention_analysis(query, record, vocab, model, label=""):
    """Print full attention analysis for a query-record pair."""
    token_labels, field_values, attn1, attn2, score = get_cross_attention_weights(
        model, query, record, vocab
    )

    print(f"\n{'='*80}")
    if label:
        print(f"  {label}")
    print(f"  Query: {query}")
    print(f"  Score: {score:.4f}")
    print(f"  Record: {field_values}")
    print(f"{'='*80}")

    # Token labels including [CLS] at position 0
    all_tokens = [("[CLS]", False)] + token_labels

    # Average across heads
    attn1_avg = attn1.mean(0)  # [seq_len+1, 6]
    attn2_avg = attn2.mean(0)

    # Print layer 1
    print("\n  Cross-Attention Layer 1 (avg across 4 heads):")
    header = f"  {'token':<15} " + " ".join(f"{fn:>8}" for fn in FIELD_NAMES)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (tok, is_num) in enumerate(all_tokens):
        num_flag = "*" if is_num else " "
        weights = attn1_avg[i]
        row = f"  {num_flag}{tok:<14} " + " ".join(f"{w:>8.3f}" for w in weights)
        print(row)

    # Print layer 2
    print("\n  Cross-Attention Layer 2 (avg across 4 heads):")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (tok, is_num) in enumerate(all_tokens):
        num_flag = "*" if is_num else " "
        weights = attn2_avg[i]
        row = f"  {num_flag}{tok:<14} " + " ".join(f"{w:>8.3f}" for w in weights)
        print(row)

    # [CLS] attention detail across heads in layer 2 (drives the score)
    print("\n  [CLS] attention per head (layer 2 — drives the score):")
    for h in range(attn2.shape[0]):
        weights = attn2[h, 0]  # [CLS] is at position 0
        row = f"    Head {h}: " + " ".join(
            f"{fn}={w:.3f}" for fn, w in zip(FIELD_NAMES, weights)
        )
        print(row)


def _resolve_checkpoint():
    """Find the latest versioned checkpoint."""
    return str(
        sorted(
            [
                p
                for p in Path(__file__).parent.glob("dual_task_model-v*.ckpt")
                if p.stem.split("-v")[1].isdigit()
            ],
            key=lambda p: int(p.stem.split("-v")[1]),
        )[-1]
    )


def _load_fnum_to_records():
    with open(Path(__file__).parent / "data" / "fnum_to_records.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def analyze_errors_csv(model, vocab, fnum_to_records, csv_path, max_detail=10):
    """Analyze errors from pipeline_errors.csv with detailed attention output."""
    with open(csv_path) as f:
        errors = list(csv.DictReader(f))

    print(f"Loaded {len(errors)} errors from {csv_path}\n")

    count = 0
    for err in errors:
        if count >= max_detail:
            break

        target_fnum = int(err["target_fnum"])
        pred_fnum = int(err["pred_fnum"])
        query = err["query"]

        target_recs = fnum_to_records.get(target_fnum, [])
        pred_recs = fnum_to_records.get(pred_fnum, [])

        target_filtered = filter_records_by_query(query, target_recs)
        pred_filtered = filter_records_by_query(query, pred_recs)

        if not target_filtered and not pred_filtered:
            continue

        if target_filtered:
            print_attention_analysis(
                query,
                target_filtered[0],
                vocab,
                model,
                label=f"TARGET (f_num={target_fnum})",
            )
        if pred_filtered:
            print_attention_analysis(
                query,
                pred_filtered[0],
                vocab,
                model,
                label=f"PREDICTED (f_num={pred_fnum})",
            )
        count += 1


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Batch:  python inspect_attention.py errors [checkpoint] [max_detail]")
        print(
            "  Single: python inspect_attention.py <query> <target_fnum>"
            " <pred_fnum> [checkpoint]"
        )
        sys.exit(1)

    if sys.argv[1] == "errors":
        checkpoint = sys.argv[2] if len(sys.argv) > 2 else _resolve_checkpoint()
        max_detail = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        print(f"Loading model from {checkpoint}...")
        model, vocab = load_model(checkpoint)
        fnum_to_records = _load_fnum_to_records()
        csv_path = Path(__file__).parent / "data" / "pipeline_errors.csv"
        analyze_errors_csv(model, vocab, fnum_to_records, csv_path, max_detail)
    else:
        if len(sys.argv) < 4:
            print(
                "Usage: python inspect_attention.py <query> <target_fnum>"
                " <pred_fnum> [checkpoint]"
            )
            sys.exit(1)

        query = sys.argv[1]
        target_fnum = int(sys.argv[2])
        pred_fnum = int(sys.argv[3])
        checkpoint = sys.argv[4] if len(sys.argv) > 4 else _resolve_checkpoint()

        print(f"Loading model from {checkpoint}...")
        model, vocab = load_model(checkpoint)
        fnum_to_records = _load_fnum_to_records()

        target_recs = fnum_to_records.get(target_fnum, [])
        pred_recs = fnum_to_records.get(pred_fnum, [])

        target_filtered = filter_records_by_query(query, target_recs)
        pred_filtered = filter_records_by_query(query, pred_recs)

        if target_filtered:
            print_attention_analysis(
                query,
                target_filtered[0],
                vocab,
                model,
                label=f"TARGET (f_num={target_fnum})",
            )
        if pred_filtered:
            print_attention_analysis(
                query,
                pred_filtered[0],
                vocab,
                model,
                label=f"PREDICTED (f_num={pred_fnum})",
            )


if __name__ == "__main__":
    main()
