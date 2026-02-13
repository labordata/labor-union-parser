#!/usr/bin/env python3
"""
Diagnostic: trace cross-attention weights for error cases.

For each query/record pair, shows:
- Query tokens (what the CharCNN/frozen_num sees)
- Record fields (union, desig, prefix, number, suffix, unit_id)
- Cross-attention weights from each query token to each field
- Which fields are masked
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
        field_labels: list of field name strings
        field_mask: [6] bool tensor
        attn_weights_1: [num_heads, seq_len, 6] from layer 1
        attn_weights_2: [num_heads, seq_len, 6] from layer 2
        score: scalar reranking score
    """
    # Encode query
    q_batch = encode_query_batch([query])
    q_batch = {k: v.to(DEVICE) for k, v in q_batch.items()}

    # Encode record
    r_batch = encode_record_batch([record], vocab)
    r_batch = {k: v.to(DEVICE) for k, v in r_batch.items()}

    with torch.no_grad():
        # Get query token embeddings
        token_emb, padding_mask = model.query_encoder(
            q_batch["char_ids"], q_batch["is_number"], q_batch["numeric_ids"]
        )

        # Get record field embeddings
        field_emb, field_mask = model.record_encoder(
            r_batch["union_idx"],
            r_batch["desig_idx"],
            r_batch["prefix_hash"],
            r_batch["num_hash"],
            r_batch["num_val"],
            r_batch["suffix_idx"],
            r_batch["unit_id_idx"],
        )

        # Field self-attention (as in CrossAttentionHead.score_pair)
        ca = model.cross_attention
        field_enhanced, _ = ca.field_self_attn(
            field_emb, field_emb, field_emb, key_padding_mask=field_mask
        )
        field_emb_sa = ca.field_norm(field_emb + field_enhanced)

        # Cross-attention layer 1 - compute attention weights manually
        attn_weights_1 = _compute_cross_attn_weights(
            ca.cross_attn1, token_emb, field_emb_sa, padding_mask, field_mask
        )

        # Get output of layer 1 for input to layer 2
        enhanced = ca.cross_attn1(token_emb, field_emb_sa, padding_mask, field_mask)

        # Cross-attention layer 2
        attn_weights_2 = _compute_cross_attn_weights(
            ca.cross_attn2, enhanced, field_emb_sa, padding_mask, field_mask
        )

        # Get final score
        enhanced2 = ca.cross_attn2(enhanced, field_emb_sa, padding_mask, field_mask)
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        pooled = (enhanced2 * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        score = ca.classifier(pooled).squeeze(-1).item()

    token_labels = get_query_tokens(query)
    field_values = _get_field_values(record, vocab)

    return (
        token_labels,
        field_values,
        field_mask[0],
        attn_weights_1,
        attn_weights_2,
        score,
    )


def _compute_cross_attn_weights(layer, query_emb, field_emb, query_mask, field_mask):
    """Manually compute cross-attention weights for a CrossAttentionLayer."""
    B, Q, _ = query_emb.shape
    _, K, _ = field_emb.shape

    q = (
        layer.q_proj(query_emb)
        .view(B, Q, layer.num_heads, layer.head_dim)
        .transpose(1, 2)
    )
    k = (
        layer.k_proj(field_emb)
        .view(B, K, layer.num_heads, layer.head_dim)
        .transpose(1, 2)
    )

    scores = torch.matmul(q, k.transpose(-2, -1)) / (layer.head_dim**0.5)

    if field_mask is not None:
        mask = field_mask.unsqueeze(1).unsqueeze(2).expand(B, layer.num_heads, Q, K)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    return attn[0]  # [num_heads, Q, K]


def _get_field_values(record, vocab):
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
    token_labels, field_values, field_mask, attn1, attn2, score = (
        get_cross_attention_weights(model, query, record, vocab)
    )

    print(f"\n{'='*80}")
    if label:
        print(f"  {label}")
    print(f"  Query: {query}")
    print(f"  Score: {score:.4f}")
    print(f"  Record: {field_values}")
    print(
        f"  Field mask: {['MASKED' if field_mask[i] else 'active' for i in range(6)]}"
    )
    print(f"{'='*80}")

    # Average across heads
    attn1_avg = attn1.mean(0)  # [Q, 6]
    attn2_avg = attn2.mean(0)  # [Q, 6]

    # Print layer 1
    print("\n  Cross-Attention Layer 1 (avg across 4 heads):")
    header = f"  {'token':<15} " + " ".join(f"{fn:>8}" for fn in FIELD_NAMES)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (tok, is_num) in enumerate(token_labels):
        num_flag = "*" if is_num else " "
        weights = attn1_avg[i]
        row = f"  {num_flag}{tok:<14} " + " ".join(f"{w:>8.3f}" for w in weights)
        print(row)

    # Print layer 2
    print("\n  Cross-Attention Layer 2 (avg across 4 heads):")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, (tok, is_num) in enumerate(token_labels):
        num_flag = "*" if is_num else " "
        weights = attn2_avg[i]
        row = f"  {num_flag}{tok:<14} " + " ".join(f"{w:>8.3f}" for w in weights)
        print(row)

    # Per-head breakdown for layer 2 (the one that matters most for final score)
    print("\n  Layer 2 per-head breakdown:")
    for h in range(attn2.shape[0]):
        print(f"  --- Head {h} ---")
        for i, (tok, is_num) in enumerate(token_labels):
            num_flag = "*" if is_num else " "
            weights = attn2[h, i]
            max_field = FIELD_NAMES[weights.argmax().item()]
            row = (
                f"    {num_flag}{tok:<14} "
                + " ".join(f"{w:>7.3f}" for w in weights)
                + f"  -> {max_field}"
            )
            print(row)


def main():

    if len(sys.argv) < 4:
        print(
            "Usage: python inspect_attention.py <query> <target_fnum> <pred_fnum> [checkpoint]"
        )
        print(
            "  Example: python inspect_attention.py 'GCC/IBT Council 1 1B' 543800 514008"
        )
        sys.exit(1)

    query = sys.argv[1]
    target_fnum = int(sys.argv[2])
    pred_fnum = int(sys.argv[3])

    checkpoint = (
        sys.argv[4]
        if len(sys.argv) > 4
        else str(
            sorted(
                [
                    p
                    for p in Path(__file__).parent.glob("dual_task_model-v*.ckpt")
                    if p.stem.split("-v")[1].isdigit()
                ],
                key=lambda p: int(p.stem.split("-v")[1]),
            )[-1]
        )
    )
    print(f"Loading model from {checkpoint}...")
    model, vocab = load_model(checkpoint)

    # Load fnum_to_records for looking up records
    with open(Path(__file__).parent / "data" / "fnum_to_records.json") as f:
        fnum_to_records = {int(k): v for k, v in json.load(f).items()}

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
