#!/usr/bin/env python3
"""Evaluate factored scoring on the full gazetteer.

Uses the structured classifier's per-field probability distributions to
score all gazetteer records and find the best match for each test query.
No retrieval step — exhaustive scoring over all records.
"""

import csv
import json
import math
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_structured_classifier import (
    StructuredDataset,
    collate_fn,
)

from labor_union_parser.classifier import (
    FIELDS,
    MAX_TOKENS,
    POINTER_FIELDS,
    StructuredClassifier,
)
from labor_union_parser.scoring import (
    POINTER_NOT_FOUND_LOG_PROB,
    build_gazetteer_matrix,
    build_pointer_lookup,
)
from labor_union_parser.tokenizer import smart_truncate_nonspace

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"

DEVICE = torch.accelerator.current_accelerator() or torch.device("cpu")


def load_model(ckpt):
    """Load trained structured classifier from checkpoint."""
    model = StructuredClassifier(
        field_sizes=ckpt["field_sizes"],
        d_model=ckpt["d_model"],
        n_heads=4,
        n_layers=ckpt["n_layers"],
        ff_dim=ckpt["d_model"] * 2,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@click.command()
@click.option("--batch-size", default=256)
@click.option("--split", default="test", type=click.Choice(["val", "test"]))
@click.option(
    "--base-w", default=0.6, type=float, help="Base fnum weight for well-seen records"
)
@click.option(
    "--floor-w", default=0.1, type=float, help="Floor fnum weight for unseen records"
)
@click.option(
    "--sat", default=16, type=int, help="Saturation count for fnum weight curve"
)
@click.option(
    "--use-temperatures",
    is_flag=True,
    help="Use fitted post-hoc temperatures instead of hand-tuned weighting",
)
@click.option(
    "--fnum-weight",
    default=None,
    type=float,
    help="With --use-temperatures, weight on f_num log-prob for known records",
)
@click.option(
    "--fnum-weight-unknown",
    default=None,
    type=float,
    help="With --use-temperatures, weight on f_num log-prob for unknown records (default: same as --fnum-weight)",
)
def main(
    batch_size,
    split,
    base_w,
    floor_w,
    sat,
    use_temperatures,
    fnum_weight,
    fnum_weight_unknown,
):
    print(f"Device: {DEVICE}")
    if use_temperatures:
        print("Using fitted post-hoc temperatures")
    else:
        print(f"f_num weighting: base_w={base_w}, floor_w={floor_w}, sat={sat}")
    print("Loading data...")

    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for ex in all_examples:
        splits[ex["split"]].append(ex)

    # Load checkpoint (includes model, vocabs, gazetteer, fnum counts)
    print("Loading model...")
    weights_dir = (
        Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
    )
    ckpt = torch.load(
        weights_dir / "structured_classifier.pt",
        weights_only=False,
        map_location=DEVICE,
    )

    field_vocabs = ckpt["field_vocabs"]
    fnum_to_records = ckpt["gazetteer"]
    fnum_train_counts = ckpt["fnum_train_counts"]

    # Build gazetteer scoring matrix
    field_indices, field_known, record_fnums, records_list = build_gazetteer_matrix(
        fnum_to_records, field_vocabs
    )
    record_fnums_array = np.array(record_fnums)
    n_records = len(record_fnums)
    print(f"Scoring against {n_records} gazetteer records")

    # Load fitted temperatures or build hand-tuned fnum weights
    head_temperatures = None
    fnum_weights = None
    if use_temperatures:
        temps_path = DATA_DIR / "temperatures.json"
        with open(temps_path) as f:
            temps = json.load(f)
        head_temperatures = {}
        for field in FIELDS:
            if field in temps:
                head_temperatures[field] = temps[field]
                print(f"  {field}: T={temps[field]:.4f}")
    else:
        fnum_weights = torch.zeros(n_records, device=DEVICE)
        for i, fnum in enumerate(record_fnums):
            n = fnum_train_counts.get(str(fnum), 0)
            fnum_weights[i] = floor_w + (base_w - floor_w) * min(
                1, math.log1p(n) / math.log1p(sat)
            )

    # Build pointer lookup tables
    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    # Move to device
    field_indices = {f: t.to(DEVICE) for f, t in field_indices.items()}
    field_known = {f: t.to(DEVICE) for f, t in field_known.items()}

    model = load_model(ckpt)

    # Evaluate on split (only examples with records)
    eval_examples = [ex for ex in splits[split] if ex["records"]]
    print(f"\nEvaluating on {len(eval_examples)} {split} examples")

    eval_ds = StructuredDataset(eval_examples, field_vocabs)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Aligned target data
    target_fnums = []
    query_texts = []
    query_token_strings = []
    for ex in eval_examples:
        if not ex["records"]:
            continue
        rec = ex["records"][0]
        target_fnums.append(rec["f_num"])
        query_texts.append(ex["query"])
        tokens = smart_truncate_nonspace(ex["query"])
        query_token_strings.append([t["token"] for t in tokens])

    # Build f_num descriptions for error reporting
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
            char_ids = inputs["char_ids"].to(DEVICE)
            mask = inputs["mask"].to(DEVICE)
            logits = model(char_ids, mask)

            # Apply temperatures to logits before softmax
            if use_temperatures:
                log_probs = {
                    f: F.log_softmax(logits[f] / head_temperatures[f], dim=-1)
                    for f in FIELDS
                }
            else:
                log_probs = {f: F.log_softmax(logits[f], dim=-1) for f in FIELDS}

            batch_size_actual = char_ids.shape[0]
            for i in range(batch_size_actual):
                scores = torch.zeros(n_records, device=DEVICE)

                for f in FIELDS:
                    if f == "f_num":
                        continue
                    if f not in POINTER_FIELDS:
                        field_lp = log_probs[f][i][field_indices[f]]
                        vocab_size = log_probs[f].shape[-1]
                        floor_lp = -math.log(vocab_size)
                        field_lp = torch.where(field_known[f], field_lp, floor_lp)
                        scores += field_lp
                    else:
                        query_toks = query_token_strings[example_idx]
                        lp = log_probs[f][i]

                        tok_to_pos = {}
                        for pos, tok in enumerate(query_toks):
                            if tok and tok not in tok_to_pos:
                                tok_to_pos[tok] = pos

                        field_scores = torch.full(
                            (n_records,), POINTER_NOT_FOUND_LOG_PROB[f], device=DEVICE
                        )

                        none_idx = pointer_none_indices[f]
                        if len(none_idx) > 0:
                            field_scores[none_idx] = lp[MAX_TOKENS]

                        val_to_idx = pointer_val_to_indices[f]
                        for tok, pos in tok_to_pos.items():
                            rec_indices = val_to_idx.get(tok)
                            if rec_indices is not None:
                                field_scores[rec_indices] = lp[pos]

                        scores += field_scores

                # f_num scoring
                fnum_lp = log_probs["f_num"][i][field_indices["f_num"]]
                fnum_vocab_size = log_probs["f_num"].shape[-1]
                fnum_floor = -math.log(fnum_vocab_size)
                fnum_lp = torch.where(field_known["f_num"], fnum_lp, fnum_floor)
                if use_temperatures:
                    # Temperature-scaled: weight f_num differently for
                    # known vs unknown records
                    w = fnum_weight if fnum_weight is not None else 1.0
                    w_unknown = (
                        fnum_weight_unknown if fnum_weight_unknown is not None else w
                    )
                    per_record_w = torch.where(field_known["f_num"], w, w_unknown)
                    scores += per_record_w * fnum_lp
                else:
                    # Hand-tuned: blend with per-record f_num weight
                    scores = (1 - fnum_weights) * scores + fnum_weights * fnum_lp

                top5_indices = scores.topk(5).indices.cpu().numpy()
                pred_fnum = record_fnums_array[top5_indices[0]]
                top5_fnums = set(record_fnums_array[top5_indices])

                target_fnum = target_fnums[example_idx]
                if pred_fnum == target_fnum:
                    correct += 1
                if target_fnum in top5_fnums:
                    correct_top5 += 1

                if pred_fnum != target_fnum:
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
                    f"  {total}/{len(target_fnums)}: "
                    f"top-1={correct/total:.4f}, top-5={correct_top5/total:.4f}"
                )

    print(f"\nFinal {split} results:")
    print(f"  Top-1: {correct}/{total} = {correct/total:.4f} ({total-correct} errors)")
    print(f"  Top-5: {correct_top5}/{total} = {correct_top5/total:.4f}")

    target_fnum_set = set(record_fnums)
    missing = sum(1 for fn in target_fnums if fn not in target_fnum_set)
    if missing:
        print(f"  ({missing} target f_nums not in gazetteer — impossible to get right)")

    # Write errors to CSV
    errors.sort(key=lambda e: e["target_rank"])
    errors_path = DATA_DIR / f"factored_scoring_errors_{split}.csv"
    with open(errors_path, "w", newline="") as f:
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

    print(f"\n{'='*80}")
    print(f"Errors ({len(errors)} total, showing first 20):")
    print(f"{'='*80}")
    for e in errors[:20]:
        print(f"\n  Query: {e['query']}")
        print(f"  Target (rank {e['target_rank']}): {e['target_desc']}")
        print(f"  Pred:   {e['pred_desc']}")
        print(
            f"  Scores: pred={e['pred_score']:.2f}, "
            f"target={e['target_score']:.2f}, "
            f"gap={e['pred_score']-e['target_score']:.2f}"
        )


if __name__ == "__main__":
    main()
