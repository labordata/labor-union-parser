#!/usr/bin/env python3
"""
Evaluate factored scoring: use the structured classifier's per-field
probability distributions to score all gazetteer records and find the
best match for each test query.

No retrieval step — exhaustive scoring over all records.
"""

import json
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
    StructuredClassifier,
    StructuredDataset,
    _get_field_value,
    build_field_vocabs,
    model_path,
)

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
GAZETTEER_PATH = DATA_DIR / "fnum_to_records.json"


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

    Returns:
        field_indices: dict of field -> tensor of shape (n_records,)
        record_fnums: list of f_num for each row
    """
    records = []
    record_fnums = []
    for fnum, recs in fnum_to_records.items():
        for rec in recs:
            records.append(rec)
            record_fnums.append(int(fnum))

    field_indices = {}
    valid_mask = np.ones(len(records), dtype=bool)

    for f in FIELDS:
        indices = []
        for i, rec in enumerate(records):
            val = _get_field_value(rec, f)
            idx = field_vocabs[f].get(val)
            if idx is None:
                valid_mask[i] = False
                indices.append(0)  # placeholder
            else:
                indices.append(idx)
        field_indices[f] = torch.tensor(indices, dtype=torch.long)

    # Filter to valid records
    valid_indices = np.where(valid_mask)[0]
    print(
        f"Gazetteer: {len(records)} records, {len(valid_indices)} with all fields in vocab"
    )

    field_indices = {f: t[valid_indices] for f, t in field_indices.items()}
    record_fnums = [record_fnums[i] for i in valid_indices]

    return field_indices, record_fnums


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
def main(batch_size, split, encoder):
    print(f"Device: {DEVICE}")
    print(f"Encoder: {encoder}")
    print("Loading data...")

    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)
    with open(GAZETTEER_PATH) as f:
        fnum_to_records = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for ex in all_examples:
        splits[ex["split"]].append(ex)

    # Load checkpoint
    print("Loading model...")
    ckpt = torch.load(model_path(encoder), weights_only=False, map_location=DEVICE)
    token_vocab = ckpt.get("token_vocab")

    # Build vocabs from training data
    field_vocabs = build_field_vocabs(splits["train"])

    # Build gazetteer scoring matrix
    field_indices, record_fnums = build_gazetteer_matrix(fnum_to_records, field_vocabs)
    record_fnums_array = np.array(record_fnums)
    n_records = len(record_fnums)
    print(f"Scoring against {n_records} gazetteer records")

    # Move field indices to device
    field_indices = {f: t.to(DEVICE) for f, t in field_indices.items()}

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

    # We need the target f_nums and queries aligned with the dataset
    target_fnums = []
    query_texts = []
    for ex in eval_examples:
        if not ex["records"]:
            continue
        rec = ex["records"][0]
        skip = False
        for f in FIELDS:
            val = _get_field_value(rec, f)
            if field_vocabs[f].get(val) is None:
                skip = True
                break
        if skip:
            continue
        target_fnums.append(rec["f_num"])
        query_texts.append(ex["query"])

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
                # Sum log-probs across fields for all records
                scores = torch.zeros(n_records, device=DEVICE)
                for f in FIELDS:
                    scores += log_probs[f][i][field_indices[f]]

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

    # Print errors sorted by target rank (worst first)
    errors.sort(key=lambda e: e["target_rank"])
    print(f"\n{'='*80}")
    print(f"Errors ({len(errors)} total, showing first 50):")
    print(f"{'='*80}")
    for e in errors[:50]:
        print(f"\n  Query: {e['query']}")
        print(f"  Target (rank {e['target_rank']}): {e['target_desc']}")
        print(f"  Pred:   {e['pred_desc']}")
        print(
            f"  Scores: pred={e['pred_score']:.2f}, target={e['target_score']:.2f}, gap={e['pred_score']-e['target_score']:.2f}"
        )


if __name__ == "__main__":
    main()
