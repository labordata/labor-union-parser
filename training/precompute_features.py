#!/usr/bin/env python3
"""Precompute per-field log-prob features for the scoring layer.

Loads the bundled structured classifier and temperatures, runs forward
passes on train/val/test splits, and writes per-field numpy memmaps
to training/data/precomputed_features/{train,val,test}/.

Also saves metadata (record_fnums, field_known, fnum_to_records mapping,
split sizes) that train_scoring_layer.py needs.
"""

import json
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_structured_classifier import (
    StructuredDataset,
    collate_fn,
)

from labor_union_parser.classifier import (
    FIELDS,
    POINTER_FIELDS,
    StructuredClassifier,
)
from labor_union_parser.scoring import (
    build_gazetteer_matrix,
    build_pointer_lookup,
    compute_record_features,
)

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
FEATURES_DIR = DATA_DIR / "precomputed_features"

# Feature column names matching compute_record_features (B, R, 12) layout.
FEATURE_NAMES = [
    "lp_union",
    "lp_desig",
    "lp_fnum",
    "lp_designum",
    "lp_prefix",
    "lp_suffix",
    "unk_union",
    "unk_desig",
    "unk_fnum",
    "nf_designum",
    "nf_prefix",
    "nf_suffix",
]

NULL_TARGET_REASONS = {"not in gazetteer", "unknown union"}


def load_classifier(ckpt, device):
    model = StructuredClassifier(
        field_sizes=ckpt["field_sizes"],
        d_model=ckpt["d_model"],
        n_heads=4,
        n_layers=ckpt["n_layers"],
        ff_dim=ckpt["d_model"] * 2,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def precompute_to_memmaps(
    model,
    examples,
    split_name,
    n_records,
    field_vocabs,
    field_indices,
    field_known,
    pointer_val_to_indices,
    pointer_none_indices,
    temperatures,
    batch_size,
    device,
):
    split_dir = FEATURES_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    n_queries = len(examples)

    memmaps = {}
    for name in FEATURE_NAMES:
        path = split_dir / f"{name}.npy"
        memmaps[name] = np.memmap(
            path, dtype=np.float32, mode="w+", shape=(n_queries, n_records)
        )

    target_fnums = np.array(
        [
            int(ex["records"][0]["f_num"]) if ex["f_num"] != -100 else -1
            for ex in examples
        ]
    )
    np.save(split_dir / "target_fnums.npy", target_fnums)

    ds = StructuredDataset(examples, field_vocabs)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    row = 0
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc=f"  {split_name}"):
            char_ids = inputs["char_ids"].to(device)
            mask = inputs["mask"].to(device)
            logits = model(char_ids, mask)

            # Temperature-scaled log-softmax, same as extractor.py lines 368-370.
            # Duplicated here because the extractor also needs log_probs for
            # head predictions and field_scores beyond just record features.
            log_probs = {
                f: F.log_softmax(logits[f] / temperatures[f], dim=-1) for f in FIELDS
            }

            bs = char_ids.shape[0]
            features = compute_record_features(
                log_probs,
                ds.token_strings[row : row + bs],
                field_indices,
                field_known,
                pointer_val_to_indices,
                pointer_none_indices,
                n_records,
            )  # (bs, R, 12)

            batch_np = features.cpu().numpy()  # (bs, R, 12)
            for col, name in enumerate(FEATURE_NAMES):
                memmaps[name][row : row + bs] = batch_np[:, :, col]

            row += bs

    for mm in memmaps.values():
        mm.flush()


@click.command()
@click.option("--batch-size", default=256, help="Batch size for forward passes")
def main(batch_size):
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    print(f"Device: {device}", flush=True)

    weights_dir = (
        Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
    )
    ckpt = torch.load(
        weights_dir / "structured_classifier.pt",
        weights_only=False,
        map_location=device,
    )

    field_vocabs = ckpt["field_vocabs"]
    fnum_to_records = ckpt["gazetteer"]
    fnum_train_counts = ckpt["fnum_train_counts"]

    # Load examples
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = [
            ex
            for ex in all_examples
            if ex["records"]
            and ex["split"] == split
            and (
                ex["f_num"] != -100
                or ex.get("reason_missing_fnum") in NULL_TARGET_REASONS
            )
        ]

    for split in ("train", "val", "test"):
        null_count = sum(1 for ex in splits[split] if ex["f_num"] == -100)
        print(
            f"  {split}: {len(splits[split])} ({null_count} null targets)",
            flush=True,
        )

    # ── Filter gazetteer to f_nums with training examples ──
    full_size = sum(len(recs) for recs in fnum_to_records.values())
    fnum_to_records = {
        fnum: recs
        for fnum, recs in fnum_to_records.items()
        if fnum_train_counts.get(str(fnum), 0) > 0
    }
    filtered_size = sum(len(recs) for recs in fnum_to_records.values())
    print(
        f"\nFiltered gazetteer: {filtered_size}/{full_size} records "
        f"({len(fnum_to_records)} f_nums with examples)",
        flush=True,
    )

    field_indices, field_known, record_fnums_list, records_list = (
        build_gazetteer_matrix(fnum_to_records, field_vocabs)
    )
    n_records = len(record_fnums_list)
    print(f"Gazetteer: {n_records} records", flush=True)

    field_indices_dev = {f: t.to(device) for f, t in field_indices.items()}
    field_known_dev = {f: t.to(device) for f, t in field_known.items()}

    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    # ── Load temperatures ──
    with open(DATA_DIR / "temperatures.json") as f:
        temperatures = json.load(f)
    print(f"Temperatures: {temperatures}", flush=True)

    # ── Save metadata for train_scoring_layer.py ──
    metadata = {
        "n_records": n_records,
        "record_fnums": record_fnums_list,
        "fnum_to_records": fnum_to_records,
        "fnum_train_counts": fnum_train_counts,
        "split_sizes": {s: len(splits[s]) for s in ("train", "val", "test")},
    }
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {FEATURES_DIR / 'metadata.json'}", flush=True)

    # ── Precompute to disk ──
    classifier = load_classifier(ckpt, device)
    for split_name in ("train", "val", "test"):
        print(
            f"\nPrecomputing {split_name} ({len(splits[split_name])} examples)...",
            flush=True,
        )
        precompute_to_memmaps(
            model=classifier,
            examples=splits[split_name],
            split_name=split_name,
            n_records=n_records,
            field_vocabs=field_vocabs,
            field_indices=field_indices_dev,
            field_known=field_known_dev,
            pointer_val_to_indices=pointer_val_to_indices,
            pointer_none_indices=pointer_none_indices,
            temperatures=temperatures,
            batch_size=batch_size,
            device=device,
        )
    del classifier

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
