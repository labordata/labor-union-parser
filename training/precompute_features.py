#!/usr/bin/env python3
"""Precompute per-field log-prob features for the scoring layer.

Loads the bundled structured classifier and temperatures, runs forward
passes on train/val/test splits, and writes per-field numpy memmaps
to training/data/precomputed_features/{train,val,test}/.

Also saves metadata (record_fnums, field_known, fnum_to_records mapping,
split sizes) that train_scoring_layer.py needs.
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
FEATURES_DIR = DATA_DIR / "precomputed_features"

CLASSIFICATION_FIELDS = ["union_name", "desig_name", "f_num"]
POINTER_FIELD_LIST = ["desig_num", "prefix", "suffix"]
ALL_LP_FIELDS = CLASSIFICATION_FIELDS + POINTER_FIELD_LIST

PRECOMPUTE_CHUNK = 2048

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


def precompute_per_field(
    model,
    examples,
    field_vocabs,
    field_indices,
    field_known,
    pointer_val_to_indices,
    pointer_none_indices,
    temperatures,
    n_records,
    batch_size,
    device,
):
    ds = StructuredDataset(examples, field_vocabs)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    query_token_strings = []
    for ex in examples:
        tokens = smart_truncate_nonspace(ex["query"])
        query_token_strings.append([t["token"] for t in tokens])

    non_fnum_fields = [f for f in FIELDS if f != "f_num"]
    all_field_scores = {f: [] for f in non_fnum_fields}
    all_fnum_lp = []

    example_idx = 0
    with torch.no_grad():
        for inputs, _ in loader:
            char_ids = inputs["char_ids"].to(device)
            mask = inputs["mask"].to(device)
            logits = model(char_ids, mask)

            log_probs = {
                f: F.log_softmax(logits[f] / temperatures[f], dim=-1) for f in FIELDS
            }

            bs = char_ids.shape[0]
            for i in range(bs):
                for f in non_fnum_fields:
                    if f not in POINTER_FIELDS:
                        field_lp = log_probs[f][i][field_indices[f]]
                        vocab_size = log_probs[f].shape[-1]
                        floor_lp = -math.log(vocab_size)
                        field_lp = torch.where(field_known[f], field_lp, floor_lp)
                        all_field_scores[f].append(field_lp.cpu().numpy())
                    else:
                        query_toks = query_token_strings[example_idx]
                        lp = log_probs[f][i]
                        tok_to_pos = {}
                        for pos, tok in enumerate(query_toks):
                            if tok and tok not in tok_to_pos:
                                tok_to_pos[tok] = pos
                        field_scores = torch.full(
                            (n_records,), POINTER_NOT_FOUND_LOG_PROB[f], device=device
                        )
                        none_idx = pointer_none_indices[f]
                        if len(none_idx) > 0:
                            field_scores[none_idx] = lp[MAX_TOKENS]
                        val_to_idx = pointer_val_to_indices[f]
                        for tok, pos in tok_to_pos.items():
                            rec_indices = val_to_idx.get(tok)
                            if rec_indices is not None:
                                field_scores[rec_indices] = lp[pos]
                        all_field_scores[f].append(field_scores.cpu().numpy())

                fnum_lp = log_probs["f_num"][i][field_indices["f_num"]]
                fnum_vocab_size = log_probs["f_num"].shape[-1]
                fnum_floor = -math.log(fnum_vocab_size)
                fnum_lp = torch.where(field_known["f_num"], fnum_lp, fnum_floor)
                all_fnum_lp.append(fnum_lp.cpu().numpy())

                example_idx += 1

            if example_idx % 1024 < batch_size:
                print(f"    {example_idx}/{len(examples)}", flush=True)

    field_arrays = {}
    for f in non_fnum_fields:
        field_arrays[f] = np.stack(all_field_scores[f])
    field_arrays["f_num"] = np.stack(all_fnum_lp)
    return field_arrays


def precompute_to_memmaps(
    model, examples, split_name, n_records, precompute_args, batch_size
):
    split_dir = FEATURES_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    n_queries = len(examples)

    memmaps = {}
    for f in ALL_LP_FIELDS:
        path = split_dir / f"{f}.npy"
        memmaps[f] = np.memmap(
            path, dtype=np.float32, mode="w+", shape=(n_queries, n_records)
        )

    target_fnums = np.array(
        [
            int(ex["records"][0]["f_num"]) if ex["f_num"] != -100 else -1
            for ex in examples
        ]
    )
    np.save(split_dir / "target_fnums.npy", target_fnums)

    for chunk_start in range(0, n_queries, PRECOMPUTE_CHUNK):
        chunk_end = min(chunk_start + PRECOMPUTE_CHUNK, n_queries)
        chunk_arrays = precompute_per_field(
            model=model,
            examples=examples[chunk_start:chunk_end],
            n_records=n_records,
            batch_size=batch_size,
            **precompute_args,
        )
        for f in ALL_LP_FIELDS:
            memmaps[f][chunk_start:chunk_end] = chunk_arrays[f]
        del chunk_arrays
        print(f"  {chunk_end}/{n_queries}", flush=True)

    for mm in memmaps.values():
        mm.flush()
    print(f"  Saved to {split_dir}", flush=True)


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
    # Save field_known as numpy arrays
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in field_known:
        np.save(FEATURES_DIR / f"field_known_{f}.npy", field_known[f].numpy())
    with open(FEATURES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {FEATURES_DIR / 'metadata.json'}", flush=True)

    # ── Precompute to disk ──
    precompute_args = dict(
        field_vocabs=field_vocabs,
        field_indices=field_indices_dev,
        field_known=field_known_dev,
        pointer_val_to_indices=pointer_val_to_indices,
        pointer_none_indices=pointer_none_indices,
        temperatures=temperatures,
        device=device,
    )

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
            precompute_args=precompute_args,
            batch_size=batch_size,
        )
    del classifier

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
