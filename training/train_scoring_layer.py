#!/usr/bin/env python3
"""Train a linear scoring layer on precomputed features.

Loads the frozen structured classifier, computes temperature-scaled
log-probs per field, then trains nn.Linear(N_features, 1) with
cross-entropy over all 44K gazetteer records.

Train on val split, eval on test split.
Features stored per-field as numpy arrays; assembled into GPU tensors
per-minibatch to avoid OOM.
"""

import json
import math
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
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

CLASSIFICATION_FIELDS = ["union_name", "desig_name", "f_num"]
POINTER_FIELD_LIST = ["desig_num", "prefix", "suffix"]

# Feature names (13 total)
FEATURE_NAMES = [
    "lp_union_name",
    "lp_desig_name",
    "lp_f_num",
    "lp_desig_num",
    "lp_prefix",
    "lp_suffix",
    "unk_union_name",
    "unk_desig_name",
    "unk_f_num",
    "notfound_desig_num",
    "notfound_prefix",
    "notfound_suffix",
    "log_fnum_count",
]
N_FEATURES = len(FEATURE_NAMES)


def load_model(ckpt):
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


def precompute_per_field(
    model,
    examples,
    field_vocabs,
    ckpt,
    field_indices,
    field_known,
    pointer_val_to_indices,
    pointer_none_indices,
    head_temperatures,
    fnum_class_temps,
    n_records,
    batch_size,
):
    """Precompute per-field score arrays: each (N_queries, N_records) float32 numpy.

    Also returns metadata arrays that are query-independent (broadcast).
    """
    ds = StructuredDataset(examples, field_vocabs)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    query_token_strings = []
    for ex in examples:
        tokens = smart_truncate_nonspace(ex["query"])
        query_token_strings.append([t["token"] for t in tokens])

    # Per-field score lists: each element is (N_records,) for one query
    non_fnum_fields = [f for f in FIELDS if f != "f_num"]
    all_field_scores = {f: [] for f in non_fnum_fields}
    all_fnum_lp = []

    example_idx = 0
    with torch.no_grad():
        for inputs, _ in loader:
            char_ids = inputs["char_ids"].to(DEVICE)
            mask = inputs["mask"].to(DEVICE)
            logits = model(char_ids, mask)

            # Apply temperatures
            scaled_logits = {}
            for f in FIELDS:
                if f == "f_num":
                    scaled_logits[f] = logits[f] / fnum_class_temps
                else:
                    scaled_logits[f] = logits[f] / head_temperatures[f]
            log_probs = {f: F.log_softmax(scaled_logits[f], dim=-1) for f in FIELDS}

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
                        all_field_scores[f].append(field_scores.cpu().numpy())

                # f_num
                fnum_lp = log_probs["f_num"][i][field_indices["f_num"]]
                fnum_vocab_size = log_probs["f_num"].shape[-1]
                fnum_floor = -math.log(fnum_vocab_size)
                fnum_lp = torch.where(field_known["f_num"], fnum_lp, fnum_floor)
                all_fnum_lp.append(fnum_lp.cpu().numpy())

                example_idx += 1

            if example_idx % 1024 < batch_size:
                print(f"  {example_idx}/{len(examples)}", flush=True)

    # Stack per-field: each (N_queries, N_records)
    field_arrays = {}
    for f in non_fnum_fields:
        field_arrays[f] = np.stack(all_field_scores[f])
    field_arrays["f_num"] = np.stack(all_fnum_lp)

    return field_arrays


@click.command()
@click.option("--batch-size", default=256)
@click.option("--lr", default=0.01)
@click.option("--epochs", default=200)
@click.option("--mb-size", default=32, help="Minibatch size (queries per step)")
@click.option("--hinge-margin", default=1.0, help="Pairwise hinge loss margin")
@click.option(
    "--reg-lambda", default=0.01, help="L2 regularization toward init weights"
)
def main(batch_size, lr, epochs, mb_size, hinge_margin, reg_lambda):
    print(f"Device: {DEVICE}", flush=True)

    # Load checkpoint
    weights_dir = (
        Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
    )
    ckpt = torch.load(
        weights_dir / "structured_classifier.pt",
        weights_only=False,
        map_location=DEVICE,
    )

    model = load_model(ckpt)
    field_vocabs = ckpt["field_vocabs"]
    fnum_to_records = ckpt["gazetteer"]

    # Load temperatures
    temps_path = DATA_DIR / "temperatures.json"
    with open(temps_path) as f:
        temps = json.load(f)
    head_temperatures = {f: temps[f] for f in FIELDS if f != "f_num" and f in temps}
    fnum_a, fnum_b = temps["f_num_a"], temps["f_num_b"]

    # Build per-class f_num temperature vector
    fnum_vocab = field_vocabs["f_num"]
    fnum_train_counts = ckpt["fnum_train_counts"]
    n_fnum_classes = len(fnum_vocab)
    fnum_class_temps = torch.ones(n_fnum_classes, device=DEVICE)
    for fnum_val, class_idx in fnum_vocab.items():
        count = fnum_train_counts.get(str(fnum_val), 0)
        fnum_class_temps[class_idx] = math.exp(fnum_a + fnum_b * math.log1p(count))

    # Build gazetteer structures
    field_indices, field_known, record_fnums_list, records_list = (
        build_gazetteer_matrix(fnum_to_records, field_vocabs)
    )
    n_records = len(record_fnums_list)
    record_fnums_array = np.array(record_fnums_list)
    print(f"Gazetteer: {n_records} records", flush=True)

    field_indices_dev = {f: t.to(DEVICE) for f, t in field_indices.items()}
    field_known_dev = {f: t.to(DEVICE) for f, t in field_known.items()}

    # Pointer lookups
    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    # Build fnum_log_counts (query-independent, one per record)
    fnum_log_counts_np = np.zeros(n_records, dtype=np.float32)
    idx = 0
    for fnum, recs in fnum_to_records.items():
        count = fnum_train_counts.get(str(fnum), 0)
        lc = math.log1p(count)
        for _ in recs:
            fnum_log_counts_np[idx] = lc
            idx += 1

    # Build known/found indicators (query-independent, one per record)
    unknown_indicators = {}
    for f in CLASSIFICATION_FIELDS:
        unknown_indicators[f] = np.logical_not(field_known[f].numpy()).astype(
            np.float32
        )

    # For pointer "found" — this is query-dependent, handled below

    # Load examples — combine val + test for cross-validation
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    examples = []
    for ex in all_examples:
        if ex["records"] and ex["split"] in ("val", "test"):
            examples.append(ex)

    n_total = len(examples)
    print(f"\nCombined val+test: {n_total} examples", flush=True)

    # Precompute per-field log-prob arrays for all examples
    print(f"\nPrecomputing features ({n_total} examples)...", flush=True)
    field_arrays = precompute_per_field(
        model,
        examples,
        field_vocabs,
        ckpt,
        field_indices_dev,
        field_known_dev,
        pointer_val_to_indices,
        pointer_none_indices,
        head_temperatures,
        fnum_class_temps,
        n_records,
        batch_size,
    )

    del model  # free GPU memory

    target_fnums = np.array([ex["records"][0]["f_num"] for ex in examples])

    # Build target masks: for each query, True for all records with the correct f_num.
    def build_target_masks(target_fnums_arr, n_queries):
        masks = np.zeros((n_queries, n_records), dtype=bool)
        for qi in range(n_queries):
            masks[qi] = record_fnums_array == target_fnums_arr[qi]
        return masks

    print("Building target masks...", flush=True)
    target_masks = build_target_masks(target_fnums, n_total)

    # Zero out log-probs for unknown/not-found records.
    # Indicators carry the fallback penalty instead.

    # Classification fields: zero where unknown
    # NOTE: avoid ~ operator — use np.logical_not or == False
    unknown_mask = {}
    for f in CLASSIFICATION_FIELDS:
        unknown_mask[f] = np.logical_not(field_known[f].numpy())
    for f in CLASSIFICATION_FIELDS:
        mask = unknown_mask[f]
        field_arrays[f] = np.where(mask, 0.0, field_arrays[f]).astype(np.float32)

    # Pointer fields: zero where not found
    # NOTE: avoid ~ operator on boolean arrays — it mutates in-place on this
    # numpy/Python version. Use == instead of != to get the not_found mask directly.
    found_arrays = {}
    for f in POINTER_FIELD_LIST:
        not_found = np.float32(POINTER_NOT_FOUND_LOG_PROB[f])
        is_nf = field_arrays[f] == not_found
        found_arrays[f] = is_nf.astype(np.float32)
        field_arrays[f] = np.where(is_nf, 0.0, field_arrays[f]).astype(np.float32)

    # Pre-convert query-independent metadata to tensors (small, shared)
    unknown_tensors = {
        f: torch.from_numpy(unknown_indicators[f]) for f in CLASSIFICATION_FIELDS
    }
    fnum_log_counts_t = torch.from_numpy(fnum_log_counts_np)

    def assemble_batch(query_indices):
        """Build (mb, N_records, 13) feature tensor on CPU for a minibatch."""
        mb = len(query_indices)
        features = torch.empty(mb, n_records, N_FEATURES)

        for col, f in enumerate(CLASSIFICATION_FIELDS + POINTER_FIELD_LIST):
            features[:, :, col] = torch.from_numpy(field_arrays[f][query_indices])

        for col, f in enumerate(CLASSIFICATION_FIELDS):
            features[:, :, 6 + col] = unknown_tensors[f]

        for col, f in enumerate(POINTER_FIELD_LIST):
            features[:, :, 9 + col] = torch.from_numpy(found_arrays[f][query_indices])

        features[:, :, 12] = fnum_log_counts_t

        return features

    def pairwise_hinge_loss(scores, target_mask, margin=1.0):
        """Pairwise hinge loss: penalize when best wrong record is within margin of best correct.

        For each query:
          s_correct = max score among records with correct fnum
          s_wrong = max score among records with wrong fnum
          loss = max(0, margin + s_wrong - s_correct)

        Only generates gradient when the model is wrong or nearly wrong.
        """
        # Best correct score per query
        correct_scores = scores.masked_fill(~target_mask, float("-inf"))
        s_correct = correct_scores.max(dim=-1).values  # (batch,)

        # Best wrong score per query
        wrong_scores = scores.masked_fill(target_mask, float("-inf"))
        s_wrong = wrong_scores.max(dim=-1).values  # (batch,)

        loss = torch.clamp(margin + s_wrong - s_correct, min=0.0)
        return loss.mean()

    init_weights = torch.tensor(
        [
            0.2757,
            0.0709,
            0.1694,
            0.2713,
            0.3470,
            0.3450,
            -1.5582,
            -0.7432,
            -0.8317,
            -3.1408,
            -0.7080,
            -0.5528,
            0.0809,
        ]
    )

    short_names = [
        "un",
        "dn",
        "fn",
        "dnum",
        "pre",
        "suf",
        "u_un",
        "u_dn",
        "u_fn",
        "nf_dnum",
        "nf_pre",
        "nf_suf",
        "lcnt",
    ]

    # Train on all data
    print(
        f"\nTraining linear layer ({N_FEATURES} -> 1) on val+test ({n_total} examples)",
        flush=True,
    )
    print(f"  lr={lr}, epochs={epochs}, mb_size={mb_size}\n", flush=True)

    scoring = nn.Linear(N_FEATURES, 1, bias=False)
    with torch.no_grad():
        scoring.weight[0] = init_weights.clone()

    # Evaluate init
    scoring.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, n_total, mb_size):
            end = min(start + mb_size, n_total)
            qi = np.arange(start, end)
            feat = assemble_batch(qi)
            scores = scoring(feat).squeeze(-1)
            all_preds.append(scores.argmax(dim=1).numpy())
    preds = np.concatenate(all_preds)
    pred_fnums = record_fnums_array[preds]
    init_correct = (pred_fnums == target_fnums).sum()
    init_errors = n_total - init_correct
    print(
        f"INIT: {init_correct}/{n_total} fnum_acc={init_correct/n_total:.4f} ({init_errors}err)",
        flush=True,
    )

    target_masks_t = torch.from_numpy(target_masks)
    optimizer = torch.optim.Adam(scoring.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        scoring.train()
        perm = np.random.permutation(n_total)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_total, mb_size):
            qi = perm[start : start + mb_size]
            feat = assemble_batch(qi)
            tgt_mask = target_masks_t[qi]

            scores = scoring(feat).squeeze(-1)
            hinge = pairwise_hinge_loss(scores, tgt_mask, margin=hinge_margin)
            reg = reg_lambda * ((scoring.weight[0] - init_weights) ** 2).sum()
            loss = hinge + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            scoring.eval()
            all_preds = []
            with torch.no_grad():
                for start in range(0, n_total, mb_size):
                    end = min(start + mb_size, n_total)
                    qi = np.arange(start, end)
                    feat = assemble_batch(qi)
                    scores = scoring(feat).squeeze(-1)
                    all_preds.append(scores.argmax(dim=1).numpy())
            preds = np.concatenate(all_preds)
            pred_fnums = record_fnums_array[preds]
            correct = (pred_fnums == target_fnums).sum()
            errors = n_total - correct

            w = scoring.weight.data[0].numpy()
            w_str = " ".join(f"{n}={w[i]:.2f}" for i, n in enumerate(short_names))
            print(
                f"E{epoch+1:3d}: loss={epoch_loss/n_batches:.4f}  "
                f"fnum_acc={correct}/{n_total}={correct/n_total:.4f} ({errors}err)  "
                f"{w_str}",
                flush=True,
            )

    # Final weights
    w = scoring.weight.data[0].numpy()
    print("\nLearned weights:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:20s}: {w[i]:.4f}")

    # Evaluate on test split only
    test_indices = np.array(
        [i for i, ex in enumerate(examples) if ex["split"] == "test"]
    )
    val_indices = np.array([i for i, ex in enumerate(examples) if ex["split"] == "val"])

    for split_name, split_idx in [("val", val_indices), ("test", test_indices)]:
        scoring.eval()
        n_split = len(split_idx)
        split_preds = []
        with torch.no_grad():
            for start in range(0, n_split, mb_size):
                end = min(start + mb_size, n_split)
                qi = split_idx[start:end]
                feat = assemble_batch(qi)
                scores = scoring(feat).squeeze(-1)
                split_preds.append(scores.argmax(dim=1).numpy())
        preds = np.concatenate(split_preds)
        pred_fnums = record_fnums_array[preds]
        split_targets = target_fnums[split_idx]
        correct = (pred_fnums == split_targets).sum()
        errors = n_split - correct
        print(
            f"\n{split_name}: {correct}/{n_split} = {correct/n_split:.4f} ({errors} errors)"
        )


if __name__ == "__main__":
    main()
