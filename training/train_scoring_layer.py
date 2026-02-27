#!/usr/bin/env python3
"""Train a scoring layer on precomputed features.

Loads the frozen structured classifier, computes log-probs per field,
then trains a ScoringLayer with cross-entropy loss over gazetteer
records whose f_num appears in some training example (~17K records,
filtered from the full ~44K gazetteer).

Train on the train split with bootstrap corruption for f_num features.
Evaluate on val and test splits separately (full scoring).

Memory strategy:
  - Per-field log-prob arrays are written to numpy memmaps on disk.
  - Training reads memmaps on-the-fly in chunks, scoring all records.
  - Val/test per-field arrays are loaded from memmaps per minibatch.
"""

import json
import math
from collections import Counter
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
FEATURES_DIR = DATA_DIR / "precomputed_features"

CLASSIFICATION_FIELDS = ["union_name", "desig_name", "f_num"]
POINTER_FIELD_LIST = ["desig_num", "prefix", "suffix"]
ALL_LP_FIELDS = CLASSIFICATION_FIELDS + POINTER_FIELD_LIST

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

PRECOMPUTE_CHUNK = 2048

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Feature column layout:
# 0-5: log-probs (union_name, desig_name, f_num, desig_num, prefix, suffix)
# 6-8: unknown indicators (union_name, desig_name, f_num)
# 9-11: not-found indicators (desig_num, prefix, suffix)
# 12: log1p(fnum_count)

PENALTY_NAMES = [
    "unk_union",
    "unk_desig",
    "unk_fnum",
    "nf_designum",
    "nf_prefix",
    "nf_suffix",
]


N_SCORING_FEATURES = 12  # 6 lp + 3 unk + 3 nf (no count)
SCORING_IDX = list(range(12))  # first 12 of the 13-feature vector


class ScoringLayer(nn.Module):
    """Linear scorer over 12 features (no count) + learnable null bias.

    score = w · [lp_fields, unk_indicators, nf_indicators] + bias

    Features: 6 log-probs, 3 unknown indicators, 3 not-found indicators.
    Learned: 12 weights + 1 bias + 1 null_bias = 14 parameters.

    The null_bias competes with real record scores in the softmax.
    When it wins, it signals "no match" for the query.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(N_SCORING_FEATURES, 1)
        # Initialize from best blended-fnum run (k collapsed → linear)
        with torch.no_grad():
            self.linear.weight[0] = torch.tensor(
                [
                    0.6301,
                    0.4906,
                    0.8958,
                    1.1942,
                    0.5614,
                    0.7468,  # lp weights
                    -2.7790,
                    -1.3739,
                    -4.6739,  # unk penalties
                    -7.1982,
                    -0.7877,
                    -0.0123,  # nf penalties
                ]
            )
            self.linear.bias[0] = -21.1324
        # Null bias: initialized near the record bias so it's competitive
        self.null_bias = nn.Parameter(torch.tensor(self.linear.bias[0].item()))

    def forward(self, x):
        return self.linear(x[:, :, SCORING_IDX])


# ---------------------------------------------------------------------------
# Precompute helpers
# ---------------------------------------------------------------------------


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


def load_field_memmaps(split_name, n_queries, n_records):
    split_dir = FEATURES_DIR / split_name
    memmaps = {}
    for f in ALL_LP_FIELDS:
        memmaps[f] = np.memmap(
            split_dir / f"{f}.npy",
            dtype=np.float32,
            mode="r",
            shape=(n_queries, n_records),
        )
    target_fnums = np.load(split_dir / "target_fnums.npy")
    return memmaps, target_fnums


def features_exist(split_name):
    split_dir = FEATURES_DIR / split_name
    if not split_dir.exists():
        return False
    for f in ALL_LP_FIELDS:
        if not (split_dir / f"{f}.npy").exists():
            return False
    return (split_dir / "target_fnums.npy").exists()


# ---------------------------------------------------------------------------
# Feature processing & mining
# ---------------------------------------------------------------------------


def process_chunk_fields(field_arrays, field_known):
    for f in CLASSIFICATION_FIELDS:
        unknown_mask = np.logical_not(field_known[f].numpy())
        field_arrays[f] = np.where(unknown_mask, 0.0, field_arrays[f]).astype(
            np.float32
        )

    found_arrays = {}
    for f in POINTER_FIELD_LIST:
        not_found = np.float32(POINTER_NOT_FOUND_LOG_PROB[f])
        is_nf = field_arrays[f] == not_found
        found_arrays[f] = is_nf.astype(np.float32)
        field_arrays[f] = np.where(is_nf, 0.0, field_arrays[f]).astype(np.float32)
    return found_arrays


def assemble_features_from_fields(
    field_arrays,
    found_arrays,
    unknown_indicators_np,
    fnum_log_counts_np,
    query_indices,
):
    mb = len(query_indices)
    n_records = field_arrays[CLASSIFICATION_FIELDS[0]].shape[1]
    features = np.empty((mb, n_records, N_FEATURES), dtype=np.float32)

    for col, f in enumerate(ALL_LP_FIELDS):
        features[:, :, col] = field_arrays[f][query_indices]
    for col, f in enumerate(CLASSIFICATION_FIELDS):
        features[:, :, 6 + col] = unknown_indicators_np[f]
    for col, f in enumerate(POINTER_FIELD_LIST):
        features[:, :, 9 + col] = found_arrays[f][query_indices]
    features[:, :, 12] = fnum_log_counts_np
    return features


def eval_from_memmaps(
    scoring_model,
    split_memmaps,
    split_target_fnums,
    field_known,
    unknown_indicators_np,
    fnum_log_counts_np,
    record_fnums_array,
    device,
    chunk_size=128,
):
    scoring_model.eval()
    n_split = len(split_target_fnums)
    n_records = len(record_fnums_array)
    all_preds = []

    with torch.no_grad():
        for chunk_start in range(0, n_split, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_split)
            chunk_slice = slice(chunk_start, chunk_end)
            n_chunk = chunk_end - chunk_start

            chunk_fields = {
                f: np.array(split_memmaps[f][chunk_slice]) for f in ALL_LP_FIELDS
            }
            chunk_found = process_chunk_fields(chunk_fields, field_known)

            feat_np = assemble_features_from_fields(
                chunk_fields,
                chunk_found,
                unknown_indicators_np,
                fnum_log_counts_np,
                np.arange(n_chunk),
            )
            real_scores = scoring_model(torch.from_numpy(feat_np).to(device)).squeeze(
                -1
            )
            # Append null score column → (chunk, R+1)
            null_col = scoring_model.null_bias.expand(n_chunk, 1)
            scores = torch.cat([real_scores, null_col], dim=1)
            all_preds.append(scores.argmax(dim=1).cpu().numpy())
            del chunk_fields, chunk_found

    preds = np.concatenate(all_preds)
    pred_is_null = preds == n_records

    # Split into match queries (target_fnum >= 0) and null queries (target_fnum == -1)
    is_null_target = split_target_fnums == -1
    is_match_target = ~is_null_target

    # Match accuracy: among queries with a correct record
    n_match = int(is_match_target.sum())
    if n_match > 0:
        real_preds = np.minimum(preds, n_records - 1)
        pred_fnums = record_fnums_array[real_preds]
        match_correct = int(
            ((pred_fnums == split_target_fnums) & ~pred_is_null & is_match_target).sum()
        )
        match_errors = n_match - match_correct
    else:
        match_correct = 0
        match_errors = 0

    # Null accuracy: among queries that should be null
    n_null = int(is_null_target.sum())
    if n_null > 0:
        null_correct = int((pred_is_null & is_null_target).sum())
        null_errors = n_null - null_correct
    else:
        null_correct = 0
        null_errors = 0

    total_errors = match_errors + null_errors
    return {
        "errors": total_errors,
        "match_correct": match_correct,
        "n_match": n_match,
        "null_correct": null_correct,
        "n_null": n_null,
    }


def fit_scoring_temperature(
    scoring_model,
    split_memmaps,
    split_target_fnums,
    field_known,
    unknown_indicators_np,
    fnum_log_counts_np,
    record_fnums_array,
    device,
    chunk_size=128,
    lr=0.01,
    steps=200,
):
    """Fit a scalar temperature on scoring layer output via NLL on val set."""
    scoring_model.eval()
    n_split = len(split_target_fnums)

    # Build target indices: for each query, which record index is correct?
    # (pick highest-scoring correct record if multiple share the same fnum)
    all_scores = []
    with torch.no_grad():
        for chunk_start in range(0, n_split, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_split)
            chunk_slice = slice(chunk_start, chunk_end)
            n_chunk = chunk_end - chunk_start

            chunk_fields = {
                f: np.array(split_memmaps[f][chunk_slice]) for f in ALL_LP_FIELDS
            }
            chunk_found = process_chunk_fields(chunk_fields, field_known)

            feat_np = assemble_features_from_fields(
                chunk_fields,
                chunk_found,
                unknown_indicators_np,
                fnum_log_counts_np,
                np.arange(n_chunk),
            )
            real_scores = scoring_model(torch.from_numpy(feat_np).to(device)).squeeze(
                -1
            )
            # Append null score column → (chunk, R+1)
            null_col = scoring_model.null_bias.expand(n_chunk, 1).to(device)
            scores = torch.cat([real_scores, null_col], dim=1)
            all_scores.append(scores.cpu())
            del chunk_fields, chunk_found

    all_scores = torch.cat(all_scores, dim=0)  # (N, R+1)

    # Build correct mask: (N, R+1) — marginalize over all records with matching f_num
    n_records = len(record_fnums_array)
    record_fnums_t = torch.from_numpy(record_fnums_array)
    correct_mask = torch.zeros(n_split, n_records + 1, dtype=torch.bool)
    for i in range(n_split):
        fnum = split_target_fnums[i]
        if fnum == -1:
            correct_mask[i, n_records] = True  # null column
        else:
            correct_mask[i, :n_records] = record_fnums_t == fnum

    neg_inf = torch.finfo(all_scores.dtype).min

    def marginalized_nll(scores):
        correct_scores = scores.masked_fill(~correct_mask, neg_inf)
        return -(
            torch.logsumexp(correct_scores, dim=1) - torch.logsumexp(scores, dim=1)
        ).mean()

    # Fit temperature
    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_temp], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        temp = log_temp.exp()
        loss = marginalized_nll(all_scores / temp)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: T={temp.item():.4f}  NLL={loss.item():.4f}")

    final_temp = log_temp.exp().item()

    # Report NLL before/after
    nll_before = marginalized_nll(all_scores).item()
    nll_after = marginalized_nll(all_scores / final_temp).item()
    print(f"  NLL: {nll_before:.4f} -> {nll_after:.4f}")

    return final_temp


# ---------------------------------------------------------------------------
# LBFGS training
# ---------------------------------------------------------------------------


def train_scoring_lbfgs(
    scoring,
    train_memmaps,
    train_target_fnums,
    field_known,
    unknown_indicators_np,
    fnum_log_counts_np,
    record_fnums_array,
    corruption_masks,
    null_mask,
    n_outer=10,
    chunk_size=128,
):
    """Train scoring layer with LBFGS on full dataset.

    Args:
        corruption_masks: dict with 'fnum', 'union', 'desig' boolean arrays (n_records,)
        null_mask: boolean array (n_queries,) — True means train as null target
    """
    record_fnums_t = torch.from_numpy(record_fnums_array)
    n_train = len(train_target_fnums)
    neg_inf = float("-inf")

    # Apply null mask: copy target fnums and set masked entries to -1
    effective_targets = train_target_fnums.copy()
    effective_targets[null_mask] = -1
    target_fnums_t = torch.from_numpy(effective_targets)

    # Pre-build corrupted record-level arrays
    c_unk = {f: unknown_indicators_np[f].copy() for f in CLASSIFICATION_FIELDS}
    c_unk["f_num"][corruption_masks["fnum"]] = 1.0
    c_unk["union_name"][corruption_masks["union"]] = 1.0
    c_unk["desig_name"][corruption_masks["desig"]] = 1.0
    c_log_counts = fnum_log_counts_np.copy()
    c_log_counts[corruption_masks["fnum"]] = 0.0

    optimizer = torch.optim.LBFGS(
        scoring.parameters(), lr=1.0, max_iter=20, history_size=20
    )
    call_count = [0]

    n_chunks = (n_train + chunk_size - 1) // chunk_size

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0

        for chunk_idx, start in enumerate(range(0, n_train, chunk_size)):
            end = min(start + chunk_size, n_train)
            n_chunk = end - start
            chunk_slice = slice(start, end)

            chunk_fields = {
                f: np.array(train_memmaps[f][chunk_slice]) for f in ALL_LP_FIELDS
            }
            chunk_found = process_chunk_fields(chunk_fields, field_known)

            # Apply corruption to lp fields
            chunk_fields["f_num"][:, corruption_masks["fnum"]] = 0.0
            chunk_fields["union_name"][:, corruption_masks["union"]] = 0.0
            chunk_fields["desig_name"][:, corruption_masks["desig"]] = 0.0

            feat_np = assemble_features_from_fields(
                chunk_fields, chunk_found, c_unk, c_log_counts, np.arange(n_chunk)
            )
            feat_t = torch.from_numpy(feat_np[:, :, :N_SCORING_FEATURES]).float()
            real_scores = scoring(feat_t).squeeze(-1)
            null_col = scoring.null_bias.expand(n_chunk, 1)
            scores = torch.cat([real_scores, null_col], dim=1)

            chunk_targets = target_fnums_t[start:end]
            correct_mask = record_fnums_t.unsqueeze(0) == chunk_targets.unsqueeze(1)
            is_null = chunk_targets == -1
            null_correct = torch.zeros(n_chunk, 1, dtype=torch.bool)
            null_correct[is_null] = True
            full_correct = torch.cat([correct_mask, null_correct], dim=1)
            full_correct[is_null, :-1] = False

            correct_scores = scores.masked_fill(~full_correct, neg_inf)
            loss = -(
                torch.logsumexp(correct_scores, dim=1) - torch.logsumexp(scores, dim=1)
            ).sum()
            loss.backward()
            total_loss += loss.item()
            del feat_np, feat_t, real_scores, scores

            if (chunk_idx + 1) % 100 == 0:
                print(
                    f"    chunk {chunk_idx + 1}/{n_chunks}",
                    end="\r",
                    flush=True,
                )

        call_count[0] += 1
        print(
            f"  closure {call_count[0]:3d}: loss={total_loss/n_train:.4f}    ",
            flush=True,
        )
        return torch.tensor(total_loss / n_train)

    for outer in range(n_outer):
        print(f"  LBFGS outer step {outer}", flush=True)
        optimizer.step(closure)

    print(f"  Done ({call_count[0]} closure calls)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--batch-size", default=256, help="Batch size for precompute")
@click.option("--chunk-size", default=128, help="Queries per training chunk")
@click.option("--n-outer", default=10, help="Number of LBFGS outer steps")
@click.option(
    "--train-sample",
    default=15000,
    help="Subsample training queries for LBFGS (0 = use all)",
)
def main(batch_size, chunk_size, n_outer, train_sample):
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

    NULL_TARGET_REASONS = {"not in gazetteer", "unknown union"}

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
    n_train = len(splits["train"])

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
    record_fnums_array = np.array(record_fnums_list)
    print(f"Gazetteer: {n_records} records", flush=True)

    field_indices_dev = {f: t.to(device) for f, t in field_indices.items()}
    field_known_dev = {f: t.to(device) for f, t in field_known.items()}

    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    fnum_log_counts_np = np.zeros(n_records, dtype=np.float32)
    idx = 0
    for fnum, recs in fnum_to_records.items():
        count = fnum_train_counts.get(str(fnum), 0)
        lc = math.log1p(count)
        for _ in recs:
            fnum_log_counts_np[idx] = lc
            idx += 1

    unknown_indicators_np = {}
    for f in CLASSIFICATION_FIELDS:
        unknown_indicators_np[f] = np.logical_not(field_known[f].numpy()).astype(
            np.float32
        )

    # ── Load temperatures ──
    with open(DATA_DIR / "temperatures.json") as f:
        temperatures = json.load(f)
    print(f"Temperatures: {temperatures}", flush=True)

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

    classifier = None
    for split_name in ("train", "val", "test"):
        if features_exist(split_name):
            print(f"\n{split_name}: features exist, skipping", flush=True)
        else:
            if classifier is None:
                classifier = load_classifier(ckpt, device)
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

    # ── Load memmaps ──
    train_memmaps, train_target_fnums = load_field_memmaps("train", n_train, n_records)
    val_memmaps, val_target_fnums = load_field_memmaps(
        "val", len(splits["val"]), n_records
    )
    test_memmaps, test_target_fnums = load_field_memmaps(
        "test", len(splits["test"]), n_records
    )

    # ── Per-record corruption probabilities (e^{-k} based on training counts) ──
    # Each gazetteer record gets its own union/desig corruption prob
    union_counts = Counter()
    desig_counts = Counter()
    fnum_to_union = {}
    fnum_to_desig = {}
    for fnum, recs in fnum_to_records.items():
        fnum_to_union[int(fnum)] = recs[0].get("union_name", "")
        fnum_to_desig[int(fnum)] = recs[0].get("desig_name", "")
    for ex in splits["train"]:
        if ex["f_num"] == -100:
            continue
        fnum = ex["records"][0]["f_num"]
        union_counts[fnum_to_union.get(fnum, "")] += 1
        desig_counts[fnum_to_desig.get(fnum, "")] += 1

    record_union_corrupt_prob = np.zeros(n_records, dtype=np.float32)
    record_desig_corrupt_prob = np.zeros(n_records, dtype=np.float32)
    idx = 0
    for fnum, recs in fnum_to_records.items():
        u = fnum_to_union.get(int(fnum), "")
        d = fnum_to_desig.get(int(fnum), "")
        u_prob = math.exp(-union_counts[u])
        d_prob = math.exp(-desig_counts[d])
        for _ in recs:
            record_union_corrupt_prob[idx] = u_prob
            record_desig_corrupt_prob[idx] = d_prob
            idx += 1

    record_fnum_corrupt_prob = np.exp(-np.expm1(fnum_log_counts_np))
    print(
        f"Per-record corruption (prob > 0.1): "
        f"union={int((record_union_corrupt_prob > 0.1).sum())}/{n_records}, "
        f"desig={int((record_desig_corrupt_prob > 0.1).sum())}/{n_records}, "
        f"fnum={int((record_fnum_corrupt_prob > 0.1).sum())}/{n_records}",
        flush=True,
    )

    # ── Subsample training queries ──
    rng = np.random.RandomState(seed=42)
    if train_sample and train_sample < n_train:
        sample_idx = np.sort(rng.choice(n_train, size=train_sample, replace=False))
        sub_memmaps = {f: np.array(train_memmaps[f][sample_idx]) for f in ALL_LP_FIELDS}
        sub_target_fnums = train_target_fnums[sample_idx]
        n_sub = train_sample
        print(f"\nSubsampled {n_sub}/{n_train} training queries", flush=True)
    else:
        sub_memmaps = train_memmaps
        sub_target_fnums = train_target_fnums
        n_sub = n_train

    # ── Pre-bake corruption masks (deterministic) ──
    fnum_mask = rng.random(n_records) < record_fnum_corrupt_prob
    union_mask = fnum_mask & (rng.random(n_records) < record_union_corrupt_prob)
    desig_mask = fnum_mask & (rng.random(n_records) < record_desig_corrupt_prob)
    corruption_masks = {"fnum": fnum_mask, "union": union_mask, "desig": desig_mask}
    print(
        f"Corruption masks: fnum={int(fnum_mask.sum())}, "
        f"union={int(union_mask.sum())}, desig={int(desig_mask.sum())}",
        flush=True,
    )

    # ── Pre-bake null mask (deterministic) ──
    # For each training query with a real target, flip to null with prob 0.15
    null_mask = rng.random(n_sub) < 0.15
    # Queries already null (target == -1) stay null regardless
    null_mask = null_mask | (sub_target_fnums == -1)
    print(f"Null mask: {int(null_mask.sum())}/{n_sub} queries", flush=True)

    # ── Train with LBFGS ──
    scoring = ScoringLayer()
    print("\nTraining scoring layer with LBFGS...", flush=True)
    train_scoring_lbfgs(
        scoring=scoring,
        train_memmaps=sub_memmaps,
        train_target_fnums=sub_target_fnums,
        field_known=field_known,
        unknown_indicators_np=unknown_indicators_np,
        fnum_log_counts_np=fnum_log_counts_np,
        record_fnums_array=record_fnums_array,
        corruption_masks=corruption_masks,
        null_mask=null_mask,
        n_outer=n_outer,
        chunk_size=chunk_size,
    )

    # ── Save weights ──
    torch.save(scoring.state_dict(), weights_dir / "scoring_layer.pt")
    print(f"\nSaved scoring weights to {weights_dir / 'scoring_layer.pt'}")

    # ── Final report ──
    cpu = torch.device("cpu")
    for split_name, split_memmaps, split_targets in [
        ("val", val_memmaps, val_target_fnums),
        ("test", test_memmaps, test_target_fnums),
    ]:
        result = eval_from_memmaps(
            scoring,
            split_memmaps,
            split_targets,
            field_known,
            unknown_indicators_np,
            fnum_log_counts_np,
            record_fnums_array,
            cpu,
        )
        n_split = len(split_targets)
        errors = result["errors"]
        correct = n_split - errors
        print(
            f"  {split_name}: {correct}/{n_split} = {correct/n_split:.4f} "
            f"({errors} errors, match: {result['match_correct']}/{result['n_match']}, "
            f"null: {result['null_correct']}/{result['n_null']})"
        )

    # ── Fit scoring temperature on val set ──
    print("\nFitting scoring temperature on val set...")
    scoring_temp = fit_scoring_temperature(
        scoring,
        val_memmaps,
        val_target_fnums,
        field_known,
        unknown_indicators_np,
        fnum_log_counts_np,
        record_fnums_array,
        cpu,
    )
    print(f"Scoring temperature: {scoring_temp:.4f}")

    with open(DATA_DIR / "temperatures.json") as f:
        temps = json.load(f)
    temps["scoring"] = scoring_temp
    with open(DATA_DIR / "temperatures.json", "w") as f:
        json.dump(temps, f, indent=2)
    print(f"Saved scoring temperature to {DATA_DIR / 'temperatures.json'}")


if __name__ == "__main__":
    main()
