#!/usr/bin/env python3
"""Train a scoring layer on precomputed features.

Assumes precompute_features.py has already been run to produce
per-field log-prob memmaps in training/data/precomputed_features/.

Trains a ScoringLayer with LBFGS and cross-entropy loss over gazetteer
records whose f_num appears in some training example.
"""

import json
import math
from collections import Counter
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
FEATURES_DIR = DATA_DIR / "precomputed_features"

# Feature column names — must match precompute_features.py / compute_record_features layout.
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
N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ScoringLayer(nn.Module):
    """Linear scorer over 12 features + learnable null bias.

    score = w · [lp_fields, unk_indicators, nf_indicators] + bias

    Features: 6 log-probs, 3 unknown indicators, 3 not-found indicators.
    Learned: 12 weights + 1 bias + 1 null_bias = 14 parameters.

    The null_bias competes with real record scores in the softmax.
    When it wins, it signals "no match" for the query.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(N_FEATURES, 1)
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
        return self.linear(x)


def load_feature_memmaps(split_name, n_queries, n_records):
    """Load precomputed feature memmaps for a split.

    Returns:
        memmaps: dict of feature_name -> (n_queries, n_records) memmap
        target_fnums: (n_queries,) array of target f_nums (-1 for null)
    """
    split_dir = FEATURES_DIR / split_name
    memmaps = {}
    for name in FEATURE_NAMES:
        memmaps[name] = np.memmap(
            split_dir / f"{name}.npy",
            dtype=np.float32,
            mode="r",
            shape=(n_queries, n_records),
        )
    target_fnums = np.load(split_dir / "target_fnums.npy")
    return memmaps, target_fnums


def load_feature_chunk(memmaps, chunk_slice):
    """Load a chunk of features as a (chunk, R, 12) numpy array."""
    arrays = [np.array(memmaps[name][chunk_slice]) for name in FEATURE_NAMES]
    return np.stack(arrays, axis=2)


def eval_from_memmaps(
    scoring_model,
    split_memmaps,
    split_target_fnums,
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

            feat_np = load_feature_chunk(split_memmaps, chunk_slice)
            real_scores = scoring_model(torch.from_numpy(feat_np).to(device)).squeeze(
                -1
            )
            # Append null score column → (chunk, R+1)
            null_col = scoring_model.null_bias.expand(n_chunk, 1)
            scores = torch.cat([real_scores, null_col], dim=1)
            all_preds.append(scores.argmax(dim=1).cpu().numpy())

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

            feat_np = load_feature_chunk(split_memmaps, chunk_slice)
            real_scores = scoring_model(torch.from_numpy(feat_np).to(device)).squeeze(
                -1
            )
            # Append null score column → (chunk, R+1)
            null_col = scoring_model.null_bias.expand(n_chunk, 1).to(device)
            scores = torch.cat([real_scores, null_col], dim=1)
            all_scores.append(scores.cpu())

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

    optimizer = torch.optim.LBFGS(
        scoring.parameters(), lr=1.0, max_iter=20, history_size=20
    )
    call_count = [0]

    n_chunks = (n_train + chunk_size - 1) // chunk_size

    # Feature column indices for corruption
    # Layout: [lp_union=0, lp_desig=1, lp_fnum=2, ..., unk_union=6, unk_desig=7, unk_fnum=8, ...]
    LP_UNION, LP_DESIG, LP_FNUM = 0, 1, 2
    UNK_UNION, UNK_DESIG, UNK_FNUM = 6, 7, 8

    def closure():
        optimizer.zero_grad()
        total_loss = 0.0

        for chunk_idx, start in enumerate(range(0, n_train, chunk_size)):
            end = min(start + chunk_size, n_train)
            n_chunk = end - start
            chunk_slice = slice(start, end)

            feat_np = load_feature_chunk(train_memmaps, chunk_slice)

            # Apply corruption: zero lp columns, set unk indicators to 1
            feat_np[:, corruption_masks["fnum"], LP_FNUM] = 0.0
            feat_np[:, corruption_masks["fnum"], UNK_FNUM] = 1.0
            feat_np[:, corruption_masks["union"], LP_UNION] = 0.0
            feat_np[:, corruption_masks["union"], UNK_UNION] = 1.0
            feat_np[:, corruption_masks["desig"], LP_DESIG] = 0.0
            feat_np[:, corruption_masks["desig"], UNK_DESIG] = 1.0

            feat_t = torch.from_numpy(feat_np).float()
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
@click.option("--chunk-size", default=128, help="Queries per training chunk")
@click.option("--n-outer", default=5, help="Number of LBFGS outer steps")
@click.option(
    "--train-sample",
    default=15000,
    help="Subsample training queries for LBFGS (0 = use all)",
)
def main(chunk_size, n_outer, train_sample):
    weights_dir = (
        Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
    )

    # ── Load metadata from precomputed features ──
    with open(FEATURES_DIR / "metadata.json") as f:
        metadata = json.load(f)

    n_records = metadata["n_records"]
    record_fnums_array = np.array(metadata["record_fnums"])
    fnum_to_records = metadata["fnum_to_records"]
    fnum_train_counts = metadata["fnum_train_counts"]
    split_sizes = metadata["split_sizes"]
    n_train = split_sizes["train"]

    print(f"Gazetteer: {n_records} records", flush=True)

    # ── Derived arrays ──
    fnum_log_counts_np = np.zeros(n_records, dtype=np.float32)
    idx = 0
    for fnum, recs in fnum_to_records.items():
        count = fnum_train_counts.get(str(fnum), 0)
        lc = math.log1p(count)
        for _ in recs:
            fnum_log_counts_np[idx] = lc
            idx += 1

    # ── Load memmaps ──
    train_memmaps, train_target_fnums = load_feature_memmaps(
        "train", n_train, n_records
    )
    val_memmaps, val_target_fnums = load_feature_memmaps(
        "val", split_sizes["val"], n_records
    )
    test_memmaps, test_target_fnums = load_feature_memmaps(
        "test", split_sizes["test"], n_records
    )

    # ── Per-record corruption probabilities (e^{-k} based on training counts) ──
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    NULL_TARGET_REASONS = {"not in gazetteer", "unknown union"}
    train_examples = [
        ex
        for ex in all_examples
        if ex["records"]
        and ex["split"] == "train"
        and (
            ex["f_num"] != -100 or ex.get("reason_missing_fnum") in NULL_TARGET_REASONS
        )
    ]

    union_counts = Counter()
    desig_counts = Counter()
    fnum_to_union = {}
    fnum_to_desig = {}
    for fnum, recs in fnum_to_records.items():
        fnum_to_union[int(fnum)] = recs[0].get("union_name", "")
        fnum_to_desig[int(fnum)] = recs[0].get("desig_name", "")
    for ex in train_examples:
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
        sub_memmaps = {
            name: np.array(train_memmaps[name][sample_idx]) for name in FEATURE_NAMES
        }
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
    null_mask = rng.random(n_sub) < 0.15
    null_mask = null_mask | (sub_target_fnums == -1)
    print(f"Null mask: {int(null_mask.sum())}/{n_sub} queries", flush=True)

    # ── Train with LBFGS ──
    scoring = ScoringLayer()
    print("\nTraining scoring layer with LBFGS...", flush=True)
    train_scoring_lbfgs(
        scoring=scoring,
        train_memmaps=sub_memmaps,
        train_target_fnums=sub_target_fnums,
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
