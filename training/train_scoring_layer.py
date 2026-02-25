#!/usr/bin/env python3
"""Train a scoring layer on precomputed features.

Loads the frozen structured classifier, computes log-probs per field,
then trains a GatedScoringLayer with cross-entropy loss over
hard-negative candidates mined with a linear baseline.

Train on the train split with bootstrap corruption for f_num features.
Evaluate on val and test splits separately (full 44K scoring).

Memory strategy:
  - Per-field log-prob arrays are written to numpy memmaps on disk
    (~17.6 GB per field × 6 fields = ~106 GB for train).
  - Hard negatives are mined from the memmaps in chunks, producing
    a small (N_train, K, 13) candidate array that fits in GPU memory.
  - Val/test per-field arrays are loaded from memmaps per minibatch.
"""

import json
import math
from collections import Counter
from pathlib import Path

import click
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

INIT_WEIGHTS = torch.tensor(
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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Feature column layout:
# 0-5: log-probs (union_name, desig_name, f_num, desig_num, prefix, suffix)
# 6-8: unknown indicators (union_name, desig_name, f_num)
# 9-11: not-found indicators (desig_num, prefix, suffix)
# 12: log1p(fnum_count)
LP_INDICES = [0, 1, 3, 4, 5]  # base log-probs (excl f_num)
PENALTY_INDICES = [6, 7, 8, 9, 10, 11]  # unknown/not-found indicators

PENALTY_NAMES = [
    "unk_union",
    "unk_desig",
    "unk_fnum",
    "nf_designum",
    "nf_prefix",
    "nf_suffix",
]


class GatedScoringLayer(nn.Module):
    """Additive scorer with count-dependent f_num weighting.

    alpha = ceil * count / (count + k)       where k = exp(log_k)

    base = lp_union + lp_desig + lp_designum + lp_prefix + lp_suffix
         + w_pen[0]·unk_union + w_pen[1]·unk_desig + w_pen[2]·unk_fnum
         + w_pen[3]·nf_designum + w_pen[4]·nf_prefix + w_pen[5]·nf_suffix + bias

    score = base + alpha * lp_fnum

    9 parameters total (6 penalties + 1 bias + 2 gate).
    """

    def __init__(self, init_weights=None):
        super().__init__()
        self.lp_idx = LP_INDICES
        self.penalty_idx = PENALTY_INDICES
        self.w_penalty = nn.Linear(len(self.penalty_idx), 1, bias=True)
        # Gate parameters: ceil and log_k (k = exp(log_k) to keep positive)
        self.gate_ceil = nn.Parameter(torch.tensor(0.6))
        self.gate_log_k = nn.Parameter(torch.tensor(math.log(16.0)))
        if init_weights is not None:
            with torch.no_grad():
                self.w_penalty.weight[0] = init_weights[self.penalty_idx].clone()
                self.w_penalty.bias.zero_()

    def forward(self, x):
        # x: (batch, n_records, 13)
        lp_fnum = x[:, :, 2:3]  # (batch, n_records, 1)
        count = torch.expm1(x[:, :, 12:13])  # log1p(count) -> count

        k = torch.exp(self.gate_log_k)
        alpha = self.gate_ceil * count / (count + k)

        lp_sum = x[:, :, self.lp_idx].sum(dim=-1, keepdim=True)
        penalties = self.w_penalty(x[:, :, self.penalty_idx])
        base = lp_sum + penalties  # (batch, n_records, 1)

        return base + alpha * lp_fnum

    def weight_str(self):
        wp = self.w_penalty.weight.data[0].cpu().numpy()
        bb = self.w_penalty.bias.data[0].item()
        ceil = self.gate_ceil.item()
        k = math.exp(self.gate_log_k.item())
        p_parts = " ".join(f"{n}={wp[i]:.2f}" for i, n in enumerate(PENALTY_NAMES))
        return f"G:ceil={ceil:.2f} k={k:.1f}  P:{p_parts} bias={bb:.2f}"


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

    target_fnums = np.array([ex["records"][0]["f_num"] for ex in examples])
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


def generate_corruption_masks(
    n_records, fnum_log_counts_np, record_union_corrupt_prob, record_desig_corrupt_prob
):
    """Generate per-record corruption masks for one mining pass.

    Returns corrupted copies of unknown_indicators and fnum_log_counts,
    plus boolean masks for zeroing lp fields during mining.
    """
    fnum_counts = np.expm1(fnum_log_counts_np)
    fnum_mask = np.random.random(n_records) < np.exp(-fnum_counts)
    union_mask = fnum_mask & (np.random.random(n_records) < record_union_corrupt_prob)
    desig_mask = fnum_mask & (np.random.random(n_records) < record_desig_corrupt_prob)
    return fnum_mask, union_mask, desig_mask


def _mine_topk(
    scores,
    target_fnums,
    record_fnums_array,
    k,
    chunk_fields,
    chunk_found,
    unknown_indicators_np,
    fnum_log_counts_np,
):
    """Given scores (n_chunk, n_records), mine top-K candidates per query."""
    n_chunk = scores.shape[0]

    # Vectorized top-K selection
    topk_idx = np.argpartition(scores, -k, axis=1)[:, -k:]  # (n_chunk, k)

    # Force-include correct records where missing (rare — typically <0.1%)
    topk_fnums = record_fnums_array[topk_idx]  # (n_chunk, k)
    has_correct = (topk_fnums == target_fnums[:, None]).any(axis=1)  # (n_chunk,)
    n_force = 0
    for qi in np.where(~has_correct)[0]:
        n_force += 1
        correct_indices = np.where(record_fnums_array == target_fnums[qi])[0]
        best_correct = correct_indices[scores[qi, correct_indices].argmax()]
        worst_pos = np.take(scores[qi], topk_idx[qi]).argmin()
        topk_idx[qi, worst_pos] = best_correct

    topk_idx.sort(axis=1)

    # Vectorized feature extraction
    candidate_features = np.empty((n_chunk, k, N_FEATURES), dtype=np.float32)
    for col, f in enumerate(ALL_LP_FIELDS):
        candidate_features[:, :, col] = np.take_along_axis(
            chunk_fields[f], topk_idx, axis=1
        )
    for col, f in enumerate(CLASSIFICATION_FIELDS):
        candidate_features[:, :, 6 + col] = unknown_indicators_np[f][topk_idx]
    for col, f in enumerate(POINTER_FIELD_LIST):
        candidate_features[:, :, 9 + col] = np.take_along_axis(
            chunk_found[f], topk_idx, axis=1
        )
    candidate_features[:, :, 12] = fnum_log_counts_np[topk_idx]

    # Vectorized target position finding
    topk_fnums = record_fnums_array[topk_idx]  # refresh after force-include
    correct_mask = topk_fnums == target_fnums[:, None]  # (n_chunk, k)
    # For queries with multiple correct records, pick highest-scoring
    correct_scores = np.where(
        correct_mask, np.take_along_axis(scores, topk_idx, axis=1), -np.inf
    )
    candidate_targets = correct_scores.argmax(axis=1).astype(np.int64)

    return candidate_features, candidate_targets, n_force


def mine_linear(
    field_memmaps,
    field_known,
    unknown_indicators_np,
    fnum_log_counts_np,
    record_union_corrupt_prob,
    record_desig_corrupt_prob,
    init_weights_np,
    target_fnums,
    record_fnums_array,
    k,
    chunk_size=2048,
):
    n_queries = len(target_fnums)
    n_records = field_memmaps[CLASSIFICATION_FIELDS[0]].shape[1]
    w = init_weights_np

    # Generate one set of per-record corruption masks for this mining pass
    fnum_mask, union_mask, desig_mask = generate_corruption_masks(
        n_records,
        fnum_log_counts_np,
        record_union_corrupt_prob,
        record_desig_corrupt_prob,
    )

    # Create corrupted copies of record-level arrays
    c_unk = {f: unknown_indicators_np[f].copy() for f in CLASSIFICATION_FIELDS}
    c_unk["f_num"][fnum_mask] = 1.0
    c_unk["union_name"][union_mask] = 1.0
    c_unk["desig_name"][desig_mask] = 1.0
    c_log_counts = fnum_log_counts_np.copy()
    c_log_counts[fnum_mask] = 0.0

    base_score = np.zeros(n_records, dtype=np.float32)
    for i, f in enumerate(CLASSIFICATION_FIELDS):
        base_score += w[6 + i] * c_unk[f]
    base_score += w[12] * c_log_counts

    all_feat, all_tgt = [], []
    total_force = 0

    for chunk_start in range(0, n_queries, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_queries)
        chunk_slice = slice(chunk_start, chunk_end)
        n_chunk = chunk_end - chunk_start

        chunk_fields = {
            f: np.array(field_memmaps[f][chunk_slice]) for f in ALL_LP_FIELDS
        }
        chunk_found = process_chunk_fields(chunk_fields, field_known)

        # Apply corruption: zero lp for corrupted records
        chunk_fields["f_num"][:, fnum_mask] = 0.0
        chunk_fields["union_name"][:, union_mask] = 0.0
        chunk_fields["desig_name"][:, desig_mask] = 0.0

        scores = np.tile(base_score, (n_chunk, 1))
        for i, f in enumerate(ALL_LP_FIELDS):
            scores += w[i] * chunk_fields[f]
        for i, f in enumerate(POINTER_FIELD_LIST):
            scores += w[9 + i] * chunk_found[f]

        cf, ct, nf = _mine_topk(
            scores,
            target_fnums[chunk_slice],
            record_fnums_array,
            k,
            chunk_fields,
            chunk_found,
            c_unk,
            c_log_counts,
        )
        all_feat.append(cf)
        all_tgt.append(ct)
        total_force += nf

        del chunk_fields, chunk_found, scores
        if chunk_end % (chunk_size * 4) < chunk_size:
            print(f"    {chunk_end}/{n_queries}", flush=True)

    n_corrupted = int(fnum_mask.sum())
    print(
        f"  Force-included: {total_force}/{n_queries}  "
        f"Corrupted: {n_corrupted}/{n_records} fnum, "
        f"{int(union_mask.sum())} union, {int(desig_mask.sum())} desig",
        flush=True,
    )
    return np.concatenate(all_feat), np.concatenate(all_tgt)


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
            scores = scoring_model(torch.from_numpy(feat_np).to(device)).squeeze(-1)
            all_preds.append(scores.argmax(dim=1).cpu().numpy())
            del chunk_fields, chunk_found

    preds = np.concatenate(all_preds)
    pred_fnums = record_fnums_array[preds]
    correct = (pred_fnums == split_target_fnums).sum()
    return n_split - correct


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
            scores = scoring_model(torch.from_numpy(feat_np).to(device)).squeeze(-1)
            all_scores.append(scores.cpu())
            del chunk_fields, chunk_found

    all_scores = torch.cat(all_scores, dim=0)  # (N, R)

    # Build target: index of correct record for each query
    targets = []
    for i in range(n_split):
        fnum = split_target_fnums[i]
        correct_mask = record_fnums_array == fnum
        correct_indices = np.where(correct_mask)[0]
        best = correct_indices[all_scores[i, correct_indices].argmax().item()]
        targets.append(best)
    targets = torch.tensor(targets, dtype=torch.long)

    # Fit temperature
    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_temp], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        temp = log_temp.exp()
        loss = F.cross_entropy(all_scores / temp, targets)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: T={temp.item():.4f}  NLL={loss.item():.4f}")

    final_temp = log_temp.exp().item()

    # Report NLL before/after
    nll_before = F.cross_entropy(all_scores, targets).item()
    nll_after = F.cross_entropy(all_scores / final_temp, targets).item()
    print(f"  NLL: {nll_before:.4f} -> {nll_after:.4f}")

    return final_temp


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class CandidateDataset(Dataset):
    """Wraps pre-corrupted candidate features + targets."""

    def __init__(self, cand_feat, cand_tgt):
        self.cand_feat = cand_feat  # (N, K, 13) numpy
        self.cand_tgt = cand_tgt  # (N,) numpy int64

    def __len__(self):
        return len(self.cand_tgt)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cand_feat[idx].copy()),
            torch.tensor(self.cand_tgt[idx], dtype=torch.long),
        )


class ScoringModule(L.LightningModule):
    def __init__(self, lr, weight_decay, eval_context=None):
        super().__init__()
        self.save_hyperparameters(ignore=["eval_context"])

        self.scoring = GatedScoringLayer(init_weights=INIT_WEIGHTS)
        self.eval_ctx = eval_context or {}

    def forward(self, x):
        return self.scoring(x)

    def training_step(self, batch, batch_idx):
        feat, targets = batch
        scores = self.scoring(feat).squeeze(-1)
        loss = F.cross_entropy(scores, targets)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_start(self):
        self._val_results = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Validation is done in on_validation_epoch_end via memmaps
        pass

    def on_validation_epoch_end(self):
        ctx = self.eval_ctx
        if not ctx:
            return

        for split_name in ("val", "test"):
            errors = eval_from_memmaps(
                self.scoring,
                ctx[f"{split_name}_memmaps"],
                ctx[f"{split_name}_target_fnums"],
                ctx["field_known"],
                ctx["unknown_indicators_np"],
                ctx["fnum_log_counts_np"],
                ctx["record_fnums_array"],
                self.device,
            )
            self.log(f"{split_name}_errors", float(errors), prog_bar=True)

        # Print weights for interpretability
        print(f"  {self.scoring.weight_str()}", flush=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--batch-size", default=256, help="Batch size for precompute")
@click.option("--lr", default=0.01)
@click.option("--epochs", default=5)
@click.option("--mb-size", default=32, help="Minibatch size for training")
@click.option("--weight-decay", default=0.01, help="Weight decay for AdamW")
@click.option("--topk", default=256, help="Top-K hard negatives per query")
def main(batch_size, lr, epochs, mb_size, weight_decay, topk):
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
            and ex.get("source") != "synthetic"
        ]

    n_train = len(splits["train"])
    print(
        f"\nTrain: {n_train}, Val: {len(splits['val'])}, Test: {len(splits['test'])}",
        flush=True,
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

    # ── Build module ──
    eval_context = dict(
        val_memmaps=val_memmaps,
        val_target_fnums=val_target_fnums,
        test_memmaps=test_memmaps,
        test_target_fnums=test_target_fnums,
        field_known=field_known,
        unknown_indicators_np=unknown_indicators_np,
        fnum_log_counts_np=fnum_log_counts_np,
        record_fnums_array=record_fnums_array,
    )

    module = ScoringModule(
        lr=lr,
        weight_decay=weight_decay,
        eval_context=eval_context,
    )

    # ── Mine hard negatives (linear baseline) ──
    print(f"\nMining top-{topk} hard negatives (linear baseline)...", flush=True)
    train_cand_feat, train_cand_tgt = mine_linear(
        train_memmaps,
        field_known,
        unknown_indicators_np,
        fnum_log_counts_np,
        record_union_corrupt_prob,
        record_desig_corrupt_prob,
        INIT_WEIGHTS.numpy(),
        train_target_fnums,
        record_fnums_array,
        k=topk,
    )

    # ── Build datasets ──
    train_ds = CandidateDataset(train_cand_feat, train_cand_tgt)
    train_loader = DataLoader(train_ds, batch_size=mb_size, shuffle=True, num_workers=0)

    # Dummy val loader (actual eval is done via memmaps in on_validation_epoch_end)
    val_loader = DataLoader([0], batch_size=1)

    # ── Callbacks ──
    checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=weights_dir,
        filename="scoring_layer",
        monitor="val_errors",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    callbacks = [checkpoint_cb]

    # ── Train ──
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=False,
        default_root_dir=str(DATA_DIR),
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ── Final report ──
    best_path = checkpoint_cb.best_model_path
    if best_path:
        best = ScoringModule.load_from_checkpoint(
            best_path,
            eval_context=eval_context,
        )
        print(f"\nBest checkpoint: {best_path}")
        for split_name in ("val", "test"):
            errors = eval_from_memmaps(
                best.scoring,
                eval_context[f"{split_name}_memmaps"],
                eval_context[f"{split_name}_target_fnums"],
                field_known,
                unknown_indicators_np,
                fnum_log_counts_np,
                record_fnums_array,
                device,
            )
            n_split = len(eval_context[f"{split_name}_target_fnums"])
            correct = n_split - errors
            print(
                f"  {split_name}: {correct}/{n_split} = {correct/n_split:.4f} ({errors} errors)"
            )

        # ── Fit scoring temperature on val set ──
        print("\nFitting scoring temperature on val set...")
        scoring_temp = fit_scoring_temperature(
            best.scoring,
            val_memmaps,
            val_target_fnums,
            field_known,
            unknown_indicators_np,
            fnum_log_counts_np,
            record_fnums_array,
            device,
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
