#!/usr/bin/env python3
"""
Polish a trained structured classifier with hard-negative reranking loss.

Loads a checkpoint, mines hard negatives per epoch by scoring all training
queries against the full gazetteer, then trains with a softmax ranking loss
over {target, top-K negatives}. An auxiliary per-field CE loss prevents
catastrophic forgetting of field-level predictions.

Usage:
    python training/polish_with_reranking.py \
        --encoder token-charcnn --epochs 5 --lr 1e-5 --neg-k 15
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_factored_scoring import (
    POINTER_NOT_FOUND_LOG_PROB,
    build_gazetteer_matrix,
    build_pointer_lookup,
    get_model_input,
)
from train_structured_classifier import (
    COLLATE_FNS,
    DEVICE,
    FIELDS,
    MAX_TOKENS,
    POINTER_FIELDS,
    StructuredClassifier,
    StructuredDataset,
    _get_field_value,
    build_field_vocabs,
    model_path,
    smart_truncate_nonspace,
)

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
GAZETTEER_PATH = DATA_DIR / "fnum_to_records.json"


def build_query_token_strings(examples, field_vocabs, encoder):
    """Build token string lists for each valid example (for pointer scoring)."""
    token_strings = []
    for ex in examples:
        if not ex["records"]:
            continue
        rec = ex["records"][0]
        skip = False
        for f in FIELDS:
            if f in POINTER_FIELDS:
                continue
            if field_vocabs[f].get(_get_field_value(rec, f)) is None:
                skip = True
                break
        if skip:
            continue
        if POINTER_FIELDS and encoder != "char":
            tokens = smart_truncate_nonspace(ex["query"])
            token_strings.append([t["token"] for t in tokens])
        else:
            token_strings.append([])
    return token_strings


def build_target_fnums(examples, field_vocabs):
    """Build list of target f_nums aligned with dataset ordering."""
    target_fnums = []
    for ex in examples:
        if not ex["records"]:
            continue
        rec = ex["records"][0]
        skip = False
        for f in FIELDS:
            if f in POINTER_FIELDS:
                continue
            if field_vocabs[f].get(_get_field_value(rec, f)) is None:
                skip = True
                break
        if skip:
            continue
        target_fnums.append(rec["f_num"])
    return target_fnums


# Sentinel index for "not found" — we'll put a constant log-prob at this position
POINTER_NOT_FOUND_IDX = MAX_TOKENS + 1


def build_pointer_gather_base(pointer_val_to_indices, pointer_none_indices, n_records):
    """Precompute a base gather-index array per pointer field.

    For each record, precomputes:
    - base_idx: (n_records,) int array — POINTER_NOT_FOUND_IDX for records with values,
      MAX_TOKENS for NONE records
    - val_to_indices: {str: numpy array of record indices} — for fast CPU assignment

    Returns:
        {field: {
            'base_idx': numpy int array (n_records,),
            'val_to_indices': {str: numpy array of record indices},
        }}
    """
    result = {}
    for f in POINTER_FIELDS:
        base_idx = np.full(n_records, POINTER_NOT_FOUND_IDX, dtype=np.int64)
        if len(pointer_none_indices[f]) > 0:
            base_idx[pointer_none_indices[f].numpy()] = MAX_TOKENS
        val_to_np = {v: idx.numpy() for v, idx in pointer_val_to_indices[f].items()}
        result[f] = {
            "base_idx": base_idx,
            "val_to_indices": val_to_np,
        }
    return result


def score_queries_against_gazetteer_batched(
    log_probs_batch,
    token_strings_list,
    field_indices,
    pointer_gather_base,
    n_records,
):
    """Score a batch of queries against all gazetteer records — vectorized.

    Classification fields: fully vectorized via advanced indexing.
    Pointer fields: build gather indices on CPU (numpy), then gather on device.
    """
    B = log_probs_batch[FIELDS[0]].shape[0]
    all_scores = torch.zeros(B, n_records, device=DEVICE)

    # Classification fields: fully vectorized
    for f in FIELDS:
        if f not in POINTER_FIELDS:
            all_scores += log_probs_batch[f][:, field_indices[f]]

    # Pointer fields: build gather index on CPU, then single gather on device
    for f in POINTER_FIELDS:
        f_lp = log_probs_batch[f]  # (B, MAX_TOKENS + 1)
        fallback = POINTER_NOT_FOUND_LOG_PROB[f]
        pbase = pointer_gather_base[f]
        base_idx = pbase["base_idx"]  # (n_records,) numpy
        val_to_indices = pbase["val_to_indices"]

        # Extend log-probs with not-found slot: (B, MAX_TOKENS + 2)
        not_found_col = torch.full((B, 1), fallback, device=DEVICE)
        f_lp_ext = torch.cat([f_lp, not_found_col], dim=1)

        # Build gather indices on CPU: (B, n_records)
        gather_idx = np.tile(base_idx, (B, 1))  # (B, n_records) copy of base

        for i in range(B):
            query_toks = token_strings_list[i]
            tok_to_pos = {}
            for pos, tok in enumerate(query_toks):
                if tok and tok not in tok_to_pos:
                    tok_to_pos[tok] = pos

            for tok, pos in tok_to_pos.items():
                rec_indices = val_to_indices.get(tok)
                if rec_indices is not None:
                    gather_idx[i, rec_indices] = pos

        # Single transfer + gather on device
        gather_idx_t = torch.from_numpy(gather_idx).to(DEVICE)
        field_scores = torch.gather(f_lp_ext, 1, gather_idx_t)
        all_scores += field_scores

    return all_scores


def mine_hard_negatives(
    model,
    dataset,
    collate_fn,
    encoder,
    token_strings_all,
    target_record_indices,
    record_fnums_array,
    field_indices,
    pointer_gather_base,
    n_records,
    neg_k=15,
    batch_size=256,
):
    """Mine hard negatives for all training examples.

    Returns list of (example_idx, target_record_idx, [neg_record_idx, ...])
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Precompute fnum masks on device for efficient negative selection
    unique_fnums = np.unique(record_fnums_array)
    fnum_to_mask = {}
    for fn in unique_fnums:
        fnum_to_mask[fn] = torch.tensor(record_fnums_array == fn, device=DEVICE)

    triplets = []
    example_idx = 0
    import time

    t_forward = 0.0
    t_score = 0.0
    t_select = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, _labels) in enumerate(loader):
            t0 = time.time()
            model_input, mask = get_model_input(inputs, encoder, DEVICE)
            logits = model(model_input, mask)
            log_probs = {f: F.log_softmax(logits[f], dim=-1) for f in FIELDS}
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            t1 = time.time()
            t_forward += t1 - t0

            batch_size_actual = model_input.shape[0]
            batch_toks = token_strings_all[
                example_idx : example_idx + batch_size_actual
            ]

            scores = score_queries_against_gazetteer_batched(
                log_probs,
                batch_toks,
                field_indices,
                pointer_gather_base,
                n_records,
            )
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            t2 = time.time()
            t_score += t2 - t1

            # Move scores to CPU for selection
            scores_cpu = scores.cpu()

            for i in range(batch_size_actual):
                eidx = example_idx + i
                target_ridx = target_record_indices[eidx]
                if target_ridx < 0:
                    continue

                target_fnum = record_fnums_array[target_ridx]
                query_scores = scores_cpu[i].clone()
                same_fnum_mask_cpu = record_fnums_array == target_fnum
                query_scores[same_fnum_mask_cpu] = float("-inf")

                topk_vals, topk_idxs = query_scores.topk(min(neg_k, len(query_scores)))
                neg_indices = topk_idxs[topk_vals > float("-inf")].tolist()

                if neg_indices:
                    triplets.append((eidx, target_ridx, neg_indices))

            t3 = time.time()
            t_select += t3 - t2
            example_idx += batch_size_actual

            if batch_idx % 50 == 0 and batch_idx > 0:
                print(
                    f"    batch {batch_idx}: fwd={t_forward:.1f}s score={t_score:.1f}s sel={t_select:.1f}s"
                )

    print(
        f"  Timing: forward={t_forward:.1f}s scoring={t_score:.1f}s selection={t_select:.1f}s"
    )
    print(f"  Mined {len(triplets)} triplets from {example_idx} examples")
    return triplets


def build_record_pointer_values(pointer_val_to_indices, pointer_none_indices):
    """Build reverse mapping: record_idx -> normalized value per pointer field.

    None means NONE (absent). Missing key means not in any lookup (shouldn't happen).
    """
    result = {}
    for f in POINTER_FIELDS:
        rec_to_val = {}
        for val, rec_idxs in pointer_val_to_indices[f].items():
            for ridx in rec_idxs.tolist():
                rec_to_val[ridx] = val
        for ridx in pointer_none_indices[f].tolist():
            rec_to_val[ridx] = None
        result[f] = rec_to_val
    return result


def differentiable_score_candidates(
    log_probs,
    batch_token_strings,
    candidate_indices,
    field_indices,
    record_pointer_values,
):
    """Compute differentiable scores for candidate records per query.

    Args:
        log_probs: {field: (B, vocab_size)} — keeps grad
        batch_token_strings: list of token string lists, length B
        candidate_indices: (B, K) tensor of record indices
        field_indices: {field: (n_records,)} tensors
        record_pointer_values: {field: {record_idx: val_str_or_None}}

    Returns:
        scores: (B, K) tensor — differentiable
    """
    B, K = candidate_indices.shape
    scores = torch.zeros(B, K, device=DEVICE)

    for f in FIELDS:
        if f not in POINTER_FIELDS:
            cand_field_idx = field_indices[f][candidate_indices]  # (B, K)
            field_scores = torch.gather(log_probs[f], 1, cand_field_idx)
            scores = scores + field_scores
        else:
            f_lp = log_probs[f]  # (B, MAX_TOKENS + 1)
            rec_to_val = record_pointer_values[f]

            for b in range(B):
                query_toks = batch_token_strings[b]
                tok_to_pos = {}
                for pos, tok in enumerate(query_toks):
                    if tok and tok not in tok_to_pos:
                        tok_to_pos[tok] = pos

                for k in range(K):
                    ridx = candidate_indices[b, k].item()
                    val = rec_to_val.get(ridx)

                    if val is None:
                        scores[b, k] = scores[b, k] + f_lp[b, MAX_TOKENS]
                    else:
                        pos = tok_to_pos.get(val)
                        if pos is not None:
                            scores[b, k] = scores[b, k] + f_lp[b, pos]
                        else:
                            scores[b, k] = scores[b, k] + POINTER_NOT_FOUND_LOG_PROB[f]

    return scores


def eval_factored(
    model,
    loader,
    token_strings_all,
    target_fnums,
    split_name,
    field_indices,
    pointer_gather_base,
    record_fnums_array,
    n_records,
):
    """Run factored scoring evaluation. Returns number of errors."""
    model.eval()
    encoder = model.encoder_type

    correct = 0
    total = 0
    example_idx = 0

    with torch.no_grad():
        for inputs, _labels in loader:
            model_input, mask = get_model_input(inputs, encoder, DEVICE)
            logits = model(model_input, mask)
            log_probs = {f: F.log_softmax(logits[f], dim=-1) for f in FIELDS}

            batch_size_actual = model_input.shape[0]
            batch_toks = token_strings_all[
                example_idx : example_idx + batch_size_actual
            ]

            scores = score_queries_against_gazetteer_batched(
                log_probs,
                batch_toks,
                field_indices,
                pointer_gather_base,
                n_records,
            )

            for i in range(batch_size_actual):
                pred_fnum = record_fnums_array[scores[i].argmax().item()]
                if pred_fnum == target_fnums[example_idx + i]:
                    correct += 1
                total += 1

            example_idx += batch_size_actual

    errors = total - correct
    acc = correct / total if total else 0
    print(f"  {split_name}: {correct}/{total} = {acc:.4f} ({errors} errors)")
    return errors


@click.command()
@click.option(
    "--encoder",
    default="token-charcnn",
    type=click.Choice(["char", "token-charcnn", "token-embed"]),
)
@click.option("--epochs", default=5, help="Number of polishing epochs")
@click.option("--lr", default=1e-5, type=float, help="Learning rate")
@click.option("--neg-k", default=15, help="Number of hard negatives per query")
@click.option(
    "--aux-weight",
    default=0.1,
    type=float,
    help="Weight for auxiliary per-field CE loss",
)
@click.option("--batch-size", default=64, help="Training batch size")
@click.option("--mine-batch-size", default=256, help="Batch size for negative mining")
def main(encoder, epochs, lr, neg_k, aux_weight, batch_size, mine_batch_size):
    print(f"Device: {DEVICE}")
    print(f"Encoder: {encoder}")
    print(f"LR: {lr}, neg_k: {neg_k}, aux_weight: {aux_weight}")

    # Load data
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)
    with open(GAZETTEER_PATH) as f:
        fnum_to_records = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for ex in all_examples:
        splits[ex["split"]].append(ex)

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(model_path(encoder), weights_only=False, map_location=DEVICE)
    token_vocab = ckpt.get("token_vocab")

    # Restore pointer fields
    saved_pointer_fields = ckpt.get("pointer_fields", [])
    POINTER_FIELDS.clear()
    POINTER_FIELDS.update(saved_pointer_fields)
    if POINTER_FIELDS:
        print(f"Pointer fields: {sorted(POINTER_FIELDS)}")

    # Build vocabs
    field_vocabs = build_field_vocabs(splits["train"])

    # Build gazetteer
    field_indices, record_fnums, records_list = build_gazetteer_matrix(
        fnum_to_records, field_vocabs
    )
    record_fnums_array = np.array(record_fnums)
    n_records = len(record_fnums)
    print(f"Gazetteer: {n_records} records")

    # Pointer lookups
    pointer_val_to_indices = {}
    pointer_none_indices = {}
    for f in POINTER_FIELDS:
        pointer_val_to_indices[f], pointer_none_indices[f] = build_pointer_lookup(
            records_list, f
        )

    # Precomputed pointer gather base for fast batched scoring
    pointer_gather_base = build_pointer_gather_base(
        pointer_val_to_indices, pointer_none_indices, n_records
    )

    # Reverse pointer map for fast candidate scoring during training
    record_pointer_values = build_record_pointer_values(
        pointer_val_to_indices, pointer_none_indices
    )

    # Move field indices to device
    field_indices = {f: t.to(DEVICE) for f, t in field_indices.items()}

    # Datasets
    train_ds = StructuredDataset(splits["train"], field_vocabs, encoder, token_vocab)
    val_ds = StructuredDataset(splits["val"], field_vocabs, encoder, token_vocab)
    test_ds = StructuredDataset(splits["test"], field_vocabs, encoder, token_vocab)

    collate_fn = COLLATE_FNS[encoder]
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Token strings for pointer scoring
    train_token_strings = build_query_token_strings(
        splits["train"], field_vocabs, encoder
    )
    val_token_strings = build_query_token_strings(splits["val"], field_vocabs, encoder)
    test_token_strings = build_query_token_strings(
        splits["test"], field_vocabs, encoder
    )

    # Target f_nums for each split
    train_target_fnums = build_target_fnums(splits["train"], field_vocabs)
    val_target_fnums = build_target_fnums(splits["val"], field_vocabs)
    test_target_fnums = build_target_fnums(splits["test"], field_vocabs)

    # Map each training example to its target record index in the gazetteer
    target_record_indices = []
    for fnum in train_target_fnums:
        matches = np.where(record_fnums_array == fnum)[0]
        target_record_indices.append(matches[0] if len(matches) > 0 else -1)
    target_record_indices = np.array(target_record_indices)

    print(f"Training examples: {len(target_record_indices)}")
    missing = (target_record_indices == -1).sum()
    if missing:
        print(f"  ({missing} targets not in gazetteer, will skip)")

    # Load model with dropout for training
    field_sizes = ckpt["field_sizes"]
    model = StructuredClassifier(
        field_sizes=field_sizes,
        encoder=encoder,
        d_model=ckpt["d_model"],
        n_heads=4,
        n_layers=ckpt["n_layers"],
        ff_dim=ckpt["d_model"] * 2,
        dropout=0.1,
        token_vocab_size=len(token_vocab) if token_vocab else None,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Baseline eval
    print("\nBaseline (before polishing):")
    eval_factored(
        model,
        val_loader,
        val_token_strings,
        val_target_fnums,
        "val",
        field_indices,
        pointer_gather_base,
        record_fnums_array,
        n_records,
    )
    eval_factored(
        model,
        test_loader,
        test_token_strings,
        test_target_fnums,
        "test",
        field_indices,
        pointer_gather_base,
        record_fnums_array,
        n_records,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_errors = float("inf")

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        # Mine hard negatives
        print("Mining hard negatives...")
        triplets = mine_hard_negatives(
            model,
            train_ds,
            collate_fn,
            encoder,
            train_token_strings,
            target_record_indices,
            record_fnums_array,
            field_indices,
            pointer_gather_base,
            n_records,
            neg_k=neg_k,
            batch_size=mine_batch_size,
        )

        np.random.shuffle(triplets)

        # Train
        model.train()
        total_rank_loss = 0.0
        total_aux_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(triplets), batch_size):
            batch_triplets = triplets[batch_start : batch_start + batch_size]
            example_indices = [t[0] for t in batch_triplets]
            target_ridxs = [t[1] for t in batch_triplets]
            neg_ridxs_list = [t[2] for t in batch_triplets]

            # Pad negatives to same length (pad with target idx, masked out)
            max_negs = max(len(n) for n in neg_ridxs_list)
            candidate_indices = []
            for tidx, negs in zip(target_ridxs, neg_ridxs_list):
                row = [tidx] + negs + [tidx] * (max_negs - len(negs))
                candidate_indices.append(row)
            candidate_indices = torch.tensor(
                candidate_indices, dtype=torch.long, device=DEVICE
            )

            # Mask: True for target + actual negatives
            valid_mask = torch.zeros(
                len(batch_triplets), 1 + max_negs, dtype=torch.bool, device=DEVICE
            )
            for i, negs in enumerate(neg_ridxs_list):
                valid_mask[i, : 1 + len(negs)] = True

            # Collate training examples
            batch_items = [train_ds[idx] for idx in example_indices]
            inputs, labels = collate_fn(batch_items)
            model_input, mask = get_model_input(inputs, encoder, DEVICE)

            # Forward
            logits = model(model_input, mask)
            log_probs = {f: F.log_softmax(logits[f], dim=-1) for f in FIELDS}

            batch_tok_strings = [train_token_strings[idx] for idx in example_indices]

            # Differentiable scoring
            scores = differentiable_score_candidates(
                log_probs,
                batch_tok_strings,
                candidate_indices,
                field_indices,
                record_pointer_values,
            )

            # Mask padded candidates
            scores = scores.masked_fill(~valid_mask, float("-inf"))

            # Ranking loss: CE with target at index 0
            rank_targets = torch.zeros(
                len(batch_triplets), dtype=torch.long, device=DEVICE
            )
            rank_loss = F.cross_entropy(scores, rank_targets)

            # Auxiliary per-field CE loss
            aux_loss = torch.tensor(0.0, device=DEVICE)
            for f in FIELDS:
                y = labels[f].to(DEVICE)
                aux_loss = aux_loss + F.cross_entropy(logits[f], y)

            loss = rank_loss + aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_rank_loss += rank_loss.item()
            total_aux_loss += aux_loss.item()
            n_batches += 1

        avg_rank = total_rank_loss / max(n_batches, 1)
        avg_aux = total_aux_loss / max(n_batches, 1)
        print(f"  rank_loss={avg_rank:.4f}  aux_loss={avg_aux:.4f}")

        # Evaluate
        val_errors = eval_factored(
            model,
            val_loader,
            val_token_strings,
            val_target_fnums,
            "val",
            field_indices,
            pointer_gather_base,
            record_fnums_array,
            n_records,
        )
        eval_factored(
            model,
            test_loader,
            test_token_strings,
            test_target_fnums,
            "test",
            field_indices,
            pointer_gather_base,
            record_fnums_array,
            n_records,
        )

        if val_errors < best_val_errors:
            best_val_errors = val_errors
            save_dict = {
                "model_state": model.state_dict(),
                "field_vocabs": ckpt["field_vocabs"],
                "field_sizes": ckpt["field_sizes"],
                "d_model": ckpt["d_model"],
                "n_layers": ckpt["n_layers"],
                "encoder": encoder,
                "pointer_fields": sorted(POINTER_FIELDS),
            }
            if token_vocab is not None:
                save_dict["token_vocab"] = token_vocab
            save_path = model_path(encoder)
            torch.save(save_dict, save_path)
            print(f"  ** Saved (val_errors={val_errors})")

    print("\nDone!")


if __name__ == "__main__":
    main()
