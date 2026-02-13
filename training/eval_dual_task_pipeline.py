#!/usr/bin/env python3
"""
Pipeline evaluation for DualTaskModel: DT retrieval + cross-attention re-ranking.

Uses all unique records from training examples as the retrieval corpus.
For each test query:
  1. Retrieve top-k candidates via dual tower cosine similarity
  2. Re-rank candidates via cross-attention head
  3. Check if target f_num ranks #1
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import encode_query_batch, encode_record_batch
from mining import encode_all_records
from train_dual_task import DEVICE, EXAMPLES_PATH, FNUM_TO_RECORDS_PATH, VOCAB_PATH

from labor_union_parser.model import DualTaskModel


@click.command()
@click.option(
    "--checkpoint",
    default="training/dual_task_model.pt",
    help="Path to model checkpoint",
)
@click.option("--k", default=50, help="Number of candidates to retrieve")
def main(checkpoint, k):
    print(f"Using device: {DEVICE}")
    print(f"Retrieval k={k}")

    # Load vocab and examples
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    # Full gazetteer
    with open(FNUM_TO_RECORDS_PATH) as f:
        raw = json.load(f)
    fnum_to_records = {int(k): v for k, v in raw.items()}

    total_records = sum(len(recs) for recs in fnum_to_records.values())
    print(f"{len(fnum_to_records)} unique f_nums, {total_records} records")

    # Split
    test_ex = [ex for ex in all_examples if ex["split"] == "test"]
    print(f"Test examples: {len(test_ex)}")

    # Load model
    model = DualTaskModel(
        num_union_names=len(vocab["union_name_to_idx"]),
        num_desig_names=len(vocab["desig_name_to_idx"]),
        num_suffixes=len(vocab["suffix_to_idx"]),
        num_unit_ids=len(vocab["unit_id_to_idx"]),
    )
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    if "state_dict" in ckpt:
        state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    else:
        state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Encode all records
    print("Encoding all records...")
    all_record_embs, all_records, record_fnums = encode_all_records(
        model, fnum_to_records, vocab, batch_size=512
    )
    all_record_embs = all_record_embs.to(DEVICE)
    print(f"Encoded {len(all_records)} records")

    # Pre-encode all record fields for cross-attention (batch encode)
    print("Pre-encoding record fields for re-ranking...")
    all_field_embs = []
    all_field_masks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_records), 512), desc="Fields"):
            batch_recs = all_records[i : i + 512]
            rec_batch = encode_record_batch(batch_recs, vocab)
            rec_batch = {k_: v.to(DEVICE) for k_, v in rec_batch.items()}
            field_emb, field_mask = model.record_encoder(
                rec_batch["union_idx"],
                rec_batch["desig_idx"],
                rec_batch["prefix_hash"],
                rec_batch["num_hash"],
                rec_batch["num_val"],
                rec_batch["suffix_idx"],
                rec_batch["unit_id_idx"],
            )
            all_field_embs.append(field_emb)
            all_field_masks.append(field_mask)
    all_field_embs = torch.cat(all_field_embs, dim=0)  # [M, 6, embed_dim]
    all_field_masks = torch.cat(all_field_masks, dim=0)  # [M, 6]

    # Evaluate pipeline - collect per-candidate scores for ensemble sweep
    print(f"\nEvaluating pipeline on {len(test_ex)} test examples...")
    # Each entry: (query, target_fnum, {fnum: ret_score}, {fnum: rer_score})
    all_results = []
    dt_missed_queries = []
    query_batch_size = 64

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(test_ex), query_batch_size), desc="Eval"):
            batch_ex = test_ex[batch_start : batch_start + query_batch_size]
            batch_queries = [ex["query"] for ex in batch_ex]

            # Encode queries
            q_batch = encode_query_batch(batch_queries)
            q_batch = {k_: v.to(DEVICE) for k_, v in q_batch.items()}
            token_emb, padding_mask = model.query_encoder(
                q_batch["char_ids"], q_batch["is_number"], q_batch["numeric_ids"]
            )
            query_embs = model.dual_tower.encode_query(token_emb, padding_mask)

            # Retrieval: cosine similarity
            sims = torch.matmul(query_embs, all_record_embs.T)  # [B, M]
            topk_vals, topk_indices = sims.topk(k, dim=1)  # [B, k]

            # Per-query reranking
            for i, ex in enumerate(batch_ex):
                target_fnum = ex["f_num"]
                cand_indices = topk_indices[i]  # [k]
                cand_sims = topk_vals[i]  # [k]
                cand_fnums = record_fnums[cand_indices.cpu()]

                # Check if target in candidates
                if target_fnum not in cand_fnums.tolist():
                    dt_missed_queries.append((ex["query"], target_fnum))
                    continue

                # Re-rank: score each candidate with cross-attention
                cand_field_embs = all_field_embs[cand_indices]  # [k, 6, E]
                cand_field_masks = all_field_masks[cand_indices]  # [k, 6]

                # Expand query for all candidates
                q_tok = token_emb[i].unsqueeze(0).expand(len(cand_indices), -1, -1)
                q_mask = padding_mask[i].unsqueeze(0).expand(len(cand_indices), -1)

                rerank_scores = model.cross_attention.score_pair(
                    q_tok, q_mask, cand_field_embs, cand_field_masks
                )

                # Aggregate by f_num (max score per f_num, for both retrieval and rerank)
                fnum_ret = defaultdict(lambda: float("-inf"))
                fnum_rer = defaultdict(lambda: float("-inf"))
                for j, idx in enumerate(cand_indices.cpu().tolist()):
                    fn = record_fnums[idx].item()
                    fnum_ret[fn] = max(fnum_ret[fn], cand_sims[j].item())
                    fnum_rer[fn] = max(fnum_rer[fn], rerank_scores[j].item())

                all_results.append(
                    (ex["query"], target_fnum, dict(fnum_ret), dict(fnum_rer))
                )

    n = len(test_ex)
    dt_missed = len(dt_missed_queries)
    dt_retrieved = n - dt_missed

    print()
    print("=" * 70)
    print(
        f"DUAL TOWER RETRIEVAL (k={k}): {dt_retrieved}/{n} = {100*dt_retrieved/n:.2f}%"
    )
    print(f"Missed by DT: {dt_missed}")
    print("=" * 70)

    # Sweep ensemble weights: score = alpha * retrieval + (1 - alpha) * rerank
    # Need to normalize scores first since they're on different scales
    print("\nEnsemble sweep (alpha * retrieval + (1-alpha) * rerank):")
    print(f"{'alpha':>7}  {'correct':>7}  {'errors':>6}  {'acc':>7}")
    print("-" * 35)

    best_alpha = None
    best_correct = 0

    for alpha_pct in range(0, 101, 5):
        alpha = alpha_pct / 100.0
        correct = 0
        for _query, target_fnum, fnum_ret, fnum_rer in all_results:
            all_fnums = set(fnum_ret) | set(fnum_rer)

            # Normalize retrieval and rerank scores within each query
            ret_vals = [fnum_ret.get(fn, float("-inf")) for fn in all_fnums]
            rer_vals = [fnum_rer.get(fn, float("-inf")) for fn in all_fnums]

            ret_min, ret_max = min(ret_vals), max(ret_vals)
            rer_min, rer_max = min(rer_vals), max(rer_vals)

            ret_range = ret_max - ret_min if ret_max > ret_min else 1.0
            rer_range = rer_max - rer_min if rer_max > rer_min else 1.0

            best_fn = None
            best_score = float("-inf")
            for fn in all_fnums:
                r = (fnum_ret.get(fn, ret_min) - ret_min) / ret_range
                e = (fnum_rer.get(fn, rer_min) - rer_min) / rer_range
                combined = alpha * r + (1 - alpha) * e
                if combined > best_score:
                    best_score = combined
                    best_fn = fn

            if best_fn == target_fnum:
                correct += 1

        errors = dt_retrieved - correct
        acc = 100 * correct / n
        print(f"  {alpha:.2f}   {correct:>7}  {errors:>6}  {acc:>6.2f}%")

        if correct > best_correct:
            best_correct = correct
            best_alpha = alpha

    print("-" * 35)
    print(
        f"Best: alpha={best_alpha:.2f}, {best_correct}/{n} = {100*best_correct/n:.2f}%"
    )
    print(
        f"  Retrieval only (alpha=1.0):  {100*sum(1 for _q, t, fr, fe in all_results if max(fr, key=fr.get) == t)/n:.2f}%"
    )
    print(
        f"  Rerank only (alpha=0.0):     {100*sum(1 for _q, t, fr, fe in all_results if max(fe, key=fe.get) == t)/n:.2f}%"
    )

    # Save detailed errors (rerank-only, alpha=0)
    import csv

    errors = []
    for query, target_fnum, fnum_ret, fnum_rer in all_results:
        ranked = sorted(fnum_rer.items(), key=lambda x: x[1], reverse=True)
        pred_fnum = ranked[0][0]
        if pred_fnum != target_fnum:
            # Look up record details
            target_recs = fnum_to_records.get(target_fnum, [])
            pred_recs = fnum_to_records.get(pred_fnum, [])
            target_desc = ""
            pred_desc = ""
            if target_recs:
                r = target_recs[0]
                target_desc = f"{r.get('union_name','')} / {r.get('desig_name','')} {r.get('prefix','')}{r.get('desig_num','')} {r.get('suffix','')}"
            if pred_recs:
                r = pred_recs[0]
                pred_desc = f"{r.get('union_name','')} / {r.get('desig_name','')} {r.get('prefix','')}{r.get('desig_num','')} {r.get('suffix','')}"
            errors.append(
                {
                    "query": query,
                    "target_fnum": target_fnum,
                    "target_desc": target_desc.strip(),
                    "target_score": fnum_rer.get(target_fnum, float("-inf")),
                    "pred_fnum": pred_fnum,
                    "pred_desc": pred_desc.strip(),
                    "pred_score": ranked[0][1],
                    "score_gap": ranked[0][1]
                    - fnum_rer.get(target_fnum, float("-inf")),
                    "ret_rank_of_target": (
                        sorted(fnum_ret.values(), reverse=True).index(
                            fnum_ret.get(target_fnum, float("-inf"))
                        )
                        + 1
                        if target_fnum in fnum_ret
                        else -1
                    ),
                }
            )

    # Add DT misses
    for query, target_fnum in dt_missed_queries:
        target_recs = fnum_to_records.get(target_fnum, [])
        target_desc = ""
        if target_recs:
            r = target_recs[0]
            target_desc = f"{r.get('union_name','')} / {r.get('desig_name','')} {r.get('prefix','')}{r.get('desig_num','')} {r.get('suffix','')}"
        errors.append(
            {
                "query": query,
                "target_fnum": target_fnum,
                "target_desc": target_desc.strip(),
                "target_score": None,
                "pred_fnum": None,
                "pred_desc": "DT_MISS",
                "pred_score": None,
                "score_gap": None,
                "ret_rank_of_target": None,
            }
        )

    error_file = Path(__file__).parent / "data" / "pipeline_errors.csv"
    with open(error_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "target_fnum",
                "target_desc",
                "target_score",
                "pred_fnum",
                "pred_desc",
                "pred_score",
                "score_gap",
                "ret_rank_of_target",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)

    print(f"\n{len(errors)} errors saved to {error_file}")
    print(f"  Rerank errors: {len(errors) - dt_missed}")
    print(f"  DT misses: {dt_missed}")


if __name__ == "__main__":
    main()
