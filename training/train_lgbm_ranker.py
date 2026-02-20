#!/usr/bin/env python3
"""
LightGBM LambdaRank reranker.

Same features as train_gbm_reranker.py but uses LightGBM's lambdarank objective
which optimizes for ranking directly rather than pointwise classification.

Features and candidate pairs are cached to disk for fast iteration.
"""

import csv
import json
import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import encode_query_batch, encode_record_batch
from mining import encode_all_records
from train_dual_task import DEVICE, EXAMPLES_PATH, FNUM_TO_RECORDS_PATH, VOCAB_PATH
from train_gbm_reranker import (
    FEATURE_NAMES,
    _parse_query,
    build_trigram_idf,
    extract_features,
)

from labor_union_parser.model import DualTaskModel

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = Path(__file__).parent.parent / "opdr.db"
CACHE_PATH = DATA_DIR / "lgbm_ranker_cache.pkl"
UNION_CLF_PATH = DATA_DIR / "union_name_classifier.pkl"
DESIG_CLF_PATH = DATA_DIR / "desig_name_classifier.pkl"
PREFIX_CLF_PATH = DATA_DIR / "prefix_classifier.pkl"
SUFFIX_CLF_PATH = DATA_DIR / "suffix_classifier.pkl"
UNION_UNIT_CLF_PATH = DATA_DIR / "union_unit_classifier.pkl"


def build_features_and_candidates(
    examples,
    model,
    all_record_embs,
    all_records,
    record_fnums,
    fnum_to_unit_name,
    vocab,
    k,
    query_batch_size,
    desc="Features",
    add_structural=False,
    force_target=False,
    idf_union=None,
    idf_unit=None,
    union_clf=None,
    desig_clf=None,
    prefix_clf=None,
    suffix_clf=None,
    union_unit_clf=None,
):
    """
    Retrieve top-k candidates and extract features for each pair.

    Returns:
        X: feature matrix [n_pairs, n_features]
        y: relevance labels [n_pairs]
        groups: list of group sizes per query
        candidate_info: list of (query, target_fnum, cand_fnums, cand_sims) or None
    """
    all_feats = []
    all_labels = []
    groups = []
    candidate_info = []
    missed = 0

    # Build index from record fields -> index in all_records
    rec_key_to_idx = {}
    for ri, rec in enumerate(all_records):
        key = (
            rec["f_num"],
            rec["desig_name"],
            rec["desig_num"],
            rec["prefix"],
            rec["suffix"],
        )
        rec_key_to_idx[key] = ri

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(examples), query_batch_size), desc=desc):
            batch_ex = examples[batch_start : batch_start + query_batch_size]
            batch_queries = [ex["query"] for ex in batch_ex]

            q_batch = encode_query_batch(batch_queries)
            q_batch = {qk: v.to(DEVICE) for qk, v in q_batch.items()}
            token_emb, padding_mask = model.query_encoder(
                q_batch["char_ids"],
                q_batch["is_number"],
                q_batch["numeric_ids"],
            )
            query_embs = model.dual_tower.encode_query(token_emb, padding_mask)

            sims = torch.matmul(query_embs, all_record_embs.T)
            topk_vals, topk_indices = sims.topk(k, dim=1)

            # Compute per-query classifier probability dicts for this batch
            batch_union_probs = [None] * len(batch_ex)
            batch_desig_probs = [None] * len(batch_ex)
            batch_prefix_probs = [None] * len(batch_ex)
            batch_suffix_probs = [None] * len(batch_ex)
            batch_union_unit_probs = [None] * len(batch_ex)
            if union_clf is not None:
                union_vec = union_clf["vectorizer"]
                union_model = union_clf["classifier"]
                union_idx_to_name = union_clf["idx_to_name"]
                union_classes = union_model.classes_
                X_u = union_vec.transform(batch_queries)
                proba_u = union_model.predict_proba(X_u)
                for bi in range(len(batch_ex)):
                    batch_union_probs[bi] = {
                        union_idx_to_name[union_classes[ci]]: float(proba_u[bi, ci])
                        for ci in range(proba_u.shape[1])
                        if proba_u[bi, ci] > 1e-6
                    }
            if desig_clf is not None:
                desig_vec = desig_clf["vectorizer"]
                desig_model = desig_clf["classifier"]
                desig_idx_to_name = desig_clf["idx_to_name"]
                desig_classes = desig_model.classes_
                X_d = desig_vec.transform(batch_queries)
                proba_d = desig_model.predict_proba(X_d)
                for bi in range(len(batch_ex)):
                    batch_desig_probs[bi] = {
                        desig_idx_to_name[desig_classes[ci]]: float(proba_d[bi, ci])
                        for ci in range(proba_d.shape[1])
                        if proba_d[bi, ci] > 1e-6
                    }
            if prefix_clf is not None:
                pfx_vec = prefix_clf["vectorizer"]
                pfx_model = prefix_clf["classifier"]
                pfx_idx_to_name = prefix_clf["idx_to_name"]
                pfx_classes = pfx_model.classes_
                X_p = pfx_vec.transform(batch_queries)
                proba_p = pfx_model.predict_proba(X_p)
                for bi in range(len(batch_ex)):
                    batch_prefix_probs[bi] = {
                        pfx_idx_to_name[pfx_classes[ci]]: float(proba_p[bi, ci])
                        for ci in range(proba_p.shape[1])
                        if proba_p[bi, ci] > 1e-6
                    }
            if suffix_clf is not None:
                sfx_vec = suffix_clf["vectorizer"]
                sfx_model = suffix_clf["classifier"]
                sfx_idx_to_name = suffix_clf["idx_to_name"]
                sfx_classes = sfx_model.classes_
                X_s = sfx_vec.transform(batch_queries)
                proba_s = sfx_model.predict_proba(X_s)
                for bi in range(len(batch_ex)):
                    batch_suffix_probs[bi] = {
                        sfx_idx_to_name[sfx_classes[ci]]: float(proba_s[bi, ci])
                        for ci in range(proba_s.shape[1])
                        if proba_s[bi, ci] > 1e-6
                    }
            if union_unit_clf is not None:
                uu_vec = union_unit_clf["vectorizer"]
                uu_model = union_unit_clf["classifier"]
                uu_idx_to_name = union_unit_clf["idx_to_name"]
                uu_classes = uu_model.classes_
                X_uu = uu_vec.transform(batch_queries)
                proba_uu = uu_model.predict_proba(X_uu)
                for bi in range(len(batch_ex)):
                    batch_union_unit_probs[bi] = {
                        uu_idx_to_name[uu_classes[ci]]: float(proba_uu[bi, ci])
                        for ci in range(proba_uu.shape[1])
                        if proba_uu[bi, ci] > 1e-6
                    }

            for i, ex in enumerate(batch_ex):
                target_fnum = ex["f_num"]
                cand_indices = topk_indices[i].cpu()
                cand_sims = topk_vals[i].cpu()
                pq = _parse_query(ex["query"])
                group_size = 0

                cand_fnums_list = []
                cand_sims_list = []
                topk_fnums = set()

                # Find example's target record indices in all_records
                target_rec_indices = set()
                if force_target:
                    for target_rec in ex["records"]:
                        key = (
                            target_rec["f_num"],
                            target_rec["desig_name"],
                            target_rec["desig_num"],
                            target_rec["prefix"],
                            target_rec["suffix"],
                        )
                        ri = rec_key_to_idx.get(key)
                        if ri is not None:
                            target_rec_indices.add(ri)

                if force_target and target_rec_indices:
                    # First: add the example's target record(s) with label=2
                    for tri in target_rec_indices:
                        rec = all_records[tri]
                        cand_fnum = record_fnums[tri].item()
                        dt_sim = torch.dot(query_embs[i], all_record_embs[tri]).item()
                        topk_fnums.add(cand_fnum)
                        cand_fnums_list.append(cand_fnum)
                        cand_sims_list.append(dt_sim)

                        cand_unit_name = fnum_to_unit_name.get(cand_fnum, "")
                        feats = extract_features(
                            ex["query"],
                            rec,
                            dt_sim,
                            pq,
                            cand_unit_name,
                            idf_union=idf_union,
                            idf_unit=idf_unit,
                            union_probs=batch_union_probs[i],
                            desig_probs=batch_desig_probs[i],
                            prefix_probs=batch_prefix_probs[i],
                            suffix_probs=batch_suffix_probs[i],
                            union_unit_probs=batch_union_unit_probs[i],
                        )
                        row = [feats[fn] for fn in FEATURE_NAMES]
                        all_feats.append(row)
                        all_labels.append(2)
                        group_size += 1

                # Add top-k candidates, skipping forced-in target records
                for j in range(k):
                    rec_idx = cand_indices[j].item()
                    if rec_idx in target_rec_indices:
                        continue
                    rec = all_records[rec_idx]
                    dt_sim = cand_sims[j].item()
                    cand_fnum = record_fnums[rec_idx].item()
                    topk_fnums.add(cand_fnum)
                    cand_fnums_list.append(cand_fnum)
                    cand_sims_list.append(dt_sim)

                    cand_unit_name = fnum_to_unit_name.get(cand_fnum, "")
                    feats = extract_features(
                        ex["query"],
                        rec,
                        dt_sim,
                        pq,
                        cand_unit_name,
                        idf_union=idf_union,
                        idf_unit=idf_unit,
                        union_probs=batch_union_probs[i],
                        desig_probs=batch_desig_probs[i],
                        prefix_probs=batch_prefix_probs[i],
                        suffix_probs=batch_suffix_probs[i],
                        union_unit_probs=batch_union_unit_probs[i],
                    )
                    row = [feats[fn] for fn in FEATURE_NAMES]
                    all_feats.append(row)
                    all_labels.append(1 if cand_fnum == target_fnum else 0)
                    group_size += 1

                # Structural negatives from training examples
                if add_structural and ex.get("structural_negatives"):
                    sn_recs = []
                    sn_fnums = []
                    for cand_data in ex["structural_negatives"]:
                        sn_fnum = cand_data["f_num"]
                        if sn_fnum not in topk_fnums:
                            sn_recs.append(cand_data["record"])
                            sn_fnums.append(sn_fnum)

                    if sn_recs:
                        sn_encoded = encode_record_batch(sn_recs, vocab)
                        sn_encoded = {sk: v.to(DEVICE) for sk, v in sn_encoded.items()}
                        field_emb, _ = model.record_encoder(
                            sn_encoded["union_idx"],
                            sn_encoded["desig_idx"],
                            sn_encoded["prefix_hash"],
                            sn_encoded["num_hash"],
                            sn_encoded["suffix_idx"],
                            sn_encoded["unit_id_idx"],
                        )
                        sn_embs = model.dual_tower.encode_record(field_emb)
                        sn_sims = (
                            torch.matmul(query_embs[i].unsqueeze(0), sn_embs.T)
                            .squeeze(0)
                            .cpu()
                        )

                        for si, (rec, sn_fnum) in enumerate(zip(sn_recs, sn_fnums)):
                            dt_sim = sn_sims[si].item()
                            cand_unit_name = fnum_to_unit_name.get(sn_fnum, "")
                            feats = extract_features(
                                ex["query"],
                                rec,
                                dt_sim,
                                pq,
                                cand_unit_name,
                                idf_union=idf_union,
                                idf_unit=idf_unit,
                                union_probs=batch_union_probs[i],
                                desig_probs=batch_desig_probs[i],
                                prefix_probs=batch_prefix_probs[i],
                                suffix_probs=batch_suffix_probs[i],
                                union_unit_probs=batch_union_unit_probs[i],
                            )
                            row = [feats[fn] for fn in FEATURE_NAMES]
                            all_feats.append(row)
                            all_labels.append(0)
                            group_size += 1

                groups.append(group_size)

                # Candidate info for evaluation
                if target_fnum in topk_fnums:
                    candidate_info.append(
                        (ex["query"], target_fnum, cand_fnums_list, cand_sims_list)
                    )
                else:
                    missed += 1
                    candidate_info.append(None)

    X = np.array(all_feats, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    n = len(examples)
    n_ret = n - missed
    print(
        f"  {X.shape[0]} pairs, {y.sum()} pos ({100*y.mean():.1f}%), "
        f"{len(groups)} groups, DT retrieval: {n_ret}/{n}"
    )
    return X, y, groups, candidate_info, n, n_ret


def build_eval_struct(candidate_info, groups):
    """Pre-compute per-query eval structures for fast alpha sweep."""
    structs = []
    pair_idx = 0
    for info, g in zip(candidate_info, groups):
        if info is None:
            pair_idx += g
            continue
        query, target_fnum, cand_fnums_list, cand_sims_list = info
        fnum_to_best_dt = defaultdict(lambda: float("-inf"))
        fnum_to_indices = defaultdict(list)
        for j in range(len(cand_fnums_list)):
            fnum = cand_fnums_list[j]
            fnum_to_best_dt[fnum] = max(fnum_to_best_dt[fnum], cand_sims_list[j])
            fnum_to_indices[fnum].append(pair_idx + j)
        structs.append(
            (query, target_fnum, dict(fnum_to_best_dt), dict(fnum_to_indices))
        )
        pair_idx += g
    return structs


def evaluate_on_split(ranker, X, eval_struct, n_retrieved):
    """Fast evaluation: returns (lgbm_errors, ens_errors, best_alpha)."""
    lgbm_scores = ranker.predict(X)

    lgbm_correct = 0
    query_aggs = []
    for query, target_fnum, fnum_to_best_dt, fnum_to_indices in eval_struct:
        fnum_to_best_lgbm = {}
        for fnum, indices in fnum_to_indices.items():
            fnum_to_best_lgbm[fnum] = max(lgbm_scores[idx] for idx in indices)

        pred_fnum = max(fnum_to_best_lgbm, key=fnum_to_best_lgbm.get)
        if pred_fnum == target_fnum:
            lgbm_correct += 1
        query_aggs.append((target_fnum, fnum_to_best_dt, fnum_to_best_lgbm))

    lgbm_errors = n_retrieved - lgbm_correct

    best_alpha = None
    best_correct = 0
    for alpha_pct in range(0, 101, 5):
        alpha = alpha_pct / 100.0
        correct = 0
        for target_fnum, fnum_to_best_dt, fnum_to_best_lgbm in query_aggs:
            best_score = float("-inf")
            best_fnum = None
            for fnum in fnum_to_best_dt:
                score = (
                    alpha * fnum_to_best_dt[fnum]
                    + (1 - alpha) * fnum_to_best_lgbm[fnum]
                )
                if score > best_score:
                    best_score = score
                    best_fnum = fnum
            if best_fnum == target_fnum:
                correct += 1
        if correct > best_correct:
            best_correct = correct
            best_alpha = alpha

    best_errors = n_retrieved - best_correct
    return lgbm_errors, best_errors, best_alpha


def build_results(candidate_info, lgbm_scores, groups):
    """Build per-query results for detailed eval / error saving."""
    results = []
    pair_idx = 0
    for info, g in zip(candidate_info, groups):
        if info is None:
            pair_idx += g
            continue
        query, target_fnum, cand_fnums_list, cand_sims_list = info
        candidates = [
            (cand_fnums_list[j], cand_sims_list[j], lgbm_scores[pair_idx + j])
            for j in range(len(cand_fnums_list))
        ]
        pair_idx += g
        results.append((query, target_fnum, candidates))
    return results


def compute_predictions(results, score_fn):
    corrects = []
    for _query, target_fnum, candidates in results:
        fnum_scores = defaultdict(lambda: float("-inf"))
        for cand_fnum, dt_s, lgbm_s in candidates:
            score = score_fn(dt_s, lgbm_s)
            fnum_scores[cand_fnum] = max(fnum_scores[cand_fnum], score)
        pred_fnum = max(fnum_scores, key=fnum_scores.get)
        corrects.append(pred_fnum == target_fnum)
    return corrects


def _describe_records(recs):
    """Format all records for an f_num as a readable string."""
    parts = []
    for r in recs:
        fields = [r.get("union_name", "")]
        fields.append(r.get("desig_name", ""))
        num_parts = []
        if r.get("prefix"):
            num_parts.append(f"pre={r['prefix']}")
        if r.get("desig_num"):
            num_parts.append(f"num={r['desig_num']}")
        if r.get("suffix"):
            num_parts.append(f"suf={r['suffix']}")
        if r.get("unit_id"):
            num_parts.append(f"uid={r['unit_id']}")
        fields.append(" ".join(num_parts) if num_parts else "")
        parts.append(" / ".join(f for f in fields if f))
    return " | ".join(parts)


def main():
    import click

    @click.command()
    @click.option(
        "--checkpoint",
        default=str(Path(__file__).parent / "dual_task_model-v104.ckpt"),
        help="Path to dual tower model checkpoint",
    )
    @click.option("--k", default=50, help="Number of candidates to retrieve per query")
    @click.option(
        "--query-batch-size", default=64, help="Batch size for query encoding"
    )
    @click.option(
        "--smoke-test", is_flag=True, help="Use tiny subset for quick smoke test"
    )
    @click.option(
        "--grid-search", is_flag=True, help="Run hyperparameter grid search on val"
    )
    @click.option("--rebuild-cache", is_flag=True, help="Force rebuild feature cache")
    def run(checkpoint, k, query_batch_size, smoke_test, grid_search, rebuild_cache):
        print(f"Using device: {DEVICE}")
        print(f"Retrieval k={k}")

        # Load vocab and examples
        with open(VOCAB_PATH) as f:
            vocab = json.load(f)
        with open(EXAMPLES_PATH) as f:
            all_examples = json.load(f)
        with open(FNUM_TO_RECORDS_PATH) as f:
            raw = json.load(f)
        fnum_to_records = {int(fk): v for fk, v in raw.items()}

        total_records = sum(len(recs) for recs in fnum_to_records.values())
        print(f"{len(fnum_to_records)} unique f_nums, {total_records} records")

        train_ex = [ex for ex in all_examples if ex["split"] == "train"]
        val_ex = [ex for ex in all_examples if ex["split"] == "val"]
        test_ex = [ex for ex in all_examples if ex["split"] == "test"]

        if smoke_test:
            train_ex = train_ex[:200]
            val_ex = val_ex[:100]
            test_ex = test_ex[:100]
            print("SMOKE TEST MODE")

        print(f"Train: {len(train_ex)}, Val: {len(val_ex)}, Test: {len(test_ex)}")

        # --- Load or build features ---
        cache_path = (
            CACHE_PATH if not smoke_test else DATA_DIR / "lgbm_ranker_cache_smoke.pkl"
        )

        if not rebuild_cache and cache_path.exists():
            print(f"\n--- Loading cached features from {cache_path} ---")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            X_train = cache["X_train"]
            y_train = cache["y_train"]
            train_groups = cache["train_groups"]
            X_val = cache["X_val"]
            y_val = cache["y_val"]
            val_groups = cache["val_groups"]
            val_candidate_info = cache["val_candidate_info"]
            val_n_total = cache["val_n_total"]
            val_n_retrieved = cache["val_n_retrieved"]
            X_test = cache["X_test"]
            y_test = cache["y_test"]
            test_groups = cache["test_groups"]
            test_candidate_info = cache["test_candidate_info"]
            test_n_total = cache["test_n_total"]
            test_n_retrieved = cache["test_n_retrieved"]
            print(
                f"  Train: {X_train.shape[0]} pairs, Val: {X_val.shape[0]} pairs, Test: {X_test.shape[0]} pairs"
            )
        else:
            # Load unit_name from DB
            fnum_to_unit_name = {}
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.execute(
                "SELECT DISTINCT f_num, unit_name FROM lm_data "
                "WHERE unit_name IS NOT NULL AND unit_name <> ''"
            )
            fnum_unit_names = defaultdict(set)
            for f_num, unit_name in cursor:
                fnum_unit_names[f_num].add(unit_name)
            conn.close()
            for f_num, names in fnum_unit_names.items():
                if len(names) == 1:
                    fnum_to_unit_name[f_num] = next(iter(names))
            print(f"{len(fnum_to_unit_name)} f_nums with unique unit_name")

            # Load model
            model = DualTaskModel(
                num_union_names=len(vocab["union_name_to_idx"]),
                num_desig_names=len(vocab["desig_name_to_idx"]),
                num_suffixes=len(vocab["suffix_to_idx"]),
                num_unit_ids=len(vocab["unit_id_to_idx"]),
            )
            ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
            if "state_dict" in ckpt:
                state = {
                    ck.removeprefix("model."): v for ck, v in ckpt["state_dict"].items()
                }
            else:
                state = ckpt["model_state_dict"]
            model.load_state_dict(state)
            model.to(DEVICE)
            model.eval()
            print(f"Loaded model from {checkpoint}")

            print("Encoding all records...")
            all_record_embs, all_records, record_fnums = encode_all_records(
                model, fnum_to_records, vocab, batch_size=512
            )
            all_record_embs = all_record_embs.to(DEVICE)
            print(f"Encoded {len(all_records)} records")

            # Build TF-IDF weights from all record union_names and unit_names
            print("Building trigram IDF weights...")
            union_names = list(
                {rec["union_name"] for rec in all_records if rec.get("union_name")}
            )
            unit_names = list(fnum_to_unit_name.values())
            idf_union = build_trigram_idf(union_names)
            idf_unit = build_trigram_idf(unit_names)
            print(
                f"  IDF union: {len(idf_union)} trigrams from {len(union_names)} names"
            )
            print(f"  IDF unit: {len(idf_unit)} trigrams from {len(unit_names)} names")

            # Load factored classifiers if available
            union_clf_data = None
            desig_clf_data = None
            if UNION_CLF_PATH.exists():
                print(f"Loading union_name classifier from {UNION_CLF_PATH}")
                with open(UNION_CLF_PATH, "rb") as f:
                    union_clf_data = pickle.load(f)
            else:
                print(
                    f"WARNING: {UNION_CLF_PATH} not found, skipping classifier features"
                )
            if DESIG_CLF_PATH.exists():
                print(f"Loading desig_name classifier from {DESIG_CLF_PATH}")
                with open(DESIG_CLF_PATH, "rb") as f:
                    desig_clf_data = pickle.load(f)
            else:
                print(
                    f"WARNING: {DESIG_CLF_PATH} not found, skipping classifier features"
                )
            prefix_clf_data = None
            if PREFIX_CLF_PATH.exists():
                print(f"Loading prefix classifier from {PREFIX_CLF_PATH}")
                with open(PREFIX_CLF_PATH, "rb") as f:
                    prefix_clf_data = pickle.load(f)
            else:
                print(
                    f"WARNING: {PREFIX_CLF_PATH} not found, skipping classifier features"
                )
            suffix_clf_data = None
            if SUFFIX_CLF_PATH.exists():
                print(f"Loading suffix classifier from {SUFFIX_CLF_PATH}")
                with open(SUFFIX_CLF_PATH, "rb") as f:
                    suffix_clf_data = pickle.load(f)
            else:
                print(
                    f"WARNING: {SUFFIX_CLF_PATH} not found, skipping classifier features"
                )
            union_unit_clf_data = None
            if UNION_UNIT_CLF_PATH.exists():
                print(f"Loading union_unit classifier from {UNION_UNIT_CLF_PATH}")
                with open(UNION_UNIT_CLF_PATH, "rb") as f:
                    union_unit_clf_data = pickle.load(f)
            else:
                print(
                    f"WARNING: {UNION_UNIT_CLF_PATH} not found, skipping classifier features"
                )

            common_args = dict(
                model=model,
                all_record_embs=all_record_embs,
                all_records=all_records,
                record_fnums=record_fnums,
                fnum_to_unit_name=fnum_to_unit_name,
                vocab=vocab,
                k=k,
                query_batch_size=query_batch_size,
                idf_union=idf_union,
                idf_unit=idf_unit,
                union_clf=union_clf_data,
                desig_clf=desig_clf_data,
                prefix_clf=prefix_clf_data,
                suffix_clf=suffix_clf_data,
                union_unit_clf=union_unit_clf_data,
            )

            print("\n--- Building training features ---")
            X_train, y_train, train_groups, _, _, _ = build_features_and_candidates(
                train_ex,
                **common_args,
                desc="Train features",
                add_structural=False,
            )

            print("\n--- Building val features ---")
            (
                X_val,
                y_val,
                val_groups,
                val_candidate_info,
                val_n_total,
                val_n_retrieved,
            ) = build_features_and_candidates(
                val_ex, **common_args, desc="Val features"
            )

            print("\n--- Building test features ---")
            (
                X_test,
                y_test,
                test_groups,
                test_candidate_info,
                test_n_total,
                test_n_retrieved,
            ) = build_features_and_candidates(
                test_ex, **common_args, desc="Test features"
            )

            # Save cache
            print(f"\n--- Saving cache to {cache_path} ---")
            cache = {
                "X_train": X_train,
                "y_train": y_train,
                "train_groups": train_groups,
                "X_val": X_val,
                "y_val": y_val,
                "val_groups": val_groups,
                "val_candidate_info": val_candidate_info,
                "val_n_total": val_n_total,
                "val_n_retrieved": val_n_retrieved,
                "X_test": X_test,
                "y_test": y_test,
                "test_groups": test_groups,
                "test_candidate_info": test_candidate_info,
                "test_n_total": test_n_total,
                "test_n_retrieved": test_n_retrieved,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"  Saved ({cache_path.stat().st_size / 1e6:.1f} MB)")

        # --- Build eval structures ---
        val_eval_struct = build_eval_struct(val_candidate_info, val_groups)
        _test_eval_struct = build_eval_struct(test_candidate_info, test_groups)

        # --- LightGBM datasets ---
        print(f"\nFeatures: {len(FEATURE_NAMES)}")
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            group=train_groups,
            feature_name=FEATURE_NAMES,
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            group=val_groups,
            feature_name=FEATURE_NAMES,
            reference=train_data,
        )

        if grid_search:
            # --- Grid search over hyperparameters (on val set) ---
            from itertools import product

            param_grid = {
                "num_leaves": [31, 63, 127],
                "max_depth": [4, 6, -1],
                "learning_rate": [0.05, 0.1],
                "min_data_in_leaf": [10, 20, 50],
                "num_boost_round": [500],
            }

            grid_keys = list(param_grid.keys())
            grid_combos = list(product(*[param_grid[gk] for gk in grid_keys]))
            print(f"\n--- Grid search on val: {len(grid_combos)} combinations ---")

            grid_results = []
            for combo_idx, combo in enumerate(grid_combos):
                combo_dict = dict(zip(grid_keys, combo))
                num_boost_round = combo_dict.pop("num_boost_round")

                params = {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "ndcg_eval_at": [1, 5],
                    "verbose": -1,
                    "seed": 42,
                    **combo_dict,
                }

                ranker = lgb.train(
                    params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[val_data],
                    valid_names=["val"],
                )

                lgbm_errors, ens_errors, best_alpha = evaluate_on_split(
                    ranker, X_val, val_eval_struct, val_n_retrieved
                )
                combo_dict["num_boost_round"] = num_boost_round
                grid_results.append((lgbm_errors, ens_errors, best_alpha, combo_dict))
                print(
                    f"  [{combo_idx+1:>3}/{len(grid_combos)}] "
                    f"lgbm={lgbm_errors:>3} ens={ens_errors:>3} α={best_alpha:.2f}  "
                    f"{combo_dict}"
                )

            grid_results.sort(key=lambda x: (x[1], x[0]))
            print("\n--- Top 10 configurations (val set) ---")
            print(f"{'lgbm':>5} {'ens':>5} {'alpha':>6}  params")
            print("-" * 80)
            for lgbm_errors, ens_errors, best_alpha, combo_dict in grid_results[:10]:
                print(
                    f"{lgbm_errors:>5} {ens_errors:>5} {best_alpha:>6.2f}  "
                    f"{combo_dict}"
                )

            # Use best config for final eval
            best_combo = grid_results[0][3].copy()
            print("\n--- Best config: final eval on test set ---")
            print(f"Params: {best_combo}")
            num_boost_round = best_combo.pop("num_boost_round")
            train_params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 5],
                "verbose": 1,
                "seed": 42,
                **best_combo,
            }
        else:
            # Default hyperparameters
            num_boost_round = 500
            train_params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [1, 5],
                "num_leaves": 63,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_data_in_leaf": 20,
                "verbose": 1,
                "seed": 42,
            }

        # --- Train final model ---
        print("\n--- Training LightGBM LambdaRank ---")
        print(f"Params: {train_params}")
        ranker = lgb.train(
            train_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.log_evaluation(50)],
        )

        # Feature importances
        print("\nFeature importances (gain):")
        importance = ranker.feature_importance(importance_type="gain")
        for fname, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
            print(f"  {fname:30s} {imp:.1f}")

        # --- Eval on test ---
        lgbm_scores = ranker.predict(X_test)
        test_results = build_results(test_candidate_info, lgbm_scores, test_groups)

        dt_preds = compute_predictions(test_results, lambda dt_s, lgbm_s: dt_s)
        dt_correct = sum(dt_preds)
        print(
            f"\nDT rerank accuracy: {dt_correct}/{test_n_retrieved} "
            f"= {100*dt_correct/test_n_retrieved:.2f}% "
            f"(overall: {100*dt_correct/test_n_total:.2f}%)"
        )

        lgbm_preds = compute_predictions(test_results, lambda dt_s, lgbm_s: lgbm_s)
        lgbm_correct = sum(lgbm_preds)
        print(
            f"LGBM rerank accuracy: {lgbm_correct}/{test_n_retrieved} "
            f"= {100*lgbm_correct/test_n_retrieved:.2f}% "
            f"(overall: {100*lgbm_correct/test_n_total:.2f}%)"
        )

        dt_wrong_lgbm_fixed = sum(g and not d for g, d in zip(lgbm_preds, dt_preds))
        dt_right_lgbm_broke = sum(d and not g for g, d in zip(lgbm_preds, dt_preds))
        both_wrong = sum(not d and not g for g, d in zip(lgbm_preds, dt_preds))
        both_right = sum(d and g for g, d in zip(lgbm_preds, dt_preds))
        dt_wrong = test_n_retrieved - dt_correct
        print(f"\nConditional analysis (among {test_n_retrieved} retrieved):")
        print(f"  Both correct:        {both_right}")
        print(f"  DT wrong, LGBM fixed: {dt_wrong_lgbm_fixed}/{dt_wrong}")
        print(f"  DT right, LGBM broke: {dt_right_lgbm_broke}/{dt_correct}")
        print(f"  Both wrong:          {both_wrong}")

        # Ensemble sweep
        print(
            f"\n{'alpha':>7}  {'correct':>7}  {'errors':>6}  "
            f"{'acc_ret':>7}  {'acc_all':>7}"
        )
        print("-" * 50)

        best_alpha = None
        best_correct = 0

        for alpha_pct in range(0, 101, 5):
            alpha = alpha_pct / 100.0
            correct = sum(
                compute_predictions(
                    test_results,
                    lambda dt_s, lgbm_s, a=alpha: a * dt_s + (1 - a) * lgbm_s,
                )
            )
            errors = test_n_retrieved - correct
            acc_ret = 100 * correct / max(test_n_retrieved, 1)
            acc_all = 100 * correct / test_n_total
            print(
                f"  {alpha:.2f}   {correct:>7}  {errors:>6}  "
                f"{acc_ret:>6.2f}%  {acc_all:>6.2f}%"
            )
            if correct > best_correct:
                best_correct = correct
                best_alpha = alpha

        print("-" * 50)
        print(
            f"Best ensemble: alpha={best_alpha:.2f}, "
            f"{best_correct}/{test_n_total} = {100*best_correct/test_n_total:.2f}%"
        )

        # Save errors
        errors_list = []
        for query, target_fnum, candidates in test_results:
            fnum_scores = defaultdict(lambda: float("-inf"))
            for cand_fnum, dt_s, lgbm_s in candidates:
                fnum_scores[cand_fnum] = max(fnum_scores[cand_fnum], lgbm_s)
            pred_fnum = max(fnum_scores, key=fnum_scores.get)
            if pred_fnum != target_fnum:
                target_recs = fnum_to_records.get(target_fnum, [])
                pred_recs = fnum_to_records.get(pred_fnum, [])
                errors_list.append(
                    {
                        "query": query,
                        "target_fnum": target_fnum,
                        "target_desc": _describe_records(target_recs),
                        "target_score": fnum_scores.get(target_fnum, float("-inf")),
                        "pred_fnum": pred_fnum,
                        "pred_desc": _describe_records(pred_recs),
                        "pred_score": fnum_scores[pred_fnum],
                    }
                )

        error_file = DATA_DIR / "lgbm_ranker_errors.csv"
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
                ],
            )
            writer.writeheader()
            writer.writerows(errors_list)

        print(f"\n{len(errors_list)} LGBM rerank errors saved to {error_file}")

    run()


if __name__ == "__main__":
    main()
