#!/usr/bin/env python3
"""
GBM Reranker for dual tower retrieval pipeline.

Trains a HistGradientBoostingClassifier on hand-crafted features extracted
from (query_text, candidate_record, dual_tower_similarity) triples.

Features directly encode field comparisons that the cross-encoder discovers
only weakly through attention: number matching, designation keywords, and
union name similarity.
"""

import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import encode_query_batch
from mining import encode_all_records
from prepare_data import DESIG_KEYWORDS, DESIG_PREFERENCE
from train_dual_task import DEVICE, EXAMPLES_PATH, FNUM_TO_RECORDS_PATH, VOCAB_PATH

from labor_union_parser.char_cnn import tokenize_to_chars
from labor_union_parser.model import DualTaskModel

DATA_DIR = Path(__file__).parent / "data"

DB_PATH = Path(__file__).parent.parent / "opdr.db"

# ============================================================================
# Feature extraction — mirrors filter_records_by_query logic per-record
# ============================================================================


def _norm_suffix(s):
    """Normalize suffix for comparison: strip leading zeros from numeric."""
    s = s.strip().upper()
    if s.isdigit():
        return str(int(s))
    return s


def _parse_query(query: str) -> dict:
    """
    Extract all matching signals from a query string once.

    Uses the same regex tokenizer as the dual tower model for consistent
    number/word splitting. Returns a dict of parsed components reusable
    across candidate records.
    """
    _, tokens, is_num, token_type, _ = tokenize_to_chars(query, max_tokens=999)

    # Strip padding and spaces, keep (token, is_number, token_type) triples
    toks = [
        (t, isn, tt)
        for t, isn, tt in zip(tokens, is_num, token_type)
        if t and tt != 4  # not empty/padding
    ]

    # Stage 1: numbers (tokenizer already strips leading zeros)
    all_query_nums = set()
    query_nums_ordered = []  # preserve appearance order (deduped)
    for t, isn, _ in toks:
        if isn:
            n = int(t)
            if n not in all_query_nums:
                query_nums_ordered.append(n)
            all_query_nums.add(n)
    query_nums_pos = set(n for n in all_query_nums if n >= 1)

    # Stage 2: codes adjacent to numbers (no space between them)
    # The tokenizer splits "74A" into ["74", "a"] and "D583" into ["d", "583"]
    # We check adjacency in the FULL token stream (with spaces) so that
    # "FOP 12" doesn't wrongly extract "FOP" as a code.
    query_codes = set()
    for i, (t, isn, tt) in enumerate(toks):
        if tt == 0 and len(t) <= 3:  # short word token
            # Check immediate neighbors (no space allowed between)
            prev_is_num = i > 0 and toks[i - 1][1]
            next_is_num = i < len(toks) - 1 and toks[i + 1][1]
            # Allow punct bridge (no space): word-punct-number or number-punct-word
            prev_is_num_via_punct = (
                i > 1 and toks[i - 1][2] == 3 and toks[i - 2][1]  # punct  # number
            )
            next_is_num_via_punct = (
                i < len(toks) - 2
                and toks[i + 1][2] == 3  # punct
                and toks[i + 2][1]  # number
            )
            if (
                prev_is_num
                or next_is_num
                or prev_is_num_via_punct
                or next_is_num_via_punct
            ):
                query_codes.add(t.upper())

    # Numeric suffixes: number-punct-smallnumber patterns like "1410-1"
    # Use nonspace to skip whitespace between tokens
    nonspace = [(t, isn, tt) for t, isn, tt in toks if tt != 2]
    numeric_suf = set()
    for i, (t, isn, tt) in enumerate(nonspace):
        if isn and int(t) >= 100:  # large number
            # Look for punct then small number after
            if (
                i < len(nonspace) - 2
                and nonspace[i + 1][2] == 3  # punct (hyphen etc)
                and nonspace[i + 2][1]  # number
                and int(nonspace[i + 2][0]) < 100  # small number
            ):
                numeric_suf.add(nonspace[i + 2][0])

    query_suffixes = query_codes | numeric_suf
    query_suffixes_norm = set(_norm_suffix(s) for s in query_suffixes)

    # Stage 3: designation keyword (longest-first)
    query_lower = query.lower()
    matched_keyword = None
    matched_desig_set = None
    for keyword, desig_set in DESIG_KEYWORDS:
        if keyword in query_lower:
            matched_keyword = keyword
            matched_desig_set = desig_set
            break

    return {
        "all_query_nums": all_query_nums,
        "query_nums_ordered": query_nums_ordered,
        "query_nums_pos": query_nums_pos,
        "query_codes": query_codes,
        "query_suffixes_norm": query_suffixes_norm,
        "matched_keyword": matched_keyword,
        "matched_desig_set": matched_desig_set,
        "matched_desig_set_size": len(matched_desig_set) if matched_desig_set else 0,
    }


def _char_trigrams(s: str) -> set:
    """Return character 3-grams of a string (lowercased)."""
    s = s.lower().strip()
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def build_trigram_idf(documents):
    """Compute IDF weights from a collection of strings."""
    import math

    df = defaultdict(int)
    n = 0
    for doc in documents:
        trigrams = _char_trigrams(doc)
        for tri in trigrams:
            df[tri] += 1
        n += 1
    idf = {tri: math.log(n / count) for tri, count in df.items()}
    return idf


def _tfidf_cosine(a_trigrams: set, b_trigrams: set, idf: dict) -> float:
    """TF-IDF weighted cosine similarity between two trigram sets.
    Uses binary TF (present/absent) weighted by IDF."""
    shared = a_trigrams & b_trigrams
    if not shared:
        return 0.0
    # Numerator: sum of idf^2 for shared trigrams
    num = sum(idf.get(tri, 0.0) ** 2 for tri in shared)
    # Denominator: product of L2 norms
    norm_a = sum(idf.get(tri, 0.0) ** 2 for tri in a_trigrams)
    norm_b = sum(idf.get(tri, 0.0) ** 2 for tri in b_trigrams)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return num / (norm_a**0.5 * norm_b**0.5)


def extract_features(
    query_text: str,
    record: dict,
    dt_sim: float,
    parsed_query: dict,
    unit_name: str = "",
    idf_union: dict | None = None,
    idf_unit: dict | None = None,
    union_probs: dict | None = None,
    desig_probs: dict | None = None,
    prefix_probs: dict | None = None,
    suffix_probs: dict | None = None,
    union_unit_probs: dict | None = None,
) -> dict:
    """
    Extract features for a (query, candidate_record) pair.

    Uses pre-parsed query signals (from _parse_query) to avoid redundant parsing
    across the k candidates per query.
    """
    features = {}
    features["dt_sim"] = dt_sim

    pq = parsed_query
    desig_num = record.get("desig_num", 0)
    prefix = record.get("prefix", 0) or 0
    suffix = (record.get("suffix", "") or "").strip()
    suffix_norm = _norm_suffix(suffix) if suffix else ""
    desig_name = record.get("desig_name", "")

    # --- Stage 1: desig_num matching ---
    features["desig_num_in_query"] = float(
        desig_num > 0 and desig_num in pq["query_nums_pos"]
    )
    features["desig_num_is_zero"] = float(desig_num == 0)
    # Inverse: query has no numbers and record has desig_num=0
    features["zero_num_no_query_nums"] = float(
        desig_num == 0 and not pq["query_nums_pos"]
    )

    # --- Number position features ---
    nums_ord = pq["query_nums_ordered"]
    if nums_ord:
        first_num = nums_ord[0]
        last_num = nums_ord[-1]
        features["desig_num_is_first"] = float(desig_num > 0 and desig_num == first_num)
        features["desig_num_is_last"] = float(desig_num > 0 and desig_num == last_num)
        features["prefix_is_first"] = float(prefix > 0 and prefix == first_num)
    else:
        features["desig_num_is_first"] = 0.0
        features["desig_num_is_last"] = 0.0
        features["prefix_is_first"] = 0.0

    # --- Stage 2: prefix matching ---
    features["prefix_in_query"] = float(prefix > 0 and prefix in pq["all_query_nums"])
    features["record_has_prefix"] = float(prefix > 0)
    # Both desig_num and prefix found in query numbers
    features["num_and_prefix_in_query"] = float(
        features["desig_num_in_query"] and features["prefix_in_query"]
    )

    # --- Stage 2: suffix matching ---
    query_has_codes = bool(pq["query_suffixes_norm"])
    features["suffix_in_query"] = float(
        bool(suffix_norm) and suffix_norm in pq["query_suffixes_norm"]
    )
    features["record_has_suffix"] = float(bool(suffix))
    # Inverse: query has no suffix codes and record has no suffix
    features["no_suffix_no_query_suffix"] = float(not suffix and not query_has_codes)
    # Mismatch: query has codes, record has suffix, but they don't match
    features["suffix_mismatch"] = float(
        bool(suffix_norm)
        and query_has_codes
        and suffix_norm not in pq["query_suffixes_norm"]
    )
    # Record has suffix but query specifies no codes (plain variant preferred)
    features["record_suffix_no_query_code"] = float(
        bool(suffix) and not query_has_codes
    )

    # --- Number coverage ---
    matched_nums = set()
    if desig_num > 0 and desig_num in pq["query_nums_pos"]:
        matched_nums.add(desig_num)
    if prefix > 0 and prefix in pq["all_query_nums"]:
        matched_nums.add(prefix)
    if suffix_norm and suffix_norm.isdigit():
        suf_val = int(suffix_norm)
        if suf_val in pq["all_query_nums"]:
            matched_nums.add(suf_val)

    n_query = len(pq["query_nums_pos"])
    n_matched = len(matched_nums)
    features["num_query_numbers"] = float(n_query)
    features["frac_nums_matched"] = n_matched / max(n_query, 1)
    features["all_nums_matched"] = float(n_query > 0 and n_matched >= n_query)

    # --- Stage 3: desig_name keyword matching ---
    matched_set = pq["matched_desig_set"]
    set_size = pq["matched_desig_set_size"]

    if matched_set is not None:
        in_set = desig_name in matched_set
        # General match/mismatch
        features["desig_keyword_match"] = float(in_set)
        features["desig_keyword_mismatch"] = float(not in_set)
        # Exclusive: keyword maps to exactly 1 desig type (e.g., "district council" → DC)
        features["desig_keyword_exclusive_match"] = float(set_size == 1 and in_set)
        features["desig_keyword_exclusive_mismatch"] = float(
            set_size == 1 and not in_set
        )
        # Ambiguous: keyword maps to 2+ types (e.g., "lodge" → LG,LLG,DLG)
        features["desig_keyword_ambiguous_match"] = float(set_size > 1 and in_set)
    else:
        features["desig_keyword_match"] = 0.0
        features["desig_keyword_mismatch"] = 0.0
        features["desig_keyword_exclusive_match"] = 0.0
        features["desig_keyword_exclusive_mismatch"] = 0.0
        features["desig_keyword_ambiguous_match"] = 0.0

    # Code-based desig_name matching (codes near numbers like "D583" -> "D")
    features["desig_code_match"] = float(
        matched_set is None
        and bool(pq["query_codes"])
        and desig_name in pq["query_codes"]
    )
    features["desig_code_mismatch"] = float(
        matched_set is None
        and bool(pq["query_codes"])
        and desig_name not in pq["query_codes"]
    )

    # Preference: is this desig_name a preferred variant over alternatives?
    is_preferred = False
    is_alternate = False
    for preferred, alternates in DESIG_PREFERENCE:
        if desig_name == preferred:
            is_preferred = True
            break
        if desig_name in alternates:
            is_alternate = True
            break
    features["desig_is_preferred"] = float(is_preferred)
    features["desig_is_alternate"] = float(is_alternate)

    # --- Union name features ---
    union_name = record.get("union_name", "")
    features["union_trigram_jaccard"] = _jaccard(
        _char_trigrams(query_text), _char_trigrams(union_name)
    )

    # --- Unit name features (geographic/organizational context from DB) ---
    query_trigrams = _char_trigrams(query_text)
    features["unit_name_trigram_jaccard"] = (
        _jaccard(query_trigrams, _char_trigrams(unit_name)) if unit_name else 0.0
    )
    features["has_unit_name"] = float(bool(unit_name))

    # --- TF-IDF weighted cosine similarity ---
    union_trigrams = _char_trigrams(union_name)
    if idf_union is not None:
        features["union_tfidf_cosine"] = _tfidf_cosine(
            query_trigrams, union_trigrams, idf_union
        )
    else:
        features["union_tfidf_cosine"] = 0.0

    if idf_unit is not None and unit_name:
        features["unit_name_tfidf_cosine"] = _tfidf_cosine(
            query_trigrams, _char_trigrams(unit_name), idf_unit
        )
    else:
        features["unit_name_tfidf_cosine"] = 0.0

    # --- Factored classifier probability features ---
    union_clf_prob = union_probs.get(union_name, 0.0) if union_probs else 0.0
    features["union_clf_prob"] = union_clf_prob
    features["union_clf_log_prob"] = math.log(max(union_clf_prob, 1e-10))

    desig_clf_prob = desig_probs.get(desig_name, 0.0) if desig_probs else 0.0
    features["desig_clf_prob"] = desig_clf_prob
    features["desig_clf_log_prob"] = math.log(max(desig_clf_prob, 1e-10))

    prefix_clf_prob = prefix_probs.get(prefix, 0.0) if prefix_probs else 0.0
    features["prefix_clf_prob"] = prefix_clf_prob
    features["prefix_clf_log_prob"] = math.log(max(prefix_clf_prob, 1e-10))

    suffix_val = record.get("suffix", "") or ""
    suffix_clf_prob = suffix_probs.get(suffix_val, 0.0) if suffix_probs else 0.0
    features["suffix_clf_prob"] = suffix_clf_prob
    features["suffix_clf_log_prob"] = math.log(max(suffix_clf_prob, 1e-10))

    # union_unit classifier: predicts union_name or union_name|unit_id
    if union_unit_probs:
        unit_id = record.get("unit_id", "")
        if unit_id:
            union_unit_key = f"{union_name}|{unit_id}"
        else:
            union_unit_key = union_name
        union_unit_clf_prob = union_unit_probs.get(union_unit_key, 0.0)
    else:
        union_unit_clf_prob = 0.0
    features["union_unit_clf_prob"] = union_unit_clf_prob
    features["union_unit_clf_log_prob"] = math.log(max(union_unit_clf_prob, 1e-10))

    return features


FEATURE_NAMES = list(
    extract_features(
        "dummy query 123",
        {"desig_num": 0, "prefix": 0},
        0.5,
        _parse_query("dummy query 123"),
    ).keys()
)


# ============================================================================
# Pipeline
# ============================================================================


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
    def run(checkpoint, k, query_batch_size, smoke_test):
        print(f"Using device: {DEVICE}")
        print(f"Retrieval k={k}")
        print(f"Features: {len(FEATURE_NAMES)}")

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

        # Load unit_name from DB (geographic/organizational context)
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
        # Use unit_name only for f_nums with exactly one distinct value
        for f_num, names in fnum_unit_names.items():
            if len(names) == 1:
                fnum_to_unit_name[f_num] = next(iter(names))
        print(f"{len(fnum_to_unit_name)} f_nums with unique unit_name")

        train_ex = [ex for ex in all_examples if ex["split"] == "train"]
        test_ex = [ex for ex in all_examples if ex["split"] == "test"]

        if smoke_test:
            train_ex = train_ex[:200]
            test_ex = test_ex[:100]
            print("SMOKE TEST MODE")

        print(f"Train: {len(train_ex)}, Test: {len(test_ex)}")

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

        # Encode all records
        print("Encoding all records...")
        all_record_embs, all_records, record_fnums = encode_all_records(
            model, fnum_to_records, vocab, batch_size=512
        )
        all_record_embs = all_record_embs.to(DEVICE)
        print(f"Encoded {len(all_records)} records")

        # --- Build training and test features ---
        def build_features(examples, desc="Building features"):
            """Retrieve top-k candidates and extract features for each pair."""
            all_feats = []
            all_labels = []

            with torch.no_grad():
                for batch_start in tqdm(
                    range(0, len(examples), query_batch_size), desc=desc
                ):
                    batch_ex = examples[batch_start : batch_start + query_batch_size]
                    batch_queries = [ex["query"] for ex in batch_ex]

                    # Encode queries
                    q_batch = encode_query_batch(batch_queries)
                    q_batch = {qk: v.to(DEVICE) for qk, v in q_batch.items()}
                    token_emb, padding_mask = model.query_encoder(
                        q_batch["char_ids"],
                        q_batch["is_number"],
                        q_batch["numeric_ids"],
                    )
                    query_embs = model.dual_tower.encode_query(token_emb, padding_mask)

                    # Retrieve top-k
                    sims = torch.matmul(query_embs, all_record_embs.T)
                    topk_vals, topk_indices = sims.topk(k, dim=1)

                    # Extract features per query
                    for i, ex in enumerate(batch_ex):
                        target_fnum = ex["f_num"]
                        cand_indices = topk_indices[i].cpu()
                        cand_sims = topk_vals[i].cpu()
                        pq = _parse_query(ex["query"])

                        for j in range(k):
                            rec_idx = cand_indices[j].item()
                            rec = all_records[rec_idx]
                            dt_sim = cand_sims[j].item()
                            cand_fnum = record_fnums[rec_idx].item()

                            cand_unit_name = fnum_to_unit_name.get(cand_fnum, "")
                            feats = extract_features(
                                ex["query"], rec, dt_sim, pq, cand_unit_name
                            )
                            all_feats.append([feats[fn] for fn in FEATURE_NAMES])
                            all_labels.append(1 if cand_fnum == target_fnum else 0)

            X = np.array(all_feats, dtype=np.float32)
            y = np.array(all_labels, dtype=np.int32)
            return X, y

        print("\n--- Building training features ---")
        X_train, y_train = build_features(train_ex, "Train features")
        print(
            f"Training: {X_train.shape[0]} pairs, "
            f"{y_train.sum()} positive ({100*y_train.mean():.1f}%)"
        )

        print("\n--- Building test features ---")
        X_test, y_test = build_features(test_ex, "Test features")
        print(
            f"Test: {X_test.shape[0]} pairs, "
            f"{y_test.sum()} positive ({100*y_test.mean():.1f}%)"
        )

        # --- Train GBM ---
        print("\n--- Training HistGradientBoostingClassifier ---")
        gbm = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=20,
            max_leaf_nodes=63,
            verbose=1,
            random_state=42,
        )
        gbm.fit(X_train, y_train)

        # Feature importances via permutation
        from sklearn.inspection import permutation_importance

        print("\nComputing permutation importances on test set...")
        perm_result = permutation_importance(
            gbm, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        print("\nFeature importances (permutation, mean decrease in accuracy):")
        for fname, imp in sorted(
            zip(FEATURE_NAMES, perm_result.importances_mean),
            key=lambda x: -x[1],
        ):
            print(f"  {fname:30s} {imp:.4f}")

        # --- Evaluate: per-query accuracy ---
        print("\n--- Evaluating on test set ---")
        # We need to evaluate per-query: for each query, score all k candidates,
        # aggregate by f_num, check if target is top-ranked.
        # Reconstruct per-query structure from flat arrays.

        # Re-retrieve to get candidate fnums for test set
        test_results = (
            []
        )  # list of (query, target_fnum, [(cand_fnum, dt_sim, gbm_score)])
        dt_missed = 0

        gbm_proba = gbm.predict_proba(X_test)[:, 1]
        pair_idx = 0

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, len(test_ex), query_batch_size), desc="Test eval"
            ):
                batch_ex = test_ex[batch_start : batch_start + query_batch_size]
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

                for i, ex in enumerate(batch_ex):
                    target_fnum = ex["f_num"]
                    cand_indices = topk_indices[i].cpu()
                    cand_sims = topk_vals[i].cpu()
                    cand_fnums_list = [
                        record_fnums[cand_indices[j].item()].item() for j in range(k)
                    ]

                    if target_fnum not in cand_fnums_list:
                        dt_missed += 1
                        pair_idx += k
                        continue

                    candidates = []
                    for j in range(k):
                        candidates.append(
                            (
                                cand_fnums_list[j],
                                cand_sims[j].item(),
                                gbm_proba[pair_idx + j],
                            )
                        )
                    pair_idx += k
                    test_results.append((ex["query"], target_fnum, candidates))

        n_total = len(test_ex)
        n_retrieved = len(test_results)
        print(
            f"\nDual tower retrieval: {n_retrieved}/{n_total} "
            f"= {100*n_retrieved/n_total:.2f}%"
        )
        print(f"DT missed: {dt_missed}")

        # Score aggregation and accuracy computation
        def compute_predictions(results, score_fn):
            """
            For each query, aggregate candidate scores by f_num (max),
            return list of bools (correct or not).
            """
            corrects = []
            for _query, target_fnum, candidates in results:
                fnum_scores = defaultdict(lambda: float("-inf"))
                for cand_fnum, dt_s, gbm_s in candidates:
                    score = score_fn(dt_s, gbm_s)
                    fnum_scores[cand_fnum] = max(fnum_scores[cand_fnum], score)
                pred_fnum = max(fnum_scores, key=fnum_scores.get)
                corrects.append(pred_fnum == target_fnum)
            return corrects

        # DT-only accuracy (among retrieved)
        dt_preds = compute_predictions(test_results, lambda dt_s, gbm_s: dt_s)
        dt_correct = sum(dt_preds)
        print(
            f"DT rerank accuracy: {dt_correct}/{n_retrieved} "
            f"= {100*dt_correct/n_retrieved:.2f}% "
            f"(overall: {100*dt_correct/n_total:.2f}%)"
        )

        # GBM-only accuracy (among retrieved)
        gbm_preds = compute_predictions(test_results, lambda dt_s, gbm_s: gbm_s)
        gbm_correct = sum(gbm_preds)
        print(
            f"GBM rerank accuracy: {gbm_correct}/{n_retrieved} "
            f"= {100*gbm_correct/n_retrieved:.2f}% "
            f"(overall: {100*gbm_correct/n_total:.2f}%)"
        )

        # Conditional accuracy: GBM given DT outcome
        dt_wrong_gbm_fixed = sum(g and not d for g, d in zip(gbm_preds, dt_preds))
        dt_right_gbm_broke = sum(d and not g for g, d in zip(gbm_preds, dt_preds))
        both_wrong = sum(not d and not g for g, d in zip(gbm_preds, dt_preds))
        both_right = sum(d and g for g, d in zip(gbm_preds, dt_preds))
        dt_wrong = n_retrieved - dt_correct
        print(f"\nConditional analysis (among {n_retrieved} retrieved):")
        print(f"  Both correct:      {both_right}")
        print(f"  DT wrong, GBM fixed: {dt_wrong_gbm_fixed}/{dt_wrong}")
        print(f"  DT right, GBM broke: {dt_right_gbm_broke}/{dt_correct}")
        print(f"  Both wrong:        {both_wrong}")

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
                    lambda dt_s, gbm_s, a=alpha: a * dt_s + (1 - a) * gbm_s,
                )
            )
            errors = n_retrieved - correct
            acc_ret = 100 * correct / max(n_retrieved, 1)
            acc_all = 100 * correct / n_total
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
            f"{best_correct}/{n_total} = {100*best_correct/n_total:.2f}%"
        )

        # Save errors for analysis
        import csv

        errors_list = []
        pair_idx = 0

        # Re-walk test results to find errors
        for query, target_fnum, candidates in test_results:
            fnum_scores = defaultdict(lambda: float("-inf"))
            for cand_fnum, dt_s, gbm_s in candidates:
                fnum_scores[cand_fnum] = max(fnum_scores[cand_fnum], gbm_s)
            pred_fnum = max(fnum_scores, key=fnum_scores.get)
            if pred_fnum != target_fnum:
                target_recs = fnum_to_records.get(target_fnum, [])
                pred_recs = fnum_to_records.get(pred_fnum, [])
                target_desc = ""
                pred_desc = ""
                if target_recs:
                    r = target_recs[0]
                    target_desc = (
                        f"{r.get('union_name', '')} / "
                        f"{r.get('desig_name', '')} "
                        f"{r.get('prefix', '')}{r.get('desig_num', '')} "
                        f"{r.get('suffix', '')}"
                    )
                if pred_recs:
                    r = pred_recs[0]
                    pred_desc = (
                        f"{r.get('union_name', '')} / "
                        f"{r.get('desig_name', '')} "
                        f"{r.get('prefix', '')}{r.get('desig_num', '')} "
                        f"{r.get('suffix', '')}"
                    )
                errors_list.append(
                    {
                        "query": query,
                        "target_fnum": target_fnum,
                        "target_desc": target_desc.strip(),
                        "target_score": fnum_scores.get(target_fnum, float("-inf")),
                        "pred_fnum": pred_fnum,
                        "pred_desc": pred_desc.strip(),
                        "pred_score": fnum_scores[pred_fnum],
                    }
                )

        error_file = DATA_DIR / "gbm_reranker_errors.csv"
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

        print(f"\n{len(errors_list)} GBM rerank errors saved to {error_file}")

    run()


if __name__ == "__main__":
    main()
