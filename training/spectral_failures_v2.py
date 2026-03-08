#!/usr/bin/env python3
"""Analyze spectral failures — only same-prefix, different-number pairs.

These are the actual constraint violations: texts that share non-number
tokens (same union family) but have disjoint number tokens.
"""

import sys
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, ".")
from training.train_record_embeddings import (
    _tokenize,
    build_token_vocab,
    load_f7_records,
)

print("Loading data...", flush=True)
records = load_f7_records("f7.db")
token_vocab = build_token_vocab(records, min_count=2)

# Build distinct texts and extract number sets + word tokens
token_sig_to_text: dict[tuple, str] = {}
for rec in records:
    un = rec["union_name"].strip()
    if not un:
        continue
    tokens = _tokenize("union_name", un)
    sig = tuple((t["token"], t["is_num"]) for t in tokens)
    if sig not in token_sig_to_text:
        token_sig_to_text[sig] = un

distinct_texts = sorted(token_sig_to_text.values())
print(f"  {len(distinct_texts)} distinct texts", flush=True)

# Extract number sets and word-token prefix per text
number_sets = []
word_prefixes = []  # tuple of non-number tokens
all_numbers = set()
for text in distinct_texts:
    tokens = _tokenize("union_name", text)
    nums = set()
    words = []
    for tok in tokens:
        if tok["is_num"]:
            nums.add(tok["token"])
            all_numbers.add(tok["token"])
        else:
            words.append(tok["token"])
    number_sets.append(frozenset(nums))
    word_prefixes.append(tuple(words))

has_nums_idx = [i for i, s in enumerate(number_sets) if s]
print(
    f"  {len(has_nums_idx)} texts with numbers, {len(all_numbers)} distinct numbers",
    flush=True,
)

# Group by word prefix — these are "same union family"
prefix_to_idx: dict[tuple, list[int]] = defaultdict(list)
for i in has_nums_idx:
    prefix_to_idx[word_prefixes[i]].append(i)

# Only keep prefixes with 2+ members (can form can't-link pairs)
multi_prefix = {p: idxs for p, idxs in prefix_to_idx.items() if len(idxs) >= 2}
print(f"  {len(multi_prefix)} prefixes with 2+ texts with numbers", flush=True)
total_in_families = sum(len(v) for v in multi_prefix.values())
print(f"  {total_in_families} texts in multi-member families", flush=True)

# Build D for all texts with numbers
num_to_col = {n: i for i, n in enumerate(sorted(all_numbers))}
n_nums = len(num_to_col)
idx_to_row = {orig: new for new, orig in enumerate(has_nums_idx)}

rows, cols = [], []
for new_i, orig_i in enumerate(has_nums_idx):
    for n in number_sets[orig_i]:
        rows.append(new_i)
        cols.append(num_to_col[n])

N = len(has_nums_idx)
D = sp.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(N, n_nums),
)

# SVD
k = 64
print(f"\nComputing SVD (k={k})...", flush=True)
U, S, Vt = svds(D.astype(np.float64), k=k)
order = np.argsort(-S)
U = U[:, order].astype(np.float32)

# Normalize rows for cosine similarity
U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)

# Find same-prefix, different-number pairs with high cosine in SVD space
print("\nScanning same-prefix can't-link pairs...", flush=True)
failures = []
n_cantlink_total = 0
n_high_cos = 0

for prefix, idxs in multi_prefix.items():
    # All pairs within this prefix family
    for a_pos in range(len(idxs)):
        for b_pos in range(a_pos + 1, len(idxs)):
            idx_a = idxs[a_pos]
            idx_b = idxs[b_pos]
            nums_a = number_sets[idx_a]
            nums_b = number_sets[idx_b]
            if nums_a & nums_b:
                continue  # shared number — not can't-link
            n_cantlink_total += 1
            row_a = idx_to_row[idx_a]
            row_b = idx_to_row[idx_b]
            cos = float(U_norm[row_a] @ U_norm[row_b])
            if cos > 0.5:
                n_high_cos += 1
                failures.append((cos, idx_a, idx_b, prefix))

failures.sort(reverse=True)
print(f"\n  Total same-prefix can't-link pairs: {n_cantlink_total}", flush=True)
print(
    f"  Pairs with cos > 0.5: {n_high_cos} ({100*n_high_cos/max(n_cantlink_total,1):.1f}%)",
    flush=True,
)
print(
    f"  Pairs with cos > 0.9: {sum(1 for c,_,_,_ in failures if c > 0.9)}", flush=True
)
print(
    f"  Pairs with cos > 0.99: {sum(1 for c,_,_,_ in failures if c > 0.99)}", flush=True
)

print("\nTop 50 same-prefix failures:")
for rank, (cos, idx_a, idx_b, prefix) in enumerate(failures[:50]):
    text_a = distinct_texts[idx_a]
    text_b = distinct_texts[idx_b]
    nums_a = number_sets[idx_a]
    nums_b = number_sets[idx_b]
    prefix_str = " ".join(prefix) if prefix else "(empty)"
    print(f"\n  {rank+1}. cos={cos:.3f}  prefix=[{prefix_str}]")
    print(f"     {text_a}  nums={nums_a}")
    print(f"     {text_b}  nums={nums_b}")

# Which prefixes have the most failures?
print("\n\n=== Prefixes with most failures (cos > 0.5) ===")
prefix_failure_count = Counter()
for cos, idx_a, idx_b, prefix in failures:
    prefix_failure_count[prefix] += 1

for prefix, count in prefix_failure_count.most_common(30):
    family_size = len(multi_prefix[prefix])
    prefix_str = " ".join(prefix) if prefix else "(empty)"
    print(f"  [{prefix_str}]: {count} failures, {family_size} family members")
