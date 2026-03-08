#!/usr/bin/env python3
"""Analyze failures of the spectral can't-link approach.

Find can't-link pairs that have high cosine similarity in the SVD(D) space
— i.e., pairs the spectral directions fail to separate.
"""

import sys
from collections import Counter

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

# Build distinct texts and extract number sets
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

# Extract number sets per text
number_sets = []
all_numbers = set()
for text in distinct_texts:
    tokens = _tokenize("union_name", text)
    nums = set()
    for tok in tokens:
        if tok["is_num"]:
            nums.add(tok["token"])
            all_numbers.add(tok["token"])
    number_sets.append(frozenset(nums))

has_nums_idx = [i for i, s in enumerate(number_sets) if s]
print(
    f"  {len(has_nums_idx)} texts with numbers, {len(all_numbers)} distinct numbers",
    flush=True,
)

# Build D for all texts with numbers
num_to_col = {n: i for i, n in enumerate(sorted(all_numbers))}
n_nums = len(num_to_col)

rows, cols = [], []
idx_map = {orig: new for new, orig in enumerate(has_nums_idx)}
for new_i, orig_i in enumerate(has_nums_idx):
    for n in number_sets[orig_i]:
        rows.append(new_i)
        cols.append(num_to_col[n])

N = len(has_nums_idx)
D = sp.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(N, n_nums),
)
print(f"D matrix: {D.shape}, nnz={D.nnz}", flush=True)

# SVD
k = 64
print(f"\nComputing SVD (k={k})...", flush=True)
U, S, Vt = svds(D.astype(np.float64), k=k)
order = np.argsort(-S)
U = U[:, order].astype(np.float32)
S = S[order]

# Normalize rows for cosine similarity
U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)

# Sample can't-link pairs and find high-cosine failures
print("\nSampling can't-link pairs...", flush=True)
rng = np.random.RandomState(42)

n_sample = 500000
failures = []
for _ in range(n_sample):
    i, j = rng.randint(0, N, size=2)
    if i == j:
        continue
    nums_i = number_sets[has_nums_idx[i]]
    nums_j = number_sets[has_nums_idx[j]]
    if nums_i & nums_j:
        continue  # shared number, not can't-link
    cos = float(U_norm[i] @ U_norm[j])
    if cos > 0.5:
        failures.append((cos, has_nums_idx[i], has_nums_idx[j]))

failures.sort(reverse=True)
print(
    f"\n{len(failures)} can't-link pairs with cos > 0.5 (out of {n_sample} samples)",
    flush=True,
)

print("\nTop 50 failures (highest cosine similarity in SVD space):")
for rank, (cos, idx_i, idx_j) in enumerate(failures[:50]):
    text_i = distinct_texts[idx_i]
    text_j = distinct_texts[idx_j]
    nums_i = number_sets[idx_i]
    nums_j = number_sets[idx_j]
    print(f"\n  {rank+1}. cos={cos:.3f}")
    print(f"     {text_i}  nums={nums_i}")
    print(f"     {text_j}  nums={nums_j}")

# Also look at which number tokens are most confused
print("\n\n=== Number token analysis of failures ===")

num_pair_counts = Counter()
for cos, idx_i, idx_j in failures:
    for ni in number_sets[idx_i]:
        for nj in number_sets[idx_j]:
            num_pair_counts[(ni, nj)] += 1

print("\nMost common number pairs in failures:")
for (ni, nj), count in num_pair_counts.most_common(30):
    print(f"  {ni} vs {nj}: {count}")
