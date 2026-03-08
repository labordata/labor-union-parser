#!/usr/bin/env python3
"""Compare spectral approaches to can't-link repulsive directions.

1. Eigenvectors of the can't-link Laplacian L^- (on a sample)
2. Right singular vectors of the digit indicator matrix D

Both should give directions that separate texts with non-overlapping numbers.
We compare them by measuring how well they separate known can't-link pairs.
"""

import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, svds

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

# ===================================================================
# Sample for tractability
# ===================================================================
rng = np.random.RandomState(42)
SAMPLE_SIZE = 5000
if len(has_nums_idx) > SAMPLE_SIZE:
    sample_idx = sorted(rng.choice(has_nums_idx, SAMPLE_SIZE, replace=False))
else:
    sample_idx = has_nums_idx

N = len(sample_idx)
print(f"\nSampled {N} texts with numbers", flush=True)

sample_number_sets = [number_sets[i] for i in sample_idx]

# ===================================================================
# Build digit indicator matrix D [N, n_distinct_numbers]
# ===================================================================
num_to_col = {n: i for i, n in enumerate(sorted(all_numbers))}
n_nums = len(num_to_col)

rows, cols = [], []
for i, nums in enumerate(sample_number_sets):
    for n in nums:
        rows.append(i)
        cols.append(num_to_col[n])

D = sp.csr_matrix(
    (np.ones(len(rows), dtype=np.float32), (rows, cols)),
    shape=(N, n_nums),
)
print(f"D matrix: {D.shape}, nnz={D.nnz} ({D.nnz/N:.1f} per text)", flush=True)

# ===================================================================
# Approach 1: SVD of D
# ===================================================================
print("\n=== Approach 1: SVD of D ===", flush=True)
k = 64
t0 = time.time()
# D is sparse, use scipy sparse SVD
U_d, S_d, Vt_d = svds(D.astype(np.float64), k=k)
# Sort by descending singular value
order = np.argsort(-S_d)
U_d = U_d[:, order]
S_d = S_d[order]
Vt_d = Vt_d[order]
t_svd = time.time() - t0
print(f"  SVD time: {t_svd:.1f}s", flush=True)
print(f"  Top singular values: {S_d[:10]}", flush=True)

# U_d: [N, k] — left singular vectors = repulsive directions in text space

# ===================================================================
# Approach 2: Eigenvectors of can't-link Laplacian L^-
# ===================================================================
print("\n=== Approach 2: Can't-link Laplacian eigenvectors ===", flush=True)
t0 = time.time()

# Overlap matrix: K = D @ D^T — sparse, entries > 0 mean shared number
K = D @ D.T  # sparse [N, N]

# Can't-link adjacency: W^- = 1 where K == 0 (both have numbers, no overlap)
# W^- is DENSE (most pairs are can't-link), so we work with the complement.
#
# Key identity: L^- = D^- - W^- where W^-_ij = 1{K_ij == 0}
# Since W^- = 11^T - diag - sign(K) (among texts with numbers),
# L^- = D^- - (11^T - I - sign(K))
# The eigenvectors of L^- with smallest eigenvalues correspond to
# the LARGEST eigenvalues of sign(K) (shifted).
#
# sign(K) has the same sparsity as K, so we can work with it.
# Actually: sign(K) = (K > 0) as float, which equals D @ D^T binarized.

# For L^- eigenvalues: since W^- is dense, we use the complement.
# L^- = (N-1)I - K_binary  (up to diagonal correction)
# where K_binary_ij = 1 if i,j share a number.
# Smallest eigenvectors of L^- = largest eigenvectors of K_binary.

K_binary = K.copy()
K_binary.data[:] = 1.0  # binarize: 1 if any shared number
K_binary.setdiag(0)

# Largest eigenvectors of K_binary ≈ smallest eigenvectors of L^-
eigenvalues, eigenvectors = eigsh(K_binary.astype(np.float64), k=k, which="LM")
# Sort by descending eigenvalue
order = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
t_eig = time.time() - t0
print(f"  Eigenvector time: {t_eig:.1f}s", flush=True)
print(f"  Top eigenvalues: {eigenvalues[:10]}", flush=True)

# ===================================================================
# Compare: how well do the directions separate can't-link pairs?
# ===================================================================
print("\n=== Comparison ===", flush=True)

# Sample random can't-link pairs and shared-number pairs
n_test = 10000
cantlink_pairs = []
shared_pairs = []

attempts = 0
while len(cantlink_pairs) < n_test or len(shared_pairs) < n_test:
    i, j = rng.randint(0, N, size=2)
    if i == j:
        continue
    if sample_number_sets[i] & sample_number_sets[j]:
        if len(shared_pairs) < n_test:
            shared_pairs.append((i, j))
    else:
        if len(cantlink_pairs) < n_test:
            cantlink_pairs.append((i, j))
    attempts += 1
    if attempts > 1000000:
        break

cantlink_pairs = np.array(cantlink_pairs)
shared_pairs = np.array(shared_pairs)
print(
    f"  {len(cantlink_pairs)} can't-link pairs, {len(shared_pairs)} shared-number pairs",
    flush=True,
)


def evaluate_directions(V, name):
    """Evaluate how well direction matrix V separates can't-link vs shared pairs.

    V: [N, k] — each row is the projection of text i onto k directions.
    For can't-link pairs, projections should differ; for shared pairs, more similar.
    """
    # L2 distance in the projected space
    cl_i, cl_j = cantlink_pairs[:, 0], cantlink_pairs[:, 1]
    sh_i, sh_j = shared_pairs[:, 0], shared_pairs[:, 1]

    cl_dist = np.linalg.norm(V[cl_i] - V[cl_j], axis=1)
    sh_dist = np.linalg.norm(V[sh_i] - V[sh_j], axis=1)

    # Cosine similarity in projected space
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    cl_cos = (V_norm[cl_i] * V_norm[cl_j]).sum(axis=1)
    sh_cos = (V_norm[sh_i] * V_norm[sh_j]).sum(axis=1)

    print(f"\n  {name} (k={V.shape[1]}):")
    print(f"    L2 distance — can't-link: {cl_dist.mean():.3f} ± {cl_dist.std():.3f}")
    print(f"    L2 distance — shared:     {sh_dist.mean():.3f} ± {sh_dist.std():.3f}")
    print(f"    Cosine sim  — can't-link: {cl_cos.mean():.3f} ± {cl_cos.std():.3f}")
    print(f"    Cosine sim  — shared:     {sh_cos.mean():.3f} ± {sh_cos.std():.3f}")
    print(
        f"    Separation ratio (L2):    {cl_dist.mean() / (sh_dist.mean() + 1e-8):.3f}"
    )

    # AUC: can we distinguish can't-link from shared using distance?
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate([np.ones(len(cl_dist)), np.zeros(len(sh_dist))])
    scores = np.concatenate([cl_dist, sh_dist])
    auc = roc_auc_score(labels, scores)
    print(f"    AUC (L2 dist): {auc:.3f}")

    scores_cos = np.concatenate(
        [-cl_cos, -sh_cos]
    )  # negate: lower cos = more can't-link
    auc_cos = roc_auc_score(labels, scores_cos)
    print(f"    AUC (cos sim): {auc_cos:.3f}")


# Evaluate both approaches
evaluate_directions(U_d, "SVD of D (left singular vectors)")
evaluate_directions(eigenvectors, "Laplacian eigenvectors (via K_binary)")

# Also try D itself (raw digit indicator) for reference
D_dense = D.toarray()
evaluate_directions(D_dense, "Raw D (digit indicator)")

# Try different k values for SVD
for k_test in [16, 32, 128, 256]:
    if k_test <= k:
        evaluate_directions(U_d[:, :k_test], f"SVD of D (k={k_test})")
    else:
        U_extra, S_extra, _ = svds(D.astype(np.float64), k=k_test)
        order = np.argsort(-S_extra)
        U_extra = U_extra[:, order]
        evaluate_directions(U_extra, f"SVD of D (k={k_test})")
