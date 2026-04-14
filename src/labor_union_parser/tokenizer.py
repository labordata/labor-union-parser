"""Tokenizer and feature extraction for the labor union parser.

Provides:
- smart_truncate_nonspace: tokenize text into non-space tokens with number recovery
- tokenize_for_arcface: full preprocessing for the ArcFace model
- Bloom hashing and FastText n-gram hashing utilities
"""

import hashlib

from .char_tokenizer import tokenize

MAX_TOKENS = 20

# ---------------------------------------------------------------------------
# Bloom hashing for number tokens
# ---------------------------------------------------------------------------

NUM_BLOOM_HASHES = 3
BLOOM_TABLE_SIZE = 4096


def bloom_hash_ids(number_str):
    """Hash a number string into NUM_BLOOM_HASHES table indices.

    Each number gets a unique (with high probability) set of indices
    into a shared embedding table. The embeddings at those indices
    are summed to produce the number's representation.
    """
    normalized = number_str.lstrip("0") or "0"
    ids = []
    for seed in range(NUM_BLOOM_HASHES):
        h = hashlib.md5(f"{seed}:{normalized}".encode()).hexdigest()
        ids.append(int(h, 16) % BLOOM_TABLE_SIZE)
    return ids


# ---------------------------------------------------------------------------
# FastText-style character n-gram hashing
# ---------------------------------------------------------------------------

_FNV_OFFSET = 2166136261
_FNV_PRIME = 16777619
_MASK32 = 0xFFFFFFFF


def _fnv1a_32(s):
    """FNV-1a 32-bit hash for a string."""
    h = _FNV_OFFSET
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * _FNV_PRIME) & _MASK32
    return h


def _token_to_ngram_hashes(token, n_buckets, min_n=3, max_n=6):
    """Compute hashed character n-gram indices for a token.

    Wraps token in <> markers, extracts n-grams of length min_n..max_n,
    hashes each to a bucket index. Also includes the whole-token hash.

    Returns list of bucket indices.
    """
    padded = f"<{token}>"
    hashes = []
    for n in range(min_n, max_n + 1):
        for i in range(len(padded) - n + 1):
            ngram = padded[i : i + n]
            hashes.append(_fnv1a_32(ngram) % n_buckets)
    # Whole token hash
    hashes.append(_fnv1a_32(token) % n_buckets)
    return hashes


# ---------------------------------------------------------------------------
# Core tokenization
# ---------------------------------------------------------------------------


def smart_truncate_nonspace(text, max_tokens=MAX_TOKENS):
    """Tokenize, drop spaces, keep first N tokens, recover lost numbers.

    Returns list of dicts with keys: token, is_num, token_type
    """
    full_tokens, full_is_num, full_token_types = tokenize(text, 999)

    nonspace = []
    for i, tt in enumerate(full_token_types):
        if full_tokens[i] and tt != 2:  # not space, not empty
            nonspace.append(
                {
                    "token": full_tokens[i],
                    "is_num": full_is_num[i],
                    "token_type": tt,
                }
            )

    trunc = nonspace[:max_tokens]

    # Recover lost numbers
    trunc_numbers = {d["token"] for d in trunc if d["is_num"]}
    lost_numbers = [
        d for d in nonspace if d["is_num"] and d["token"] not in trunc_numbers
    ]

    if lost_numbers:
        replace_positions = []
        for i in range(len(trunc) - 1, -1, -1):
            if not trunc[i]["is_num"] and trunc[i]["token"]:
                replace_positions.append(i)
                if len(replace_positions) >= len(lost_numbers):
                    break
        replace_positions.reverse()
        for pos, lost in zip(replace_positions, lost_numbers):
            trunc[pos] = lost

    # Pad
    while len(trunc) < max_tokens:
        trunc.append(
            {
                "token": "",
                "is_num": 0,
                "token_type": 4,
            }
        )

    return trunc


# ---------------------------------------------------------------------------
# ArcFace-specific preprocessing
# ---------------------------------------------------------------------------

DEFAULT_N_BUCKETS = 50000
DEFAULT_MAX_NGRAMS = 32


def tokenize_for_arcface(
    query, n_buckets=DEFAULT_N_BUCKETS, max_ngrams=DEFAULT_MAX_NGRAMS
):
    """Full preprocessing for the ArcFace model.

    Tokenizes a query string and computes all numeric features needed
    for the FastText + Bloom encoder.

    Returns:
        tokens: list of str (non-pad tokens only)
        is_num: list of bool
        ngram_ids: list of lists of int (padded to max_ngrams)
        ngram_counts: list of int
        bloom_ids: list of lists of int (NUM_BLOOM_HASHES per token)
    """
    tok_dicts = smart_truncate_nonspace(query)
    tokens = []
    is_num = []
    for td in tok_dicts:
        if td["token_type"] == 4:  # pad
            break
        tokens.append(td["token"])
        is_num.append(bool(td["is_num"]))

    # FastText n-gram hashes
    ngram_ids = []
    ngram_counts = []
    for tok in tokens:
        if not tok:
            ngram_ids.append([0] * max_ngrams)
            ngram_counts.append(0)
            continue
        hashes = _token_to_ngram_hashes(tok, n_buckets)
        count = min(len(hashes), max_ngrams)
        padded = (hashes[:max_ngrams] + [0] * max_ngrams)[:max_ngrams]
        ngram_ids.append(padded)
        ngram_counts.append(count)

    # Bloom hash IDs for number tokens
    bloom_ids = []
    for tok, is_n in zip(tokens, is_num):
        if is_n and tok:
            bloom_ids.append(bloom_hash_ids(tok))
        else:
            bloom_ids.append([0] * NUM_BLOOM_HASHES)

    return tokens, is_num, ngram_ids, ngram_counts, bloom_ids
