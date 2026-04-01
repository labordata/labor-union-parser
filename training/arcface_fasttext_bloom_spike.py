"""ArcFace F_num Classifier with FastText Encoder + Bloom Hash Numbers.

Extends arcface_fasttext_spike.py: number tokens (is_num=True) use bloom hash
embeddings (sum of k independent hash lookups) instead of vocab+n-gram.
This treats numbers as opaque identifiers with near-zero collision probability,
rather than imposing numeric or substring structure.

Non-number tokens still use vocab + character n-gram (FastText-style).
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

print = partial(print, flush=True)  # noqa: A001

sys.path.insert(0, "src")
from labor_union_parser.tokenizer import smart_truncate_nonspace  # noqa: E402

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


NUM_BLOOM_HASHES = 3
BLOOM_TABLE_SIZE = 4096


def tokenize_example(query):
    """Tokenize a query string, returning tokens and is_num lists."""
    tok_dicts = smart_truncate_nonspace(query)
    tokens = []
    is_num = []
    for td in tok_dicts:
        if td["token_type"] == 4:  # pad
            break
        tokens.append(td["token"])
        is_num.append(bool(td["is_num"]))
    return tokens, is_num


def bloom_hash_ids(number_str):
    """Hash a number string into NUM_BLOOM_HASHES table indices.

    Each number gets a unique (with high probability) set of indices
    into a shared embedding table. The embeddings at those indices
    are summed to produce the number's representation.
    """
    # Normalize: strip leading zeros
    normalized = number_str.lstrip("0") or "0"
    ids = []
    for seed in range(NUM_BLOOM_HASHES):
        h = hashlib.md5(f"{seed}:{normalized}".encode()).hexdigest()
        ids.append(int(h, 16) % BLOOM_TABLE_SIZE)
    return ids


def precompute_bloom_ids(tokens, is_num):
    """Precompute bloom hash IDs for all tokens. Non-number tokens get all zeros."""
    all_ids = []
    for tok, is_n in zip(tokens, is_num):
        if is_n and tok:
            all_ids.append(bloom_hash_ids(tok))
        else:
            all_ids.append([0] * NUM_BLOOM_HASHES)
    return all_ids


# ---------------------------------------------------------------------------
# FastText-style character n-gram embedding
# ---------------------------------------------------------------------------

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
MASK32 = 0xFFFFFFFF


def _fnv1a_32(s):
    """FNV-1a 32-bit hash for a string."""
    h = FNV_OFFSET
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * FNV_PRIME) & MASK32
    return h


def token_to_ngram_hashes(token, n_buckets, min_n=3, max_n=6):
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


def precompute_ngram_hashes(tokens, n_buckets, min_n=3, max_n=6, max_ngrams=32):
    """Precompute n-gram hashes for a list of tokens.

    Returns:
        ngram_ids: list of lists of ints (padded to max_ngrams)
        ngram_counts: list of ints (number of actual n-grams per token)
    """
    all_ids = []
    all_counts = []
    for tok in tokens:
        if not tok:  # empty/pad token
            all_ids.append([0] * max_ngrams)
            all_counts.append(0)
            continue
        hashes = token_to_ngram_hashes(tok, n_buckets, min_n, max_n)
        count = min(len(hashes), max_ngrams)
        # Pad or truncate
        padded = (hashes[:max_ngrams] + [0] * max_ngrams)[:max_ngrams]
        all_ids.append(padded)
        all_counts.append(count)
    return all_ids, all_counts


class FastTextEmbedding(nn.Module):
    """FastText-style token embedding: vocab lookup + hashed character n-gram sum.

    Known tokens get a dedicated embedding (like a standard vocab lookup).
    All tokens also get an n-gram embedding averaged from hashed character
    n-grams. The two are summed, so common tokens converge fast via the
    vocab embedding while rare/unseen tokens still get a reasonable
    representation from n-gram sharing.
    """

    def __init__(self, d_model, vocab_size, n_buckets=50000, min_n=3, max_n=6):
        super().__init__()
        self.n_buckets = n_buckets
        self.min_n = min_n
        self.max_n = max_n
        # Vocab embedding: index 0 = padding, index 1 = unknown
        self.vocab_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # N-gram bucket embedding: index 0 = padding
        self.ngram_embed = nn.Embedding(n_buckets + 1, d_model, padding_idx=0)
        nn.init.normal_(self.vocab_embed.weight, std=0.01)
        nn.init.normal_(self.ngram_embed.weight, std=0.01)
        self.vocab_embed.weight.data[0].zero_()
        self.ngram_embed.weight.data[0].zero_()

    def forward(self, token_ids, ngram_ids, ngram_counts):
        """
        Args:
            token_ids: (B, L) int tensor of vocab indices (0=pad, 1=unk)
            ngram_ids: (B, L, max_ngrams) int tensor of bucket indices
            ngram_counts: (B, L) int tensor of actual n-gram counts per token

        Returns:
            (B, L, d_model) combined embeddings
        """
        # Vocab embedding
        word_emb = self.vocab_embed(token_ids)  # (B, L, d_model)

        # N-gram embedding
        shifted = ngram_ids + 1  # shift so 0 maps to padding_idx=0
        mask = torch.arange(
            shifted.shape[-1], device=shifted.device
        ) < ngram_counts.unsqueeze(-1)
        shifted = shifted * mask

        ngram_emb = self.ngram_embed(shifted)  # (B, L, max_ngrams, d_model)
        summed = ngram_emb.sum(dim=2)  # (B, L, d_model)
        counts_safe = ngram_counts.float().clamp(min=1).unsqueeze(-1)
        avg_ngram = summed / counts_safe  # (B, L, d_model)

        return word_emb + avg_ngram


# ---------------------------------------------------------------------------
# Encoder: FastText + RoPE self-attention
# ---------------------------------------------------------------------------


class BloomNumberEmbedding(nn.Module):
    """Bloom hash embedding for number tokens — treats numbers as opaque identifiers."""

    def __init__(self, d_model, table_size=BLOOM_TABLE_SIZE):
        super().__init__()
        self.embed = nn.Embedding(table_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, bloom_ids):
        """
        Args:
            bloom_ids: (..., NUM_BLOOM_HASHES) int tensor of table indices

        Returns:
            (..., d_model) — sum of hash lookups
        """
        return self.embed(bloom_ids).sum(dim=-2)  # sum over the k hashes


class FastTextRoPEEncoder(nn.Module):
    def __init__(
        self, d_model=128, n_heads=4, n_layers=3, n_buckets=50000, vocab_size=2
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.token_embed = FastTextEmbedding(
            d_model, vocab_size=vocab_size, n_buckets=n_buckets
        )
        self.bloom_embed = BloomNumberEmbedding(d_model)
        self.num_flag = nn.Linear(1, d_model)

        self.attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(d_model, d_model),
                        "k_proj": nn.Linear(d_model, d_model),
                        "v_proj": nn.Linear(d_model, d_model),
                        "out_proj": nn.Linear(d_model, d_model),
                        "norm1": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 2),
                            nn.GELU(),
                            nn.Linear(d_model * 2, d_model),
                        ),
                        "norm2": nn.LayerNorm(d_model),
                    }
                )
            )

        self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _rope(x, seq_len):
        """Apply rotary position embeddings: (batch, heads, seq_len, head_dim)."""
        head_dim = x.shape[-1]
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        dim_idx = torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
        freq = 1.0 / (10000.0 ** (dim_idx / head_dim))
        angles = pos * freq
        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(
            -2
        )

    def forward(self, token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths):
        """Encode tokens to sequence of hidden states.

        Args:
            token_ids: (B, L) int — vocab indices
            ngram_ids: (B, L, max_ngrams) int
            ngram_counts: (B, L) int
            bloom_ids: (B, L, NUM_BLOOM_HASHES) int — bloom hash indices for number tokens
            is_num: (B, L) float
            lengths: (B,) int

        Returns:
            h: (B, L, d_model)
        """
        B, L, _ = ngram_ids.shape
        head_dim = self.d_model // self.n_heads

        # FastText embedding for all tokens
        text_emb = self.token_embed(token_ids, ngram_ids, ngram_counts)
        # Bloom hash embedding for number tokens
        num_emb = self.bloom_embed(bloom_ids)

        # Use bloom embedding where is_num, FastText elsewhere
        is_num_mask = is_num.unsqueeze(-1)  # (B, L, 1)
        x = text_emb * (1 - is_num_mask) + num_emb * is_num_mask
        x = x + self.num_flag(is_num.unsqueeze(-1))

        # Padding mask
        pad_mask = torch.arange(L, device=ngram_ids.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9

        for layer in self.attn_layers:
            residual = x
            x = layer["norm1"](x)

            q = layer["q_proj"](x).view(B, L, self.n_heads, head_dim).transpose(1, 2)
            k = layer["k_proj"](x).view(B, L, self.n_heads, head_dim).transpose(1, 2)
            v = layer["v_proj"](x).view(B, L, self.n_heads, head_dim).transpose(1, 2)

            q = self._rope(q, L)
            k = self._rope(k, L)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            scores = scores + attn_mask
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            x = residual + layer["out_proj"](attn_out)

            residual = x
            x = residual + layer["ff"](layer["norm2"](x))

        return self.final_norm(x)


# ---------------------------------------------------------------------------
# ArcFace Heads (unchanged from arcface_fnum_spike.py)
# ---------------------------------------------------------------------------


class FactoredArcFaceClassifier(nn.Module):
    """ArcFace with factored prototypes: W_union + W_desig_name + W_desig_num + W_prefix + W_suffix + W_fnum."""

    def __init__(
        self,
        d_model,
        n_classes,
        field_sizes,
        fnum_field_map,
        fnum_desig_bloom,
        scale=30.0,
        margin=0.0,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin

        self.W_union = nn.Embedding(field_sizes["union_name"] + 1, d_model)
        self.W_desig_name = nn.Embedding(field_sizes["desig_name"] + 1, d_model)
        self.W_prefix = nn.Embedding(field_sizes["prefix"] + 1, d_model)
        self.W_suffix = nn.Embedding(field_sizes["suffix"] + 1, d_model)

        # Bloom hash embedding for desig_num (shared table with encoder side)
        self.bloom_embed = BloomNumberEmbedding(d_model)

        self.W_fnum = nn.Parameter(torch.randn(n_classes, d_model) * 0.01)

        self.register_buffer("field_map", fnum_field_map)
        self.register_buffer(
            "desig_bloom", fnum_desig_bloom
        )  # (n_classes, NUM_BLOOM_HASHES)

        for emb in [
            self.W_union,
            self.W_desig_name,
            self.W_prefix,
            self.W_suffix,
        ]:
            nn.init.normal_(emb.weight, std=0.01)

    def _prototypes(self):
        u = self.W_union(self.field_map[:, 0])
        dn = self.W_desig_name(self.field_map[:, 1])
        pfx = self.W_prefix(self.field_map[:, 2])
        sfx = self.W_suffix(self.field_map[:, 3])

        # Bloom hash for desig_num: (n_classes, NUM_BLOOM_HASHES) -> (n_classes, d_model)
        dnum = self.bloom_embed(self.desig_bloom)

        return u + dn + dnum + pfx + sfx + self.W_fnum

    def forward(self, embeddings, targets=None):
        W = F.normalize(self._prototypes(), dim=1)
        logits = self.scale * F.linear(embeddings, W)
        if targets is None:
            return logits, None

        if self.margin > 0:
            cos_theta = logits / self.scale
            theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
            one_hot = F.one_hot(targets, W.shape[0]).float()
            logits = self.scale * torch.cos(theta + one_hot * self.margin)

        loss = F.cross_entropy(logits, targets)
        return logits, loss


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ArcFaceFastTextModel(nn.Module):
    def __init__(
        self,
        n_classes,
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_buckets=50000,
        vocab_size=2,
        scale=30.0,
        margin=0.5,
        factored_info=None,
    ):
        super().__init__()
        self.encoder = FastTextRoPEEncoder(
            d_model, n_heads, n_layers, n_buckets, vocab_size
        )
        self.arcface = FactoredArcFaceClassifier(
            d_model,
            n_classes,
            field_sizes=factored_info["field_sizes"],
            fnum_field_map=factored_info["field_map"],
            fnum_desig_bloom=factored_info["desig_bloom"],
            scale=scale,
            margin=margin,
        )

    def encode(self, token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths):
        """Encode to L2-normalized embeddings via masked mean-pool."""
        h = self.encoder(token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths)

        L = h.shape[1]
        mask = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)

        return F.normalize(pooled, dim=1)

    def forward(
        self,
        token_ids,
        ngram_ids,
        ngram_counts,
        bloom_ids,
        is_num,
        lengths,
        targets=None,
    ):
        embeddings = self.encode(
            token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths
        )
        return self.arcface(embeddings, targets)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_vocab(data):
    """Build token -> id mapping from training data."""
    counter = Counter()
    for ex in data:
        if ex["split"] == "train":
            for tok in ex["tokens"]:
                counter[tok] += 1

    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, count in counter.most_common():
        if count >= 2:
            vocab[tok] = len(vocab)
    return vocab


def load_data(path, n_buckets, synthetic_path=None):
    """Load training examples, tokenize, and precompute n-gram hashes."""
    with open(path) as f:
        raw = json.load(f)

    raw = [ex for ex in raw if ex.get("source") != "synthetic"]

    if synthetic_path:
        with open(synthetic_path) as f:
            synthetic = json.load(f)
        print(f"Loaded {len(synthetic)} synthetic examples")
        for ex in synthetic:
            ex["split"] = "train"
        raw = raw + synthetic

    data = []
    skipped = 0
    for ex in raw:
        f_num = ex.get("f_num")
        if not f_num or f_num == -100:
            skipped += 1
            continue
        if not ex.get("records"):
            skipped += 1
            continue

        tokens, is_num = tokenize_example(ex["query"])
        if not tokens:
            skipped += 1
            continue

        ngram_ids, ngram_counts = precompute_ngram_hashes(tokens, n_buckets)
        bloom_ids = precompute_bloom_ids(tokens, is_num)

        data.append(
            {
                "tokens": tokens,
                "is_num": is_num,
                "length": len(tokens),
                "f_num": int(f_num),
                "split": ex["split"],
                "source": ex.get("source"),
                "union_name": ex.get("union_name"),
                "record": ex["records"][0] if ex.get("records") else {},
                "ngram_ids": ngram_ids,
                "ngram_counts": ngram_counts,
                "bloom_ids": bloom_ids,
            }
        )

    return data, skipped


def build_fnum_mapping(data):
    fnums = sorted(set(ex["f_num"] for ex in data if ex["split"] == "train"))
    return {f: i for i, f in enumerate(fnums)}


def encode_examples(data, vocab, fnum_to_idx):
    for ex in data:
        ex["token_ids"] = [vocab.get(tok, 1) for tok in ex["tokens"]]
        ex["is_num_f"] = [float(n) for n in ex["is_num"]]
        ex["target"] = fnum_to_idx[ex["f_num"]]


def collate_batch(batch, device):
    max_len = max(ex["length"] for ex in batch)
    max_ngrams = len(batch[0]["ngram_ids"][0])
    B = len(batch)

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num_t = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)

    for i, ex in enumerate(batch):
        L = ex["length"]
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(ex["token_ids"][:L], dtype=torch.long)
        ngram_ids[i, :L] = torch.tensor(ex["ngram_ids"][:L], dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ex["ngram_counts"][:L], dtype=torch.long)
        bloom_ids[i, :L] = torch.tensor(ex["bloom_ids"][:L], dtype=torch.long)
        is_num_t[i, :L] = torch.tensor(ex["is_num_f"], dtype=torch.float)
        targets[i] = ex["target"]

    return (
        token_ids.to(device),
        ngram_ids.to(device),
        ngram_counts.to(device),
        bloom_ids.to(device),
        is_num_t.to(device),
        lengths.to(device),
        targets.to(device),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_fnum_freq(data):
    counts = Counter(
        ex["f_num"]
        for ex in data
        if ex["split"] == "train" and ex.get("source") != "synthetic_mdlm"
    )
    return counts


def bucket_label(count):
    if count == 1:
        return "1"
    elif count <= 5:
        return "2-5"
    elif count <= 15:
        return "6-15"
    else:
        return "16+"


def evaluate(model, data, fnum_freq, device, batch_size=512):
    model.eval()

    buckets = {"1": [0, 0, 0], "2-5": [0, 0, 0], "6-15": [0, 0, 0], "16+": [0, 0, 0]}
    total_top1 = 0
    total_top5 = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            if not batch:
                continue

            (
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
            ) = collate_batch(batch, device)
            logits, _ = model(
                token_ids, ngram_ids, ngram_counts, bloom_ids, is_num_t, lengths
            )

            _, top5_preds = logits.topk(5, dim=1)
            top1_correct = (top5_preds[:, 0] == targets).cpu()
            top5_correct = (top5_preds == targets.unsqueeze(1)).any(dim=1).cpu()

            for i, ex in enumerate(batch):
                freq = fnum_freq.get(ex["f_num"], 0)
                b = bucket_label(freq)
                buckets[b][2] += 1
                if top1_correct[i]:
                    buckets[b][0] += 1
                if top5_correct[i]:
                    buckets[b][1] += 1
                total += 1
                total_top1 += int(top1_correct[i])
                total_top5 += int(top5_correct[i])

    return {
        "top1": total_top1 / max(total, 1),
        "top5": total_top5 / max(total, 1),
        "total": total,
        "buckets": buckets,
    }


def print_results(results, label=""):
    if label:
        print(f"\n--- {label} ---")
    print(
        f"  Overall: top1={results['top1']:.1%}  top5={results['top5']:.1%}  "
        f"(n={results['total']})"
    )
    print(f"  {'Bucket':>8s} | {'Top-1':>8s} | {'Top-5':>8s} | {'Count':>6s}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6}")
    for b in ["1", "2-5", "6-15", "16+"]:
        t1, t5, n = results["buckets"][b]
        if n > 0:
            print(f"  {b:>8s} | {t1/n:>7.1%} | {t5/n:>7.1%} | {n:>6d}")
        else:
            print(f"  {b:>8s} | {'n/a':>8s} | {'n/a':>8s} | {0:>6d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ArcFace with FastText encoder")
    parser.add_argument(
        "--data", type=str, default="training/data/training_examples.json"
    )
    parser.add_argument(
        "--synthetic", type=str, default=None, help="Optional synthetic JSON to merge"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-buckets", type=int, default=50000)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.0)
    parser.add_argument("--save-checkpoint", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    data, skipped = load_data(args.data, args.n_buckets, args.synthetic)
    print(f"Loaded {len(data)} examples with f_num ({skipped} skipped)")

    # Split
    train_data = [ex for ex in data if ex["split"] == "train"]
    test_data = [ex for ex in data if ex["split"] == "test"]
    val_data = [ex for ex in data if ex["split"] == "val"]

    # Build f_num mapping
    fnum_to_idx = build_fnum_mapping(data)
    n_classes = len(fnum_to_idx)
    fnum_freq = compute_fnum_freq(data)

    print(f"Classes: {n_classes} f_nums")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    bucket_counts = Counter(bucket_label(fnum_freq[f]) for f in fnum_to_idx)
    print(f"F_num frequency buckets: {dict(sorted(bucket_counts.items()))}")

    # Filter val/test to only f_nums seen in training
    val_data = [ex for ex in val_data if ex["f_num"] in fnum_to_idx]
    test_data = [ex for ex in test_data if ex["f_num"] in fnum_to_idx]
    print(f"Val (filtered): {len(val_data)}, Test (filtered): {len(test_data)}")

    # Build vocab and encode targets
    vocab = build_vocab(data)
    print(f"Vocab: {len(vocab)} tokens")
    encode_examples(train_data, vocab, fnum_to_idx)
    encode_examples(val_data, vocab, fnum_to_idx)
    encode_examples(test_data, vocab, fnum_to_idx)

    # Build factored field mappings
    field_vocabs = {}
    fnum_records = {}

    for ex in train_data:
        fn = ex["f_num"]
        if fn not in fnum_records and ex.get("union_name"):
            raw_rec = ex.get("record", {})
            rec = {
                "union_name": ex["union_name"],
                "desig_name": raw_rec.get("desig_name", -100),
                "desig_num": raw_rec.get("desig_num", -100),
                "prefix": raw_rec.get("prefix", -100),
                "suffix": raw_rec.get("suffix", -100),
            }
            fnum_records[fn] = rec

    for field in ["union_name", "desig_name", "prefix", "suffix"]:
        vals = sorted(
            set(
                r[field]
                for r in fnum_records.values()
                if r[field] not in (-100, 0, "", None)
            ),
            key=str,
        )
        field_vocabs[field] = {v: i + 1 for i, v in enumerate(vals)}

    field_sizes = {f: len(v) for f, v in field_vocabs.items()}
    print(f"Field sizes: {field_sizes}")

    idx_to_fnum_map = {v: k for k, v in fnum_to_idx.items()}
    field_map = torch.zeros(n_classes, 4, dtype=torch.long)
    desig_bloom_t = torch.zeros(n_classes, NUM_BLOOM_HASHES, dtype=torch.long)

    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        rec = fnum_records.get(fn, {})
        for col, field in enumerate(["union_name", "desig_name", "prefix", "suffix"]):
            val = rec.get(field, -100)
            if val in (-100, 0, "", None):
                field_map[i, col] = 0
            else:
                field_map[i, col] = field_vocabs[field].get(val, 0)

        dnum = rec.get("desig_num", -100)
        if dnum not in (-100, 0, None):
            hashes = bloom_hash_ids(str(int(dnum)))
            for j, h in enumerate(hashes):
                desig_bloom_t[i, j] = h

    factored_info = {
        "field_vocabs": field_vocabs,
        "field_sizes": field_sizes,
        "field_map": field_map,
        "desig_bloom": desig_bloom_t,
    }

    # Model
    model = ArcFaceFastTextModel(
        n_classes=n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_buckets=args.n_buckets,
        vocab_size=len(vocab),
        scale=args.arcface_scale,
        margin=args.arcface_margin,
        factored_info=factored_info,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\n{'Epoch':>7} | {'Loss':>8} | {'Top-1':>7} | {'Top-5':>7} | {'Time':>6}")
    print("-" * 48)

    best_val_top1 = 0.0
    best_state = None
    wait = 0
    n_train = len(train_data)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        indices = list(range(n_train))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            batch = [train_data[i] for i in batch_indices]

            (
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
            ) = collate_batch(batch, device)

            _, loss = model(
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_results = evaluate(model, val_data, fnum_freq, device)
        elapsed = time.time() - t0

        marker = ""
        if val_results["top1"] > best_val_top1:
            best_val_top1 = val_results["top1"]
            import copy

            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(
            f"  {epoch+1:2d}/{args.epochs:2d} | {avg_loss:8.4f} | "
            f"{val_results['top1']:6.1%} | {val_results['top5']:6.1%} | "
            f"{elapsed:5.1f}s{marker}"
        )

        if wait >= args.patience:
            print(f"  Early stopping (no improvement for {args.patience} epochs)")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (val top1={best_val_top1:.1%})")

    # Save checkpoint
    if args.save_checkpoint:
        checkpoint = {
            "state_dict": model.state_dict(),
            "fnum_to_idx": fnum_to_idx,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "n_classes": n_classes,
            "n_buckets": args.n_buckets,
            "arcface_scale": args.arcface_scale,
            "arcface_margin": args.arcface_margin,
            "field_vocabs": factored_info["field_vocabs"],
            "field_sizes": factored_info["field_sizes"],
            "field_map": factored_info["field_map"],
            "desig_bloom": factored_info["desig_bloom"],
        }
        torch.save(checkpoint, args.save_checkpoint)
        print(f"Checkpoint saved to {args.save_checkpoint}")

    # Final test evaluation
    test_results = evaluate(model, test_data, fnum_freq, device)
    print_results(test_results, "Test Set")

    val_results = evaluate(model, val_data, fnum_freq, device)
    print_results(val_results, "Val Set")


if __name__ == "__main__":
    main()
