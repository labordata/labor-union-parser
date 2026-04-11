"""ArcFace with Multi-Prototype + Union Name Classification Head.

Based on arcface_fasttext_bloom_multiproto_spike.py. Adds an auxiliary
union_name classification head trained jointly with ArcFace. This pushes
the encoder to produce stronger union representations, which should help
cross-union disambiguation.

Phase 2 (--disagree-penalty): adds a penalty when the f_num prediction's
union disagrees with the union head's prediction.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

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


# Token role tags for number disambiguation
TAG_O = 0  # non-number or unmatched number
TAG_DN = 1  # designation number
TAG_PFX = 2  # prefix
TAG_SFX = 3  # suffix
N_TAGS = 4


def generate_token_tags(tokens, is_num, record):
    """Generate per-token role tags by aligning numeric tokens to record fields.

    Returns list of tags (same length as tokens). Non-numeric tokens get TAG_O.
    Numeric tokens are matched to desig_num, prefix, or suffix from the record.
    Ambiguous matches (same number appears in multiple roles) get TAG_O.
    """
    tags = [TAG_O] * len(tokens)

    desig_num = record.get("desig_num", 0)
    prefix = record.get("prefix", 0)
    suffix = record.get("suffix", "")

    # Normalize field values to strings for comparison
    dn_str = (
        str(int(desig_num)) if desig_num and desig_num not in (-100, 0, None) else ""
    )
    pfx_str = str(int(prefix)) if prefix and prefix not in (-100, 0, None) else ""
    # suffix can be a string like "S" or a number
    sfx_str = ""
    if suffix and suffix not in (-100, 0, "", None):
        try:
            sfx_str = str(int(float(suffix)))
        except (ValueError, TypeError):
            sfx_str = ""  # non-numeric suffix, can't tag

    if not dn_str and not pfx_str and not sfx_str:
        return tags  # nothing to tag

    # Collect numeric token indices and their normalized values
    num_tokens = []
    for i, (tok, is_n) in enumerate(zip(tokens, is_num)):
        if is_n:
            normalized = tok.lstrip("0") or "0"
            num_tokens.append((i, normalized))

    # Match from right to left: last number is most likely desig_num,
    # earlier numbers are prefix candidates
    remaining = list(num_tokens)

    # First pass: match desig_num (usually the last or largest number)
    if dn_str:
        dn_normalized = dn_str.lstrip("0") or "0"
        # Try matching from the right
        for j in range(len(remaining) - 1, -1, -1):
            idx, val = remaining[j]
            if val == dn_normalized:
                tags[idx] = TAG_DN
                remaining.pop(j)
                break

    # Second pass: match prefix
    if pfx_str:
        pfx_normalized = pfx_str.lstrip("0") or "0"
        # Try matching from the left (prefix usually comes first)
        for j in range(len(remaining)):
            idx, val = remaining[j]
            if val == pfx_normalized:
                tags[idx] = TAG_PFX
                remaining.pop(j)
                break

    # Third pass: match suffix (rare, usually at end)
    if sfx_str:
        sfx_normalized = sfx_str.lstrip("0") or "0"
        for j in range(len(remaining) - 1, -1, -1):
            idx, val = remaining[j]
            if val == sfx_normalized:
                tags[idx] = TAG_SFX
                remaining.pop(j)
                break

    return tags


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


class MultiProtoArcFaceClassifier(nn.Module):
    """ArcFace with multi-prototype support.

    Each f_num can have multiple prototype rows (e.g., CWA locals with both
    full and short number forms). Scores are aggregated via logsumexp per class.

    Args:
        n_protos: total number of prototype rows (>= n_classes)
        proto_to_class: (n_protos,) int tensor mapping each row to its class index
    """

    def __init__(
        self,
        d_model,
        n_classes,
        n_protos,
        proto_to_class,
        field_sizes,
        fnum_field_map,
        fnum_desig_bloom,
        scale=30.0,
        margin=0.0,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.n_classes = n_classes

        self.W_union = nn.Embedding(field_sizes["union_name"] + 1, d_model)
        self.W_desig_name = nn.Embedding(field_sizes["desig_name"] + 1, d_model)
        self.W_prefix = nn.Embedding(field_sizes["prefix"] + 1, d_model)
        self.W_suffix = nn.Embedding(field_sizes["suffix"] + 1, d_model)

        self.bloom_embed = BloomNumberEmbedding(d_model)

        # Per-class residual (n_classes, not n_protos — shared across variants)
        self.W_fnum = nn.Parameter(torch.randn(n_classes, d_model) * 0.01)

        self.register_buffer("field_map", fnum_field_map)  # (n_protos, 4)
        self.register_buffer(
            "desig_bloom", fnum_desig_bloom
        )  # (n_protos, NUM_BLOOM_HASHES)
        self.register_buffer("proto_to_class", proto_to_class)  # (n_protos,)

        for emb in [self.W_union, self.W_desig_name, self.W_prefix, self.W_suffix]:
            nn.init.normal_(emb.weight, std=0.01)

    def _prototypes(self):
        u = self.W_union(self.field_map[:, 0])
        dn = self.W_desig_name(self.field_map[:, 1])
        pfx = self.W_prefix(self.field_map[:, 2])
        sfx = self.W_suffix(self.field_map[:, 3])
        dnum = self.bloom_embed(self.desig_bloom)

        # W_fnum is indexed by class, not proto row
        fnum_emb = self.W_fnum[self.proto_to_class]

        return u + dn + dnum + pfx + sfx + fnum_emb

    def _aggregate_logits(self, proto_logits):
        """Aggregate proto-level logits to class-level via logsumexp."""
        # Numerically stable: subtract per-example max before exp to avoid overflow
        B = proto_logits.shape[0]
        max_logit = proto_logits.max(dim=1, keepdim=True).values  # (B, 1)
        shifted = proto_logits - max_logit  # all <= 0
        exp_shifted = shifted.exp()
        class_exp = torch.zeros(B, self.n_classes, device=proto_logits.device)
        class_exp.scatter_add_(
            1, self.proto_to_class.unsqueeze(0).expand(B, -1), exp_shifted
        )
        return class_exp.log() + max_logit  # add back the max

    def forward(self, embeddings, targets=None):
        W = F.normalize(self._prototypes(), dim=1)
        proto_logits = self.scale * F.linear(embeddings, W)  # (B, n_protos)
        logits = self._aggregate_logits(proto_logits)  # (B, n_classes)

        if targets is None:
            return logits, None

        valid = targets >= 0
        if not valid.any():
            return logits, torch.tensor(0.0, device=logits.device)

        valid_logits = logits[valid]
        valid_targets = targets[valid]

        if self.margin > 0:
            cos_theta = valid_logits / self.scale
            theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
            one_hot = F.one_hot(valid_targets, self.n_classes).float()
            valid_logits = self.scale * torch.cos(theta + one_hot * self.margin)

        loss = F.cross_entropy(valid_logits, valid_targets)
        return logits, loss


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ArcFaceFastTextModel(nn.Module):
    def __init__(
        self,
        n_classes,
        n_unions,
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
        self.arcface = MultiProtoArcFaceClassifier(
            d_model,
            n_classes,
            n_protos=factored_info["field_map"].shape[0],
            proto_to_class=factored_info["proto_to_class"],
            field_sizes=factored_info["field_sizes"],
            fnum_field_map=factored_info["field_map"],
            fnum_desig_bloom=factored_info["desig_bloom"],
            scale=scale,
            margin=margin,
        )

        # Auxiliary shared heads — gradient flows into prototype embeddings
        self.union_scale = nn.Parameter(torch.tensor(10.0))
        self.desig_scale = nn.Parameter(torch.tensor(10.0))
        # Token role tagging head (number disambiguation)
        self.tag_head = nn.Linear(d_model, N_TAGS)
        # class→field mappings for disagree penalties (set after construction)
        self.class_to_union = None
        self.class_to_desig = None

    def encode(self, token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths):
        """Encode to L2-normalized embeddings via masked mean-pool.

        Returns (pooled_embeddings, hidden_states).
        """
        h = self.encoder(token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths)

        L = h.shape[1]
        mask = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)

        return F.normalize(pooled, dim=1), h

    def forward(
        self,
        token_ids,
        ngram_ids,
        ngram_counts,
        bloom_ids,
        is_num,
        lengths,
        targets=None,
        field_targets=None,
    ):
        embeddings, hidden_states = self.encode(
            token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths
        )
        logits, arcface_loss = self.arcface(embeddings, targets)

        # Auxiliary heads — shared weights with prototypes
        W_u = self.arcface.W_union.weight[1:]
        union_logits = self.union_scale * F.linear(embeddings, F.normalize(W_u, dim=1))
        W_dn = self.arcface.W_desig_name.weight[1:]
        desig_logits = self.desig_scale * F.linear(embeddings, F.normalize(W_dn, dim=1))

        # Token role tagging loss (number disambiguation)
        tag_logits = self.tag_head(hidden_states)  # (B, L, N_TAGS)

        field_losses = {}
        if field_targets is not None:
            for field, flogits in [
                ("union_name", union_logits),
                ("desig_name", desig_logits),
            ]:
                ft = field_targets.get(field)
                if ft is not None:
                    valid = ft >= 0
                    if valid.any():
                        field_losses[field] = F.cross_entropy(flogits[valid], ft[valid])

            # Token role tagging loss
            tt = field_targets.get("token_tags")
            if tt is not None:
                # tag_logits: (B, L, N_TAGS), tt: (B, L) with -100 for ignore
                field_losses["token_tags"] = F.cross_entropy(
                    tag_logits.view(-1, N_TAGS), tt.view(-1), ignore_index=-100
                )

        # Disagree penalties: penalize f_num predictions that disagree
        # with the union head and/or desig_name head (only for examples with valid targets)
        disagree_loss = torch.tensor(0.0, device=logits.device)
        if logits is not None and targets is not None:
            fnum_valid = targets >= 0
            if fnum_valid.any():
                fnum_probs = F.softmax(logits[fnum_valid], dim=1)

                if self.class_to_union is not None:
                    union_log_probs = F.log_softmax(union_logits[fnum_valid], dim=1)
                    union_per_class = union_log_probs[:, self.class_to_union]
                    disagree_loss = (
                        disagree_loss - (fnum_probs * union_per_class).sum(dim=1).mean()
                    )

                if self.class_to_desig is not None:
                    desig_log_probs = F.log_softmax(desig_logits[fnum_valid], dim=1)
                    desig_per_class = desig_log_probs[:, self.class_to_desig]
                    disagree_loss = (
                        disagree_loss - (fnum_probs * desig_per_class).sum(dim=1).mean()
                    )

        return logits, arcface_loss, field_losses, disagree_loss


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
    n_nofnum = 0
    for ex in raw:
        f_num = ex.get("f_num")
        has_fnum = f_num and f_num != -100

        if not has_fnum:
            # Include no-fnum examples only if they have a union_name
            # (for union head training)
            union_name = ex.get("union_name")
            if not union_name or union_name == -100:
                skipped += 1
                continue
        else:
            if not ex.get("records"):
                skipped += 1
                continue

        tokens, is_num = tokenize_example(ex["query"])
        if not tokens:
            skipped += 1
            continue

        ngram_ids, ngram_counts = precompute_ngram_hashes(tokens, n_buckets)
        bloom_ids = precompute_bloom_ids(tokens, is_num)
        record = ex["records"][0] if ex.get("records") else {}
        token_tags = generate_token_tags(tokens, is_num, record)

        data.append(
            {
                "tokens": tokens,
                "is_num": is_num,
                "length": len(tokens),
                "f_num": int(f_num) if has_fnum else -100,
                "split": ex["split"],
                "source": ex.get("source"),
                "union_name": ex.get("union_name"),
                "record": record,
                "ngram_ids": ngram_ids,
                "ngram_counts": ngram_counts,
                "bloom_ids": bloom_ids,
                "token_tags": token_tags,
            }
        )
        if not has_fnum:
            n_nofnum += 1

    return data, skipped, n_nofnum


def build_fnum_mapping(data):
    fnums = sorted(
        set(
            ex["f_num"] for ex in data if ex["split"] == "train" and ex["f_num"] != -100
        )
    )
    return {f: i for i, f in enumerate(fnums)}


def encode_examples(data, vocab, fnum_to_idx, field_vocabs_aux=None):
    """Encode examples with targets for all auxiliary heads.

    field_vocabs_aux: dict with keys union_name, desig_name, prefix, suffix
    mapping field values to indices for classification heads.
    """
    for ex in data:
        ex["token_ids"] = [vocab.get(tok, 1) for tok in ex["tokens"]]
        ex["is_num_f"] = [float(n) for n in ex["is_num"]]
        ex["target"] = fnum_to_idx.get(ex["f_num"], -100)

        rec = ex.get("record", {})
        if field_vocabs_aux:
            # Union target from top-level union_name
            uv = field_vocabs_aux.get("union_name", {})
            ex["union_target"] = uv.get(ex.get("union_name", ""), -1)
            # Field targets from record
            for field in ["desig_name", "prefix", "suffix"]:
                fv = field_vocabs_aux.get(field, {})
                val = rec.get(field, -100)
                if val in (-100, 0, "", None):
                    ex[f"{field}_target"] = -1
                else:
                    ex[f"{field}_target"] = fv.get(val, -1)
        else:
            ex["union_target"] = -1
            for field in ["desig_name", "prefix", "suffix"]:
                ex[f"{field}_target"] = -1


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
    tag_targets = torch.full((B, max_len), -100, dtype=torch.long)
    union_targets = torch.full((B,), -1, dtype=torch.long)
    desig_name_targets = torch.full((B,), -1, dtype=torch.long)
    prefix_targets = torch.full((B,), -1, dtype=torch.long)
    suffix_targets = torch.full((B,), -1, dtype=torch.long)

    for i, ex in enumerate(batch):
        L = ex["length"]
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(ex["token_ids"][:L], dtype=torch.long)
        ngram_ids[i, :L] = torch.tensor(ex["ngram_ids"][:L], dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ex["ngram_counts"][:L], dtype=torch.long)
        bloom_ids[i, :L] = torch.tensor(ex["bloom_ids"][:L], dtype=torch.long)
        is_num_t[i, :L] = torch.tensor(ex["is_num_f"], dtype=torch.float)
        targets[i] = ex["target"]
        if "token_tags" in ex:
            tag_targets[i, :L] = torch.tensor(ex["token_tags"][:L], dtype=torch.long)
        union_targets[i] = ex.get("union_target", -1)
        desig_name_targets[i] = ex.get("desig_name_target", -1)
        prefix_targets[i] = ex.get("prefix_target", -1)
        suffix_targets[i] = ex.get("suffix_target", -1)

    field_targets = {
        "union_name": union_targets.to(device),
        "desig_name": desig_name_targets.to(device),
        "prefix": prefix_targets.to(device),
        "suffix": suffix_targets.to(device),
        "token_tags": tag_targets.to(device),
    }
    return (
        token_ids.to(device),
        ngram_ids.to(device),
        ngram_counts.to(device),
        bloom_ids.to(device),
        is_num_t.to(device),
        lengths.to(device),
        targets.to(device),
        field_targets,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_fnum_freq(data):
    counts = Counter(
        ex["f_num"]
        for ex in data
        if ex["split"] == "train"
        and ex["f_num"] != -100
        and ex.get("source") != "synthetic_mdlm"
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
                _field_targets,
            ) = collate_batch(batch, device)
            logits, _, _, _ = model(
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
    parser.add_argument(
        "--union-weight",
        type=float,
        default=1.0,
        help="Weight for auxiliary union classification loss",
    )
    parser.add_argument(
        "--disagree-penalty",
        type=float,
        default=0.0,
        help="Penalty weight when f_num pred's union disagrees with union head (phase 2)",
    )
    parser.add_argument(
        "--fnum-reg",
        type=float,
        default=0.0,
        help="L2 regularization weight on W_fnum to encourage zero-shot generalization",
    )
    parser.add_argument(
        "--frozen-oov",
        action="store_true",
        help="Add all gazetteer f_nums as OOV classes with W_fnum frozen at zero",
    )
    parser.add_argument(
        "--tag-weight",
        type=float,
        default=0.0,
        help="Weight for token role tagging auxiliary loss",
    )
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
    data, skipped, n_nofnum = load_data(args.data, args.n_buckets, args.synthetic)
    n_with_fnum = len(data) - n_nofnum
    print(
        f"Loaded {n_with_fnum} examples with f_num, {n_nofnum} union-only ({skipped} skipped)"
    )

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

    # Build vocab
    vocab = build_vocab(data)
    print(f"Vocab: {len(vocab)} tokens")

    # Build union vocab for auxiliary head
    # Uses 0-indexed (matching W_union[1:] skip)
    union_names = sorted(
        set(ex.get("union_name", "") for ex in train_data if ex.get("union_name"))
    )
    union_vocab = {name: i for i, name in enumerate(union_names)}
    n_unions = len(union_vocab)
    print(f"Union vocab: {n_unions} unions")

    # Defer encode_examples until field_vocabs are built (need prototype field_vocabs for aux targets)

    # Build factored field mappings
    field_vocabs = {}
    fnum_records = {}  # fn -> primary record
    fnum_all_records = defaultdict(list)  # fn -> list of unique record variants

    for ex in train_data:
        fn = ex["f_num"]
        if fn == -100 or not ex.get("union_name"):
            continue
        raw_rec = ex.get("record", {})
        rec = {
            "union_name": ex["union_name"],
            "desig_name": raw_rec.get("desig_name", -100),
            "desig_num": raw_rec.get("desig_num", -100),
            "prefix": raw_rec.get("prefix", -100),
            "suffix": raw_rec.get("suffix", -100),
        }
        if fn not in fnum_records:
            fnum_records[fn] = rec
        # Track all unique desig_nums per f_num for multi-proto
        dnum = rec["desig_num"]
        if dnum not in (-100, 0, None):
            existing_dnums = {r["desig_num"] for r in fnum_all_records[fn]}
            if dnum not in existing_dnums:
                fnum_all_records[fn].append(rec)

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

    # Build aux head vocabs: convert from 1-indexed field_vocabs to 0-indexed
    # (matching W_field[1:] skip — index 0 in aux = index 1 in prototype)
    field_vocabs_aux = {}
    for field in ["union_name", "desig_name", "prefix", "suffix"]:
        field_vocabs_aux[field] = {v: idx - 1 for v, idx in field_vocabs[field].items()}

    # Now encode examples with all aux targets
    encode_examples(train_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(val_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(test_data, vocab, fnum_to_idx, field_vocabs_aux)

    idx_to_fnum_map = {v: k for k, v in fnum_to_idx.items()}

    # Build prototypes — one per unique record variant per class
    proto_rows = []  # list of (class_idx, field_indices, bloom_hashes)

    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        variants = fnum_all_records.get(fn, [])
        if not variants:
            # Fallback to primary record
            variants = [fnum_records.get(fn, {})]

        seen_hashes = set()
        for rec in variants:
            fields = [0, 0, 0, 0]
            for col, field in enumerate(
                ["union_name", "desig_name", "prefix", "suffix"]
            ):
                val = rec.get(field, -100)
                if val not in (-100, 0, "", None):
                    fields[col] = field_vocabs[field].get(val, 0)

            dnum = rec.get("desig_num", -100)
            hashes = [0] * NUM_BLOOM_HASHES
            if dnum not in (-100, 0, None):
                hashes = bloom_hash_ids(str(int(dnum)))

            hashes_key = tuple(hashes)
            if hashes_key not in seen_hashes:
                seen_hashes.add(hashes_key)
                proto_rows.append((i, fields, hashes))

    n_train_protos = len(proto_rows)
    n_train_classes = n_classes
    n_aliases = n_train_protos - n_classes
    print(
        f"Train prototypes: {n_train_protos} ({n_aliases} variant aliases from {sum(1 for fn in fnum_all_records if len(fnum_all_records[fn]) > 1)} f_nums)"
    )

    # Add frozen OOV prototypes from gazetteer
    if args.frozen_oov:
        gaz_path = Path(args.data).parent / "gazetteer.json"
        with open(gaz_path) as f:
            gazetteer_data = json.load(f)

        n_oov = 0
        for fnum_str, gaz_records in sorted(
            gazetteer_data.items(), key=lambda x: int(x[0])
        ):
            fn = int(fnum_str)
            if fn in fnum_to_idx:
                continue
            class_idx = len(fnum_to_idx)
            fnum_to_idx[fn] = class_idx
            idx_to_fnum_map[class_idx] = fn

            seen_hashes = set()
            for rec in gaz_records:
                fields = [0, 0, 0, 0]
                for col, field_name in enumerate(
                    ["union_name", "desig_name", "prefix", "suffix"]
                ):
                    val = rec.get(field_name, "")
                    if val and val not in (0, -100, None, ""):
                        fields[col] = field_vocabs[field_name].get(val, 0)
                dnum = rec.get("desig_num", 0)
                hashes = [0] * NUM_BLOOM_HASHES
                if dnum and dnum not in (0, -100, None):
                    hashes = bloom_hash_ids(str(int(dnum)))
                hashes_key = (tuple(fields), tuple(hashes))
                if hashes_key not in seen_hashes:
                    seen_hashes.add(hashes_key)
                    proto_rows.append((class_idx, fields, hashes))
            n_oov += 1

        n_classes = len(fnum_to_idx)
        print(
            f"Frozen OOV: {n_oov} f_nums "
            f"({len(proto_rows) - n_train_protos} proto rows)"
        )
        print(f"Total classes: {n_classes}")

    n_protos = len(proto_rows)

    field_map = torch.zeros(n_protos, 4, dtype=torch.long)
    desig_bloom_t = torch.zeros(n_protos, NUM_BLOOM_HASHES, dtype=torch.long)
    proto_to_class = torch.zeros(n_protos, dtype=torch.long)

    for p, (class_idx, fields, hashes) in enumerate(proto_rows):
        proto_to_class[p] = class_idx
        for col in range(4):
            field_map[p, col] = fields[col]
        for j in range(NUM_BLOOM_HASHES):
            desig_bloom_t[p, j] = hashes[j]

    factored_info = {
        "field_vocabs": field_vocabs,
        "field_sizes": field_sizes,
        "field_map": field_map,
        "desig_bloom": desig_bloom_t,
        "proto_to_class": proto_to_class,
    }

    # Model
    model = ArcFaceFastTextModel(
        n_classes=n_classes,
        n_unions=n_unions,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_buckets=args.n_buckets,
        vocab_size=len(vocab),
        scale=args.arcface_scale,
        margin=args.arcface_margin,
        factored_info=factored_info,
    ).to(device)

    # Zero out W_fnum for frozen OOV classes and register gradient hook
    if args.frozen_oov and n_train_classes < n_classes:
        with torch.no_grad():
            model.arcface.W_fnum.data[n_train_classes:].zero_()

        def _zero_oov_grad(grad):
            grad[n_train_classes:] = 0
            return grad

        model.arcface.W_fnum.register_hook(_zero_oov_grad)
        print(f"Frozen W_fnum for OOV classes {n_train_classes}..{n_classes}")

    # Build class→union mapping for disagree penalty
    # Use prototype field_vocabs (1-indexed) → subtract 1 to match W_union[1:] (0-indexed)
    class_to_union = torch.zeros(n_classes, dtype=torch.long)
    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        if fn in fnum_records:
            rec = fnum_records[fn]
        elif args.frozen_oov:
            # OOV class — look up from gazetteer
            gaz_recs = gazetteer_data.get(str(fn), [{}])
            rec = gaz_recs[0] if gaz_recs else {}
        else:
            rec = {}
        un = rec.get("union_name", "")
        proto_un_idx = field_vocabs["union_name"].get(un, 0)
        # field_vocabs is 1-indexed (0=padding), W_union[1:] is 0-indexed
        class_to_union[i] = max(proto_un_idx - 1, 0)
    model.class_to_union = class_to_union.to(device)

    # Desig disagree disabled — not accurate enough to help as a constraint
    # model.class_to_desig remains None
    print(f"Disagree penalty: union only ({n_classes}→{n_unions})")

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
                field_targets,
            ) = collate_batch(batch, device)

            _, arcface_loss, field_losses, disagree_loss = model(
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
                field_targets,
            )
            loss = arcface_loss
            for fname, fl in field_losses.items():
                w = args.tag_weight if fname == "token_tags" else args.union_weight
                loss = loss + w * fl
            if disagree_loss is not None and args.disagree_penalty > 0:
                loss = loss + args.disagree_penalty * disagree_loss
            if args.fnum_reg > 0:
                loss = loss + args.fnum_reg * model.arcface.W_fnum.pow(2).mean()

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
            "vocab": vocab,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "n_classes": n_classes,
            "n_train_classes": n_train_classes,
            "n_buckets": args.n_buckets,
            "arcface_scale": args.arcface_scale,
            "arcface_margin": args.arcface_margin,
            "field_vocabs": factored_info["field_vocabs"],
            "field_sizes": factored_info["field_sizes"],
            "field_map": factored_info["field_map"],
            "desig_bloom": factored_info["desig_bloom"],
            "proto_to_class": factored_info["proto_to_class"],
            "idx_to_fnum": idx_to_fnum_map,
            "union_vocab": union_vocab,
            "n_unions": n_unions,
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
