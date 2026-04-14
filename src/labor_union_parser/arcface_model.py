"""Factored ArcFace model for union text → f_num matching.

Encoder and embedding classes are shared between training and inference.
The FactoredPrototypeClassifier and ArcFaceModel classes are inference-only
(the training model in train_arcface_classifier.py adds CRF, union head,
and disagree penalty on top of these shared components).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import BLOOM_TABLE_SIZE, NUM_BLOOM_HASHES

# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------


class FastTextEmbedding(nn.Module):
    """FastText-style token embedding: vocab lookup + averaged character n-gram."""

    def __init__(self, d_model, vocab_size, n_buckets=50000):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.ngram_embed = nn.Embedding(n_buckets + 1, d_model, padding_idx=0)
        nn.init.normal_(self.vocab_embed.weight, std=0.01)
        nn.init.normal_(self.ngram_embed.weight, std=0.01)
        self.vocab_embed.weight.data[0].zero_()
        self.ngram_embed.weight.data[0].zero_()

    def forward(self, token_ids, ngram_ids, ngram_counts):
        word_emb = self.vocab_embed(token_ids)

        shifted = ngram_ids + 1
        mask = torch.arange(
            shifted.shape[-1], device=shifted.device
        ) < ngram_counts.unsqueeze(-1)
        shifted = shifted * mask

        ngram_emb = self.ngram_embed(shifted)
        summed = ngram_emb.sum(dim=2)
        counts_safe = ngram_counts.float().clamp(min=1).unsqueeze(-1)
        avg_ngram = summed / counts_safe

        return word_emb + avg_ngram


class BloomNumberEmbedding(nn.Module):
    """Bloom hash embedding for number tokens."""

    def __init__(self, d_model, table_size=BLOOM_TABLE_SIZE):
        super().__init__()
        self.embed = nn.Embedding(table_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.01)

    def forward(self, bloom_ids):
        return self.embed(bloom_ids).sum(dim=-2)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class FastTextRoPEEncoder(nn.Module):
    """FastText + Bloom + RoPE self-attention encoder."""

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
        B, L, _ = ngram_ids.shape
        head_dim = self.d_model // self.n_heads

        text_emb = self.token_embed(token_ids, ngram_ids, ngram_counts)
        num_emb = self.bloom_embed(bloom_ids)

        is_num_mask = is_num.unsqueeze(-1)
        x = text_emb * (1 - is_num_mask) + num_emb * is_num_mask
        x = x + self.num_flag(is_num.unsqueeze(-1))

        pad_mask = torch.arange(L, device=token_ids.device).unsqueeze(
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
# Factored prototype classifier (inference only — no ArcFace margin)
# ---------------------------------------------------------------------------


class FactoredPrototypeClassifier(nn.Module):
    """Factored prototype classifier with multi-prototype logsumexp aggregation.

    Each f_num's prototype is the sum of field embeddings:
        prototype = W_union + W_desig_name + bloom(desig_num) + W_prefix + W_suffix + W_fnum

    At inference, query embedding is scored against all prototypes via cosine
    similarity, then aggregated to class-level logits via logsumexp.
    """

    def __init__(self, d_model, n_classes, field_sizes):
        super().__init__()
        self.n_classes = n_classes
        self.W_union = nn.Embedding(field_sizes["union_name"] + 1, d_model)
        self.W_desig_name = nn.Embedding(field_sizes["desig_name"] + 1, d_model)
        self.W_prefix = nn.Embedding(field_sizes["prefix"] + 1, d_model)
        self.W_suffix = nn.Embedding(field_sizes["suffix"] + 1, d_model)
        self.bloom_embed = BloomNumberEmbedding(d_model)
        self.W_fnum = nn.Parameter(torch.zeros(n_classes, d_model))

        # Buffers set after loading checkpoint
        self.register_buffer("field_map", torch.zeros(1, 4, dtype=torch.long))
        self.register_buffer(
            "desig_bloom", torch.zeros(1, NUM_BLOOM_HASHES, dtype=torch.long)
        )
        self.register_buffer("proto_to_class", torch.zeros(1, dtype=torch.long))

    def _prototypes(self):
        u = self.W_union(self.field_map[:, 0])
        dn = self.W_desig_name(self.field_map[:, 1])
        pfx = self.W_prefix(self.field_map[:, 2])
        sfx = self.W_suffix(self.field_map[:, 3])
        dnum = self.bloom_embed(self.desig_bloom)
        fnum_emb = self.W_fnum[self.proto_to_class]
        return u + dn + dnum + pfx + sfx + fnum_emb

    def forward(self, embeddings, scale=30.0):
        """Score embeddings against all prototypes.

        Returns (B, n_classes) class logits.
        """
        W = F.normalize(self._prototypes(), dim=1)
        proto_logits = scale * F.linear(embeddings, W)

        # Aggregate multi-prototype classes via numerically stable logsumexp
        B = embeddings.shape[0]
        max_logit = proto_logits.max(dim=1, keepdim=True).values
        shifted = proto_logits - max_logit
        exp_shifted = shifted.exp()
        class_exp = torch.zeros(B, self.n_classes, device=embeddings.device)
        class_exp.scatter_add_(
            1,
            self.proto_to_class.unsqueeze(0).expand(B, -1),
            exp_shifted,
        )
        return class_exp.log() + max_logit


# ---------------------------------------------------------------------------
# Full inference model
# ---------------------------------------------------------------------------


class ArcFaceModel(nn.Module):
    """Factored ArcFace model for union text → f_num matching.

    Combines FastTextRoPEEncoder with FactoredPrototypeClassifier.
    Produces class logits over all known f_nums and union logits from
    the shared union head.
    """

    def __init__(
        self,
        n_classes,
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_buckets=50000,
        vocab_size=2,
        scale=30.0,
        union_scale=10.0,
        field_sizes=None,
    ):
        super().__init__()
        self.scale = scale
        self.encoder = FastTextRoPEEncoder(
            d_model, n_heads, n_layers, n_buckets, vocab_size
        )
        self.classifier = FactoredPrototypeClassifier(d_model, n_classes, field_sizes)

        # Shared union head — uses same W_union weights as prototypes
        self.union_scale = nn.Parameter(torch.tensor(union_scale))

    def encode(self, token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths):
        """Encode to L2-normalized pooled embeddings."""
        h = self.encoder(token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths)
        L = h.shape[1]
        mask = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
        return F.normalize(pooled, dim=1)

    def forward(self, token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths):
        """Full forward pass: encode → classify + union head.

        Returns:
            class_logits: (B, n_classes) f_num logits
            union_logits: (B, n_unions) union classification logits
        """
        embeddings = self.encode(
            token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths
        )
        class_logits = self.classifier(embeddings, scale=self.scale)

        # Union head: shared W_union weights (skip padding at index 0)
        W_u = self.classifier.W_union.weight[1:]
        union_logits = self.union_scale * F.linear(embeddings, F.normalize(W_u, dim=1))

        return class_logits, union_logits
