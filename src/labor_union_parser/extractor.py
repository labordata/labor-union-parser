"""Two-stage extractor for labor union parsing.

Stage 1: Union vs Non-union detection (contrastive similarity to union centroid)
Stage 2: Factored ArcFace model for f_num matching + union head prediction
"""

import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arcface_model import ArcFaceModel, FastTextRoPEEncoder
from .tokenizer import (
    NUM_BLOOM_HASHES,
    tokenize_for_arcface,
)


class UnionDetectorEncoder(nn.Module):
    """FastText+RoPE encoder for union detection.

    Same architecture as the f_num classifier encoder, with mean-pool
    and a projector to the union detection embedding space.
    """

    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=2,
        n_buckets=50000,
        vocab_size=2,
        embed_dim=64,
    ):
        super().__init__()
        self.encoder = FastTextRoPEEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_buckets=n_buckets,
            vocab_size=vocab_size,
        )
        self.projector = nn.Sequential(
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths):
        h = self.encoder(token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths)
        L = h.shape[1]
        mask = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
        pooled = (h * mask.unsqueeze(-1).float()).sum(dim=1) / lengths.unsqueeze(
            1
        ).float().clamp(min=1)
        proj = self.projector(pooled)
        return F.normalize(proj, p=2, dim=-1)


class Extractor:
    """Two-stage extractor for labor union names.

    Stage 1: Union vs Non-union detection using contrastive similarity.
    Stage 2: Factored ArcFace model for f_num matching + union head.

    Output dict keys:
        is_union: bool — detected as union text
        union_score: float — calibrated probability of being a union (0-1)
        union_name: str — predicted parent union name from shared head
        f_num: int — OLMS filing number of best-matching gazetteer record
        match_found: bool — whether match confidence exceeds threshold
        match_score: float — softmax probability of best match (0-1)
    """

    def __init__(
        self,
        device: str | None = None,
        union_threshold: float = 0.5,
    ):
        if device is None:
            self.device = torch.accelerator.current_accelerator(
                check_available=True
            ) or torch.device("cpu")
        else:
            self.device = device

        self.union_threshold = union_threshold
        self._load_models()

    def _load_models(self):
        """Load union detector and ArcFace classifier."""
        weights_dir = Path(__file__).parent / "weights"

        # Stage 1: Union detector
        union_path = weights_dir / "union_detector.pt"
        union_checkpoint = torch.load(
            union_path, map_location=self.device, weights_only=False
        )

        self.union_encoder = UnionDetectorEncoder(
            d_model=union_checkpoint.get("d_model", 128),
            n_heads=union_checkpoint.get("n_heads", 4),
            n_layers=union_checkpoint.get("n_layers", 2),
            n_buckets=union_checkpoint.get("n_buckets", 50000),
            vocab_size=union_checkpoint.get("vocab_size", 2),
            embed_dim=union_checkpoint.get("embed_dim", 64),
        )
        self.union_encoder.load_state_dict(union_checkpoint["model_state_dict"])
        self.union_encoder.to(self.device)
        self.union_encoder.eval()

        self.union_vocab = union_checkpoint.get("vocab")
        self.union_centroid = union_checkpoint["union_centroid"].to(self.device)
        self.platt_a = union_checkpoint.get("platt_a", 1.0)
        self.platt_b = union_checkpoint.get("platt_b", 0.0)

        # Stage 2: ArcFace classifier
        ac_path = weights_dir / "arcface_classifier.pt"
        ac_ckpt = torch.load(ac_path, map_location=self.device, weights_only=False)

        self.arcface_model = ArcFaceModel(
            n_classes=ac_ckpt["n_classes"],
            d_model=ac_ckpt["d_model"],
            n_heads=ac_ckpt.get("n_heads", 4),
            n_layers=ac_ckpt["n_layers"],
            n_buckets=ac_ckpt.get("n_buckets", 50000),
            vocab_size=len(ac_ckpt["vocab"]),
            scale=ac_ckpt.get("arcface_scale", 30.0),
            field_sizes=ac_ckpt["field_sizes"],
        )

        # Remove buffer keys from state dict (they'll be set from bundle)
        sd = {
            k: v
            for k, v in ac_ckpt["state_dict"].items()
            if k
            not in (
                "classifier.field_map",
                "classifier.desig_bloom",
                "classifier.proto_to_class",
            )
        }
        self.arcface_model.load_state_dict(sd, strict=False)

        # Set prototype buffers from bundle (includes OOV prototypes)
        self.arcface_model.classifier.field_map = ac_ckpt["field_map"]
        self.arcface_model.classifier.desig_bloom = ac_ckpt["desig_bloom"]
        self.arcface_model.classifier.proto_to_class = ac_ckpt["proto_to_class"]
        self.arcface_model.to(self.device)
        self.arcface_model.eval()

        # Vocab and index mappings
        self.vocab = ac_ckpt["vocab"]
        self.idx_to_fnum = ac_ckpt["idx_to_fnum"]  # {class_idx: f_num}
        self.match_threshold = ac_ckpt.get("match_threshold", 0.0)
        self.fnum_temperature = ac_ckpt.get("fnum_temperature", 1.0)
        self.union_temperature = ac_ckpt.get("union_temperature", 1.0)

        # Union name vocab (for decoding union head predictions)
        self.union_names = ac_ckpt.get(
            "union_names"
        )  # list of union name strings, indexed by head output

    def _collate_for_union(self, batch_features):
        """Collate tokenized features for union detector (same format as ArcFace)."""
        return self._collate_arcface(batch_features, vocab=self.union_vocab)

    def _collate_arcface(self, batch_features, vocab=None):
        """Collate tokenized features into tensors for ArcFace or union detector."""
        if vocab is None:
            vocab = self.vocab
        B = len(batch_features)
        max_len = max(len(f[0]) for f in batch_features)

        token_ids = torch.zeros(B, max_len, dtype=torch.long)
        max_ngrams = len(batch_features[0][2][0]) if batch_features[0][2] else 32
        ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
        ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
        bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
        is_num_t = torch.zeros(B, max_len, dtype=torch.float)
        lengths = torch.zeros(B, dtype=torch.long)

        for i, (tokens, is_num, ng_ids, ng_counts, bl_ids) in enumerate(batch_features):
            L = len(tokens)
            lengths[i] = L
            token_ids[i, :L] = torch.tensor(
                [vocab.get(tok, 1) for tok in tokens], dtype=torch.long
            )
            ngram_ids[i, :L] = torch.tensor(ng_ids, dtype=torch.long)
            ngram_counts[i, :L] = torch.tensor(ng_counts, dtype=torch.long)
            bloom_ids[i, :L] = torch.tensor(bl_ids, dtype=torch.long)
            is_num_t[i, :L] = torch.tensor(
                [float(n) for n in is_num], dtype=torch.float
            )

        return (
            token_ids.to(self.device),
            ngram_ids.to(self.device),
            ngram_counts.to(self.device),
            bloom_ids.to(self.device),
            is_num_t.to(self.device),
            lengths.to(self.device),
        )

    def extract(self, text: str) -> dict:
        """Extract union fields from a single text."""
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str], batch_size: int = 256) -> list[dict]:
        """Extract from multiple texts."""
        if not texts:
            return []

        if len(texts) <= batch_size:
            return self._extract_batch_internal(texts)

        results = []
        for batch in itertools.batched(texts, batch_size):
            results.extend(self._extract_batch_internal(list(batch)))
        return results

    def _extract_batch_internal(self, texts: list[str]) -> list[dict]:
        """Internal batch processing."""
        # Tokenize once — shared format for both stages
        batch_features = [tokenize_for_arcface(text) for text in texts]

        # Stage 1: Union detection
        (
            union_tk,
            union_ng,
            union_nc,
            union_bl,
            union_isn,
            union_ln,
        ) = self._collate_for_union(batch_features)

        with torch.no_grad():
            union_emb = self.union_encoder(
                union_tk, union_ng, union_nc, union_bl, union_isn, union_ln
            )
            raw_sims = torch.matmul(
                union_emb, self.union_centroid.unsqueeze(0).T
            ).squeeze(-1)
            # Platt-scaled probability: sigmoid(a * sim + b)
            union_probs = torch.sigmoid(self.platt_a * raw_sims + self.platt_b)

        union_scores_list = union_probs.cpu().tolist()

        # Stage 2: ArcFace classification
        (
            token_ids,
            ngram_ids,
            ngram_counts,
            bloom_ids,
            is_num_t,
            lengths,
        ) = self._collate_arcface(batch_features)

        with torch.no_grad():
            class_logits, union_logits = self.arcface_model(
                token_ids, ngram_ids, ngram_counts, bloom_ids, is_num_t, lengths
            )

        # Match scores: temperature-scaled softmax probability of top class
        class_probs = F.softmax(class_logits / self.fnum_temperature, dim=1)
        top_probs, top_indices = class_probs.max(dim=1)

        # Union head predictions (temperature-scaled)
        union_probs = F.softmax(union_logits / self.union_temperature, dim=1)
        union_preds = union_probs.argmax(dim=1).cpu().tolist()

        results = []
        for i in range(len(texts)):
            class_idx = top_indices[i].item()
            match_score = top_probs[i].item()
            f_num = self.idx_to_fnum[class_idx]

            # Union name from shared head
            union_idx = union_preds[i]
            if self.union_names and union_idx < len(self.union_names):
                union_name = self.union_names[union_idx]
            else:
                union_name = ""

            results.append(
                {
                    "is_union": union_scores_list[i] >= self.union_threshold,
                    "union_score": union_scores_list[i],
                    "union_name": union_name,
                    "f_num": f_num,
                    "match_found": match_score >= self.match_threshold,
                    "match_score": match_score,
                }
            )

        return results
