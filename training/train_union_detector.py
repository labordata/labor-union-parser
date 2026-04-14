#!/usr/bin/env python3
"""Train the union vs non-union detector.

Uses PyTorch Lightning for structured training with:
- UnionDetectorModel (LightningModule): FastText+RoPE encoder + learned prototype
- UnionDetectorDataModule (LightningDataModule): data loading, vocab, collation
"""

import json
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F

from labor_union_parser.extractor import UnionDetectorEncoder
from labor_union_parser.tokenizer import NUM_BLOOM_HASHES, tokenize_for_arcface

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
MODEL_PATH = WEIGHTS_DIR / "union_detector.pt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
N_BUCKETS = 50000
EMBED_DIM = 64

F7_DB_PATH = DATA_DIR / "f7.db"
N_EMPLOYER_NEGATIVES = 20000


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    token_ids: torch.Tensor
    ngram_ids: torch.Tensor
    ngram_counts: torch.Tensor
    bloom_ids: torch.Tensor
    is_num: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class UnionDetectorModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = UnionDetectorEncoder(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            n_buckets=N_BUCKETS,
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
        )
        self.union_prototype = torch.nn.Parameter(
            F.normalize(torch.randn(EMBED_DIM), dim=0)
        )

    def forward(self, batch: Batch):
        embeddings = self.encoder(
            batch.token_ids,
            batch.ngram_ids,
            batch.ngram_counts,
            batch.bloom_ids,
            batch.is_num,
            batch.lengths,
        )
        proto = F.normalize(self.union_prototype, dim=0)
        cos_sim = embeddings @ proto

        # Angular margin for union examples
        theta = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        is_union = batch.labels == 1
        margin_cos = torch.where(is_union, torch.cos(theta + ARCFACE_MARGIN), cos_sim)
        logits = ARCFACE_SCALE * margin_cos
        loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
        return loss, cos_sim

    def training_step(self, batch: Batch):
        loss, _ = self(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch):
        embeddings = self.encoder(
            batch.token_ids,
            batch.ngram_ids,
            batch.ngram_counts,
            batch.bloom_ids,
            batch.is_num,
            batch.lengths,
        )
        proto = F.normalize(self.union_prototype, dim=0)
        sims = embeddings @ proto
        preds = (sims > 0.0).float()
        acc = (preds == batch.labels).float().mean()
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _collate_batch(examples) -> Batch:
    B = len(examples)
    max_len = max(ex["length"] for ex in examples)
    max_ngrams = len(examples[0]["ngram_ids"][0]) if examples[0]["ngram_ids"] else 32

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.float)

    for i, ex in enumerate(examples):
        L = ex["length"]
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(ex["token_ids"][:L], dtype=torch.long)
        ngram_ids[i, :L] = torch.tensor(ex["ngram_ids"][:L], dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ex["ngram_counts"][:L], dtype=torch.long)
        bloom_ids[i, :L] = torch.tensor(ex["bloom_ids"][:L], dtype=torch.long)
        is_num[i, :L] = torch.tensor(ex["is_num_f"][:L], dtype=torch.float)
        labels[i] = ex["label"]

    return Batch(
        token_ids=token_ids,
        ngram_ids=ngram_ids,
        ngram_counts=ngram_counts,
        bloom_ids=bloom_ids,
        is_num=is_num,
        lengths=lengths,
        labels=labels,
    )


class UnionDetectorDataModule(L.LightningDataModule):
    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        print("Loading data...")
        data = self._load_data()

        self.train_data = [ex for ex in data if ex["split"] == "train"]
        self.val_data = [ex for ex in data if ex["split"] == "val"]

        self._build_vocab()
        self._encode_all()

        n_train_union = sum(1 for ex in self.train_data if ex["label"] == 1)
        n_val_union = sum(1 for ex in self.val_data if ex["label"] == 1)
        print(
            f"Train: {len(self.train_data)} ({n_train_union} union), "
            f"Val: {len(self.val_data)} ({n_val_union} union)"
        )
        print(f"Vocab: {len(self.vocab)} tokens")

    def _load_data(self):
        with open(DATA_DIR / "training_examples.json") as f:
            all_examples = json.load(f)

        def _is_union(ex):
            return ex["records"] and ex.get("reason_missing_fnum") not in (
                "multi-union",
                "multi-local",
            )

        data = []
        for ex in all_examples:
            is_union = _is_union(ex)
            if ex["split"] not in ("train", "val", "test"):
                continue

            tokens, is_num, ngram_ids, ngram_counts, bloom_ids = tokenize_for_arcface(
                ex["query"]
            )
            if not tokens:
                continue

            # Merge val and test into val for union detector
            split = "val" if ex["split"] in ("val", "test") else "train"

            data.append(
                {
                    "tokens": tokens,
                    "is_num": is_num,
                    "length": len(tokens),
                    "ngram_ids": ngram_ids,
                    "ngram_counts": ngram_counts,
                    "bloom_ids": bloom_ids,
                    "label": 1.0 if is_union else 0.0,
                    "split": split,
                }
            )

        # Add F7 employer names as hard negatives
        f7_path = F7_DB_PATH.resolve()
        if f7_path.exists():
            conn = sqlite3.connect(str(f7_path))
            rows = conn.execute(
                "SELECT DISTINCT employer FROM f7 "
                "WHERE employer IS NOT NULL AND employer != '' "
                "ORDER BY employer"
            ).fetchall()
            conn.close()

            rng = random.Random(42)
            employers = [r[0] for r in rows]
            n_sample = min(N_EMPLOYER_NEGATIVES, len(employers))
            employers = rng.sample(employers, n_sample)

            n_val = n_sample // 10
            for i, text in enumerate(employers):
                tokens, is_num, ngram_ids, ngram_counts, bloom_ids = (
                    tokenize_for_arcface(text)
                )
                if not tokens:
                    continue
                data.append(
                    {
                        "tokens": tokens,
                        "is_num": is_num,
                        "length": len(tokens),
                        "ngram_ids": ngram_ids,
                        "ngram_counts": ngram_counts,
                        "bloom_ids": bloom_ids,
                        "label": 0.0,
                        "split": "val" if i < n_val else "train",
                    }
                )

            print(f"F7 employers: {n_sample} ({n_sample - n_val} train, {n_val} val)")
        else:
            print(f"Warning: {f7_path} not found, skipping employer negatives")

        return data

    def _build_vocab(self):
        counter = Counter()
        for ex in self.train_data:
            for tok in ex["tokens"]:
                counter[tok] += 1
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for tok, count in counter.most_common():
            if count >= 2:
                self.vocab[tok] = len(self.vocab)

    def _encode_all(self):
        for dataset in [self.train_data, self.val_data]:
            for ex in dataset:
                ex["token_ids"] = [self.vocab.get(tok, 1) for tok in ex["tokens"]]
                ex["is_num_f"] = [float(n) for n in ex["is_num"]]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_batch,
            num_workers=0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_batch,
            num_workers=0,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    L.seed_everything(42)

    dm = UnionDetectorDataModule(batch_size=1024)
    dm.setup()

    model = UnionDetectorModel(vocab_size=len(dm.vocab))

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = L.Trainer(
        max_epochs=30,
        enable_progress_bar=True,
        enable_checkpointing=False,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_acc", patience=10, mode="max"
            ),
        ],
    )
    trainer.fit(model, dm)

    # Save
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": {
                k.removeprefix("encoder."): v
                for k, v in state.items()
                if k.startswith("encoder.")
            },
            "union_centroid": model.union_prototype.data.cpu(),
            "vocab": dm.vocab,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "n_buckets": N_BUCKETS,
            "vocab_size": len(dm.vocab),
            "embed_dim": EMBED_DIM,
            "platt_a": 1.0,
            "platt_b": 0.0,
        },
        MODEL_PATH,
    )
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
