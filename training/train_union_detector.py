#!/usr/bin/env python3
"""Train the union vs non-union detector.

Uses AttentionPoolingEncoder from the production package with PyTorch Lightning.
Trains CharCNN from scratch (no pretrained weights needed).
"""

from pathlib import Path

import click
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from labor_union_parser.char_cnn import CharacterCNN, tokenize_to_chars
from labor_union_parser.extractor import AttentionPoolingEncoder

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
MODEL_PATH = WEIGHTS_DIR / "union_detector.pt"

MAX_TOKENS = 30


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class UnionDataset(Dataset):
    def __init__(self, texts, labels):
        self.char_ids = []
        self.token_types = []
        self.is_numbers = []
        self.labels = labels

        for text in texts:
            char_ids, _, is_number, token_type = tokenize_to_chars(
                text, max_tokens=MAX_TOKENS
            )
            self.char_ids.append(torch.tensor(char_ids, dtype=torch.long))
            self.token_types.append(torch.tensor(token_type, dtype=torch.long))
            self.is_numbers.append(torch.tensor(is_number, dtype=torch.long))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "char_ids": self.char_ids[idx],
            "token_type": self.token_types[idx],
            "is_number": self.is_numbers[idx],
            "label": self.labels[idx],
        }


def collate_fn(batch):
    return {
        "char_ids": torch.stack([b["char_ids"] for b in batch]),
        "token_type": torch.stack([b["token_type"] for b in batch]),
        "is_number": torch.stack([b["is_number"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Lightning Module — ArcFace with learned union prototype
# ---------------------------------------------------------------------------

ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.5


class UnionDetectorModule(L.LightningModule):
    def __init__(self, lr=1e-3, embed_dim=64):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        char_cnn = CharacterCNN(embed_dim=embed_dim, char_embed_dim=16)
        self.model = AttentionPoolingEncoder(
            char_cnn, embed_dim=embed_dim, num_embed_dim=8, num_heads=4
        )

        # Learned union prototype on the hypersphere
        self.union_prototype = torch.nn.Parameter(
            F.normalize(torch.randn(embed_dim), dim=0)
        )

    def forward(self, char_ids, token_type, is_number):
        return self.model(char_ids, token_type, is_number)

    def training_step(self, batch, batch_idx):
        embeddings = self(batch["char_ids"], batch["token_type"], batch["is_number"])
        labels = batch["label"].float().to(embeddings.device)

        prototype = F.normalize(self.union_prototype, dim=0)
        cos_sim = embeddings @ prototype  # (B,)

        # Angular margin for union examples
        theta = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        is_union = labels == 1
        margin_cos = torch.where(
            is_union,
            torch.cos(theta + ARCFACE_MARGIN),
            cos_sim,
        )

        logits = ARCFACE_SCALE * margin_cos
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = self(batch["char_ids"], batch["token_type"], batch["is_number"])
        labels = batch["label"].to(embeddings.device)

        prototype = F.normalize(self.union_prototype, dim=0)
        cos_sim = embeddings @ prototype

        preds = cos_sim > 0.5
        is_union = labels == 1

        self.log("val_acc", (preds == is_union).float().mean(), prog_bar=True)
        self.log("val_fp", ((preds) & (~is_union)).float().sum(), reduce_fx="sum")
        self.log("val_fn", ((~preds) & (is_union)).float().sum(), reduce_fx="sum")
        self.log("val_mean_sim_union", cos_sim[is_union].mean())
        if (~is_union).any():
            self.log("val_mean_sim_nonunion", cos_sim[~is_union].mean())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class UnionDataModule(L.LightningDataModule):
    def __init__(self, batch_size=256, n_union_samples=0, seed=42):
        super().__init__()
        self.batch_size = batch_size
        self.n_union_samples = n_union_samples
        self.seed = seed

    def setup(self, stage=None):
        import json
        import random

        with open(DATA_DIR / "training_examples.json") as f:
            all_examples = json.load(f)

        # Union: non-empty records; Non-union: empty records
        # Multi-union filings are treated as non-union since they aren't
        # a single identifiable union.
        def _is_union(ex):
            return ex["records"] and ex.get("reason_missing_fnum") != "multi-union"

        train_union = [
            ex["query"]
            for ex in all_examples
            if _is_union(ex) and ex["split"] == "train"
        ]
        test_union = [
            ex["query"]
            for ex in all_examples
            if _is_union(ex) and ex["split"] in ("val", "test")
        ]
        train_nonunion = [
            ex["query"]
            for ex in all_examples
            if not _is_union(ex) and ex["split"] == "train"
        ]
        test_nonunion = [
            ex["query"]
            for ex in all_examples
            if not _is_union(ex) and ex["split"] in ("val", "test")
        ]

        # Subsample union training examples if requested
        if self.n_union_samples and len(train_union) > self.n_union_samples:
            rng = random.Random(self.seed)
            train_union = rng.sample(train_union, self.n_union_samples)

        print(f"Union examples: {len(train_union)} train, {len(test_union)} test")
        print(
            f"Non-union examples: {len(train_nonunion)} train, {len(test_nonunion)} test"
        )

        train_texts = train_union + train_nonunion
        train_labels = [1] * len(train_union) + [0] * len(train_nonunion)
        test_texts = test_union + test_nonunion
        test_labels = [1] * len(test_union) + [0] * len(test_nonunion)

        self.train_ds = UnionDataset(train_texts, train_labels)
        self.test_ds = UnionDataset(test_texts, test_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )


# ---------------------------------------------------------------------------
# Post-training: compute threshold
# ---------------------------------------------------------------------------


def compute_threshold(module, dm):
    """Find optimal threshold using the learned prototype on test set."""
    device = next(module.parameters()).device
    module.eval()

    prototype = F.normalize(module.union_prototype.data, dim=0).to(device)

    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            char_ids = batch["char_ids"].to(device)
            token_type = batch["token_type"].to(device)
            is_number = batch["is_number"].to(device)
            labels = batch["label"]
            embeddings = module(char_ids, token_type, is_number)
            sims = (embeddings @ prototype).cpu().tolist()
            y_true.extend(labels.tolist())
            y_scores.extend(sims)

    y_true, y_scores = np.array(y_true), np.array(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Pick threshold that minimizes total errors (not Youden's J, which
    # ignores class imbalance and over-penalises the minority class).
    n = len(y_true)
    n_pos = (y_true == 1).sum()
    n_neg = n - n_pos
    total_errors = (1 - tpr) * n_pos + fpr * n_neg
    optimal_threshold = float(thresholds[np.argmin(total_errors)])

    accuracy = accuracy_score(y_true, (y_scores > optimal_threshold).astype(int))
    roc_auc = roc_auc_score(y_true, y_scores)

    n_union = int((y_true == 1).sum())
    n_nonunion = int((y_true == 0).sum())
    preds = y_scores > optimal_threshold
    false_neg = int(((y_true == 1) & ~preds).sum())
    false_pos = int(((y_true == 0) & preds).sum())

    print("\nResults:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  False negatives:   {false_neg}/{n_union} ({false_neg/n_union:.4f})")
    print(f"  False positives:   {false_pos}/{n_nonunion} ({false_pos/n_nonunion:.4f})")

    return prototype.cpu(), optimal_threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--epochs", default=30, help="Number of training epochs")
@click.option("--batch-size", default=1024, help="Batch size")
@click.option("--lr", default=1e-3, help="Learning rate")
def main(epochs, batch_size, lr):
    dm = UnionDataModule(batch_size=batch_size)

    module = UnionDetectorModule(lr=lr)

    trainer = L.Trainer(
        max_epochs=epochs,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )
    trainer.fit(module, dm)

    # Compute threshold using learned prototype
    union_centroid, optimal_threshold = compute_threshold(module, dm)

    # Save — union_centroid is the learned prototype, compatible with existing
    # inference code in extractor.py
    torch.save(
        {
            "model_state_dict": module.model.state_dict(),
            "union_centroid": union_centroid,
            "optimal_threshold": optimal_threshold,
        },
        MODEL_PATH,
    )
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
