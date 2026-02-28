#!/usr/bin/env python3
"""Train the union vs non-union detector.

Uses CrossAttentionEncoder from the production package with PyTorch Lightning.
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
from labor_union_parser.extractor import CrossAttentionEncoder

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
MODEL_PATH = WEIGHTS_DIR / "union_detector.pt"

MAX_TOKENS = 80  # union detector uses longer sequences than structured classifier


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class UnionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        char_ids, _, is_number, token_type = tokenize_to_chars(
            self.texts[idx], max_tokens=MAX_TOKENS
        )
        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
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
# Loss
# ---------------------------------------------------------------------------


def one_class_contrastive_loss(embeddings, labels, temperature=0.1):
    """One-class contrastive loss: only union examples form positive pairs."""
    device = embeddings.device
    batch_size = embeddings.shape[0]

    sim = torch.matmul(embeddings, embeddings.T) / temperature

    labels_row = labels.unsqueeze(0)
    labels_col = labels.unsqueeze(1)
    pos_mask = ((labels_row == 1) & (labels_col == 1)).float()

    eye = torch.eye(batch_size, device=device)
    pos_mask = pos_mask * (1 - eye)

    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max

    exp_sim = torch.exp(sim) * (1 - eye)
    denom = exp_sim.sum(dim=1, keepdim=True)

    log_prob = sim - torch.log(denom + 1e-8)

    num_pos = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)

    union_mask = (labels == 1) & (num_pos > 0)
    if union_mask.sum() > 0:
        loss = -mean_log_prob_pos[union_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class UnionDetectorModule(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        char_cnn = CharacterCNN(embed_dim=64, char_embed_dim=16)
        self.model = CrossAttentionEncoder(
            char_cnn, embed_dim=64, num_embed_dim=8, num_heads=4
        )

    def forward(self, char_ids, token_type, is_number):
        return self.model(char_ids, token_type, is_number)

    def training_step(self, batch, batch_idx):
        embeddings = self(batch["char_ids"], batch["token_type"], batch["is_number"])
        loss = one_class_contrastive_loss(embeddings, batch["label"])
        if torch.isnan(loss):
            return None
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class UnionDataModule(L.LightningDataModule):
    def __init__(self, batch_size=128, n_union_samples=10000, seed=42):
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
        train_union = [
            ex["query"]
            for ex in all_examples
            if ex["records"] and ex["split"] == "train"
        ]
        test_union = [
            ex["query"]
            for ex in all_examples
            if ex["records"] and ex["split"] in ("val", "test")
        ]
        train_nonunion = [
            ex["query"]
            for ex in all_examples
            if not ex["records"] and ex["split"] == "train"
        ]
        test_nonunion = [
            ex["query"]
            for ex in all_examples
            if not ex["records"] and ex["split"] in ("val", "test")
        ]

        # Subsample union training examples if needed
        if len(train_union) > self.n_union_samples:
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
# Post-training: compute centroid and threshold
# ---------------------------------------------------------------------------


def compute_centroid_and_threshold(module, dm):
    """Compute union centroid from training data and optimal threshold on test."""
    device = next(module.parameters()).device
    module.eval()

    # Compute centroid from training union embeddings
    union_embs = []
    with torch.no_grad():
        for batch in dm.train_dataloader():
            char_ids = batch["char_ids"].to(device)
            token_type = batch["token_type"].to(device)
            is_number = batch["is_number"].to(device)
            labels = batch["label"]
            embeddings = module(char_ids, token_type, is_number)
            for i, label in enumerate(labels.tolist()):
                if label == 1:
                    union_embs.append(embeddings[i].cpu())

    union_centroid = F.normalize(torch.stack(union_embs).mean(dim=0), p=2, dim=0).to(
        device
    )

    # Find optimal threshold on test set
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            char_ids = batch["char_ids"].to(device)
            token_type = batch["token_type"].to(device)
            is_number = batch["is_number"].to(device)
            labels = batch["label"]
            embeddings = module(char_ids, token_type, is_number)
            sims = torch.matmul(embeddings, union_centroid.unsqueeze(0).T).squeeze(-1)
            y_true.extend(labels.tolist())
            y_scores.extend(sims.cpu().tolist())

    y_true, y_scores = np.array(y_true), np.array(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_threshold = float(thresholds[np.argmax(tpr - fpr)])

    accuracy = accuracy_score(y_true, (y_scores > optimal_threshold).astype(int))
    roc_auc = roc_auc_score(y_true, y_scores)

    print("\nResults:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  ROC-AUC:           {roc_auc:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")

    return union_centroid.cpu(), optimal_threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--epochs", default=30, help="Number of training epochs")
@click.option("--batch-size", default=128, help="Batch size")
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

    # Compute centroid and threshold
    union_centroid, optimal_threshold = compute_centroid_and_threshold(module, dm)

    # Save
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
