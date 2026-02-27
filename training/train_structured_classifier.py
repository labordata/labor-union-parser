#!/usr/bin/env python3
"""Train the structured multi-head classifier.

Uses the StructuredClassifier from the production package with
PyTorch Lightning for training orchestration.
"""

import json
from pathlib import Path

import click
import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from labor_union_parser.classifier import (
    FIELDS,
    MAX_TOKENS,
    POINTER_FIELDS,
    StructuredClassifier,
)
from labor_union_parser.scoring import _get_field_value, build_field_vocabs
from labor_union_parser.tokenizer import smart_truncate_nonspace

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _find_pointer_label(tokens, target_value):
    """Find which token position contains the target value.

    Returns position index (0..MAX_TOKENS-1) or MAX_TOKENS for NONE.
    """
    target_str = str(target_value).strip()
    if not target_str or target_str == "0":
        return MAX_TOKENS
    target_lower = target_str.lower()
    target_num = target_str.lstrip("0") or "0"
    for i, t in enumerate(tokens):
        if t["is_num"] and t["token"] == target_num:
            return i
        if not t["is_num"] and t["token"] == target_lower:
            return i
    return MAX_TOKENS


class StructuredDataset(Dataset):
    def __init__(self, examples, field_vocabs):
        classification_fields = [f for f in FIELDS if f not in POINTER_FIELDS]

        labels = {f: [] for f in classification_fields}
        pointer_labels = {f: [] for f in POINTER_FIELDS}
        self.token_char_ids = []
        self.masks = []
        self.token_strings = []

        for ex in examples:
            rec = ex["records"][0]
            tokens = smart_truncate_nonspace(ex["query"])

            for f in classification_fields:
                val = _get_field_value(rec, f)
                if val == -100:
                    labels[f].append(-100)
                else:
                    idx = field_vocabs[f].get(val)
                    labels[f].append(idx if idx is not None else -100)

            for f in POINTER_FIELDS:
                val = _get_field_value(rec, f)
                if val == -100:
                    pointer_labels[f].append(-100)
                else:
                    pointer_labels[f].append(_find_pointer_label(tokens, val))

            self.token_char_ids.append([t["chars"] for t in tokens])
            self.masks.append([1 if t["token"] else 0 for t in tokens])
            self.token_strings.append([t["token"] for t in tokens])

        self.labels = {f: torch.tensor(v, dtype=torch.long) for f, v in labels.items()}
        self.pointer_labels = {
            f: torch.tensor(v, dtype=torch.long) for f, v in pointer_labels.items()
        }
        self._len = len(examples)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        labels = {f: self.labels[f][idx] for f in FIELDS if f not in POINTER_FIELDS}
        for f in POINTER_FIELDS:
            labels[f] = self.pointer_labels[f][idx]

        return {
            "char_ids": self.token_char_ids[idx],
            "mask": self.masks[idx],
        }, labels


def collate_fn(batch):
    """Collate for token-charcnn: fixed-size (B, MAX_TOKENS, MAX_CHARS_PER_TOKEN)."""
    inputs_list, labels_list = zip(*batch)
    char_ids = torch.tensor([inp["char_ids"] for inp in inputs_list], dtype=torch.long)
    mask = torch.tensor([inp["mask"] for inp in inputs_list], dtype=torch.bool)
    labels = {f: torch.stack([el[f] for el in labels_list]) for f in FIELDS}
    return {"char_ids": char_ids, "mask": mask}, labels


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class StructuredClassifierModule(L.LightningModule):
    def __init__(self, field_sizes, field_vocabs, d_model, n_layers, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = StructuredClassifier(
            field_sizes=field_sizes,
            d_model=d_model,
            n_heads=4,
            n_layers=n_layers,
            ff_dim=d_model * 2,
            dropout=0.1,
        )

        # Precompute null labels for non-null accuracy
        self.null_labels = {
            "prefix": self.hparams.field_vocabs.get("prefix", {}).get(0),
            "suffix": self.hparams.field_vocabs.get("suffix", {}).get(""),
            "desig_name": self.hparams.field_vocabs.get("desig_name", {}).get("LU"),
            "desig_num": self.hparams.field_vocabs.get("desig_num", {}).get(0),
        }

    def forward(self, char_ids, mask):
        return self.model(char_ids, mask)

    def _compute_loss(self, batch):
        inputs, labels = batch
        char_ids = inputs["char_ids"]
        mask = inputs["mask"]
        logits = self(char_ids, mask)

        loss = 0.0
        for f in FIELDS:
            y = labels[f]
            loss = loss + F.cross_entropy(logits[f], y)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._compute_loss(batch)
        self.log("train_loss", loss)

        accs = []
        for f in FIELDS:
            preds = logits[f].argmax(dim=-1)
            y = labels[f]
            valid = y != -100
            if valid.any():
                accs.append((preds[valid] == y[valid]).float().mean())
        if accs:
            self.log("train_mean_acc", torch.stack(accs).mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._compute_loss(batch)
        self.log("val_loss", loss, sync_dist=True)

        accs = []
        for f in FIELDS:
            preds = logits[f].argmax(dim=-1)
            y = labels[f]
            valid = y != -100
            if valid.any():
                acc = (preds[valid] == y[valid]).float().mean()
                self.log(f"val_{f}_acc", acc, prog_bar=False, sync_dist=True)
                accs.append(acc)
        if accs:
            self.log(
                "val_mean_acc", torch.stack(accs).mean(), prog_bar=True, sync_dist=True
            )

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._compute_loss(batch)

        accs = []
        for f in FIELDS:
            preds = logits[f].argmax(dim=-1)
            y = labels[f]
            valid = y != -100
            if not valid.any():
                continue
            acc = (preds[valid] == y[valid]).float().mean()
            self.log(f"test_{f}_acc", acc, sync_dist=True)
            accs.append(acc)

            if f in POINTER_FIELDS:
                nl = MAX_TOKENS
            else:
                nl = self.null_labels.get(f)
            if nl is not None:
                non_null_mask = valid & (y != nl)
                if non_null_mask.any():
                    nn_acc = (preds[non_null_mask] == y[non_null_mask]).float().mean()
                    self.log(f"test_{f}_nonnull_acc", nn_acc, sync_dist=True)
        if accs:
            self.log(
                "test_mean_acc", torch.stack(accs).mean(), prog_bar=True, sync_dist=True
            )

    def configure_optimizers(self):
        head_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if name.startswith("heads."):
                head_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": 0.01},
                {"params": head_params, "weight_decay": 1.0},
            ],
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class StructuredDataModule(L.LightningDataModule):
    def __init__(self, examples_path, batch_size):
        super().__init__()
        self.examples_path = examples_path
        self.batch_size = batch_size
        self.field_vocabs = None
        self.field_sizes = None

    def setup(self, stage=None):
        with open(self.examples_path) as f:
            all_examples = json.load(f)

        splits = {"train": [], "val": [], "test": []}
        for ex in all_examples:
            splits[ex["split"]].append(ex)

        self.field_vocabs = build_field_vocabs(splits["train"])
        self.field_sizes = {f: len(v) for f, v in self.field_vocabs.items()}

        self.train_ds = StructuredDataset(splits["train"], self.field_vocabs)
        self.val_ds = StructuredDataset(splits["val"], self.field_vocabs)
        self.test_ds = StructuredDataset(splits["test"], self.field_vocabs)

        print(
            f"Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, "
            f"Test: {len(self.test_ds)}"
        )
        print("Field sizes:", self.field_sizes)

        # Report pointer label stats
        for f in POINTER_FIELDS:
            ptr_labels = self.train_ds.pointer_labels[f]
            n_none = (ptr_labels == MAX_TOKENS).sum().item()
            n_found = (ptr_labels < MAX_TOKENS).sum().item()
            print(
                f"  {f} pointer: {n_found} found in text, "
                f"{n_none} NONE ({n_none / len(ptr_labels) * 100:.1f}%)"
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--epochs", default=20, help="Number of training epochs")
@click.option("--batch-size", default=256, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--d-model", default=256, help="Model hidden dimension")
@click.option("--n-layers", default=2, help="Number of transformer layers")
def main(epochs, batch_size, lr, d_model, n_layers):
    # Data
    dm = StructuredDataModule(EXAMPLES_PATH, batch_size)
    dm.setup()

    # Model
    module = StructuredClassifierModule(
        field_sizes=dm.field_sizes,
        field_vocabs=dm.field_vocabs,
        d_model=d_model,
        n_layers=n_layers,
        lr=lr,
    )

    n_params = sum(p.numel() for p in module.model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Checkpoint callback — save best by mean validation accuracy
    checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=DATA_DIR,
        filename="structured_classifier",
        monitor="val_mean_acc",
        mode="max",
        save_top_k=1,
    )

    # Train
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )
    trainer.fit(module, dm)

    # Test with best checkpoint
    trainer.test(module, dm, ckpt_path="best")
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
