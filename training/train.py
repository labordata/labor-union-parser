#!/usr/bin/env python3
"""
Training script for the labor union parser BiLSTM-CRF model.

Usage:
    python training/train.py

This script:
1. Loads training data from training/data/labeled_data.csv
2. Trains the BiLSTM-CRF model
3. Saves the best model to src/labor_union_parser/weights/bilstm_bio_crf.pt
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.model import (
    UnionNameExtractor,
    create_desig_label,
    extract_desig_from_pred,
)
from labor_union_parser.tokenizer import (
    tokenize,
    build_token_vocab,
    build_aff_vocab,
    text_to_token_ids,
    text_to_is_number,
    MAX_TOKEN_LEN,
)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "labeled_data.csv"
WEIGHTS_PATH = (
    SCRIPT_DIR.parent / "src" / "labor_union_parser" / "weights" / "bilstm_bio_crf.pt"
)
PRETRAINED_PATH = (
    SCRIPT_DIR.parent
    / "src"
    / "labor_union_parser"
    / "weights"
    / "pretrained_embeddings.pt"
)

# Training config
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOKEN_EMBED_DIM = 64
HIDDEN_DIM = 512
AFF_EMBED_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10


def load_data(csv_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load training data from consolidated CSV."""
    train_examples = []
    val_examples = []
    test_examples = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = {
                "text": row["text"],
                "aff_abbr": row["aff_abbr"],
                "desig_num": row["desig_num"],
            }
            split = row["split"]
            if split == "train":
                train_examples.append(example)
            elif split == "val":
                val_examples.append(example)
            elif split == "test":
                test_examples.append(example)

    return train_examples, val_examples, test_examples


def collate_batch(batch, token_to_idx, aff_to_idx, max_token_len):
    """Collate a batch of examples for training."""
    token_ids, token_masks, is_number = [], [], []
    aff_labels, desig_labels = [], []
    texts, true_desigs = [], []

    for ex in batch:
        text = ex["text"]
        texts.append(text)
        true_desigs.append(ex["desig_num"])

        tokens = tokenize(text)
        seq_len = min(len(tokens), max_token_len)
        token_ids.append(text_to_token_ids(text, token_to_idx, max_token_len))
        token_masks.append([1.0] * seq_len + [0.0] * (max_token_len - seq_len))
        is_number.append(text_to_is_number(text, max_token_len))

        aff_labels.append(aff_to_idx.get(ex["aff_abbr"], 0))
        desig_labels.append(create_desig_label(text, ex["desig_num"], max_token_len))

    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "token_mask": torch.tensor(token_masks, dtype=torch.float),
        "is_number": torch.tensor(is_number, dtype=torch.long),
        "aff_labels": torch.tensor(aff_labels, dtype=torch.long),
        "desig_labels": torch.tensor(desig_labels, dtype=torch.long),
        "texts": texts,
        "true_desigs": true_desigs,
    }


def evaluate(model, examples, token_to_idx, aff_to_idx, idx_to_aff, device):
    """Evaluate model on a set of examples."""
    model.eval()
    aff_correct = desig_correct = total = 0

    with torch.no_grad():
        for i in range(0, len(examples), BATCH_SIZE):
            batch = collate_batch(
                examples[i : i + BATCH_SIZE], token_to_idx, aff_to_idx, MAX_TOKEN_LEN
            )
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            is_number = batch["is_number"].to(device)
            aff_labels = batch["aff_labels"].to(device)

            results = model(token_ids, token_mask, is_number=is_number)
            aff_preds = results["aff_idx"].cpu().numpy()
            desig_preds = results["desig_pred"].cpu().numpy()
            aff_labels_np = aff_labels.cpu().numpy()

            for j in range(len(batch["texts"])):
                if aff_preds[j] == aff_labels_np[j]:
                    aff_correct += 1

                pred_desig = extract_desig_from_pred(
                    batch["texts"][j], int(desig_preds[j])
                )

                true_desig = batch["true_desigs"][j]
                true_desig_norm = (
                    true_desig.lstrip("0") or "0"
                    if true_desig and true_desig.isdigit()
                    else true_desig
                )

                if pred_desig == true_desig_norm:
                    desig_correct += 1
                total += 1

    return {
        "aff_acc": aff_correct / total if total > 0 else 0,
        "desig_acc": desig_correct / total if total > 0 else 0,
    }


def main():
    print("BiLSTM-CRF Union Name Parser Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {WEIGHTS_PATH}")

    # Load data
    print("\nLoading data...")
    train_examples, val_examples, test_examples = load_data(DATA_PATH)
    print(
        f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}"
    )

    # Build vocabularies
    token_to_idx = build_token_vocab(train_examples + val_examples)
    aff_to_idx, idx_to_aff = build_aff_vocab(train_examples)
    print(f"{len(token_to_idx)} tokens, {len(aff_to_idx)} affiliations")

    # Create model
    model = UnionNameExtractor(
        token_vocab_size=len(token_to_idx),
        num_affs=len(aff_to_idx),
        token_embed_dim=TOKEN_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        aff_embed_dim=AFF_EMBED_DIM,
    )

    # Load pretrained embeddings if available
    if PRETRAINED_PATH.exists():
        print(f"\nLoading pretrained embeddings from {PRETRAINED_PATH}")
        pretrained = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
        pretrained_vocab = pretrained["vocab"]
        pretrained_embed = pretrained["embeddings"]

        # Copy pretrained embeddings for tokens that exist in both vocabs
        with torch.no_grad():
            matched = 0
            for token, idx in token_to_idx.items():
                if token in pretrained_vocab:
                    pretrained_idx = pretrained_vocab[token]
                    model.token_embed.weight[idx] = pretrained_embed[pretrained_idx]
                    matched += 1
        print(f"  Initialized {matched}/{len(token_to_idx)} tokens from pretrained")

    model.to(DEVICE)
    print(f"\n{sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_combined = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_examples))
        train_shuffled = [train_examples[i] for i in perm]

        total_loss = num_batches = 0
        pbar = tqdm(total=len(train_shuffled), desc=f"Epoch {epoch+1}")

        for i in range(0, len(train_shuffled), BATCH_SIZE):
            batch_data = train_shuffled[i : i + BATCH_SIZE]
            batch = collate_batch(
                batch_data,
                token_to_idx,
                aff_to_idx,
                MAX_TOKEN_LEN,
            )
            token_ids = batch["token_ids"].to(DEVICE)
            token_mask = batch["token_mask"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            aff_labels = batch["aff_labels"].to(DEVICE)
            desig_labels = batch["desig_labels"].to(DEVICE)

            optimizer.zero_grad()
            results = model(token_ids, token_mask, is_number, aff_labels, desig_labels)
            loss = results["total_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.update(len(batch_data))
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        pbar.close()

        val_metrics = evaluate(
            model, val_examples, token_to_idx, aff_to_idx, idx_to_aff, DEVICE
        )
        print(f"  Epoch {epoch+1}: loss={total_loss/num_batches:.4f}")
        print(
            f"    Val: aff={100*val_metrics['aff_acc']:.1f}%, desig={100*val_metrics['desig_acc']:.1f}%"
        )

        combined = val_metrics["aff_acc"] * val_metrics["desig_acc"]
        if combined > best_combined:
            best_combined = combined
            print("    New best! Saving...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "token_to_idx": token_to_idx,
                    "aff_to_idx": aff_to_idx,
                    "idx_to_aff": idx_to_aff,
                },
                WEIGHTS_PATH,
            )

    # Final test
    print("\n" + "=" * 60)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model, test_examples, token_to_idx, aff_to_idx, idx_to_aff, DEVICE
    )
    print(f"Test aff_acc: {100*test_metrics['aff_acc']:.2f}%")
    print(f"Test desig_acc: {100*test_metrics['desig_acc']:.2f}%")
    print(f"\nModel saved to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
