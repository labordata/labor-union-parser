#!/usr/bin/env python3
"""
Pretrain token embeddings using Word2Vec-style skip-gram on unlabeled union names.

Usage:
    python training/pretrain_embeddings.py

This creates pretrained embeddings that can be loaded by the main training script.
"""

import csv
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.tokenizer import tokenize

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILES = [
    SCRIPT_DIR.parent / "all_union_names.csv",
    SCRIPT_DIR.parent / "nlrb_union_names.csv",
]
OUTPUT_PATH = (
    SCRIPT_DIR.parent
    / "src"
    / "labor_union_parser"
    / "weights"
    / "pretrained_embeddings.pt"
)

# Config
EMBED_DIM = 64
WINDOW_SIZE = 3
NUM_NEGATIVES = 5
BATCH_SIZE = 4096
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
MIN_COUNT = 2  # Ignore tokens appearing less than this
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_texts(deduplicate=False):
    """Load all union name texts from CSV files."""
    texts = []
    for path in DATA_FILES:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            col = "union_name" if "union_name" in reader.fieldnames else "participant"
            for row in reader:
                text = row[col]
                if text:
                    texts.append(text)

    if deduplicate:
        texts = list(set(texts))

    return texts


def build_vocab(texts, min_count=MIN_COUNT):
    """Build vocabulary from texts, filtering rare tokens."""
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenize(text))

    # Filter by min_count
    vocab = {
        t: i
        for i, (t, c) in enumerate(
            [(t, c) for t, c in token_counts.items() if c >= min_count]
        )
    }

    # Add special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1, **{t: i + 2 for t, i in vocab.items()}}

    return vocab, token_counts


def generate_pairs(texts, vocab, window_size=WINDOW_SIZE):
    """Generate (target, context) pairs for skip-gram training."""
    pairs = []
    unk_id = vocab["<UNK>"]

    for text in texts:
        tokens = tokenize(text)
        token_ids = [vocab.get(t, unk_id) for t in tokens]

        for i, target_id in enumerate(token_ids):
            if target_id == unk_id:
                continue

            # Context window
            start = max(0, i - window_size)
            end = min(len(token_ids), i + window_size + 1)

            for j in range(start, end):
                if i != j and token_ids[j] != unk_id:
                    pairs.append((target_id, token_ids[j]))

    return pairs


class SkipGram(nn.Module):
    """Simple skip-gram model with negative sampling."""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize
        nn.init.xavier_uniform_(self.target_embed.weight)
        nn.init.xavier_uniform_(self.context_embed.weight)

    def forward(self, target_ids, context_ids, negative_ids):
        """
        Compute loss using negative sampling.

        target_ids: [batch]
        context_ids: [batch]
        negative_ids: [batch, num_negatives]
        """
        target_emb = self.target_embed(target_ids)  # [batch, dim]
        context_emb = self.context_embed(context_ids)  # [batch, dim]
        neg_emb = self.context_embed(negative_ids)  # [batch, num_neg, dim]

        # Positive score
        pos_score = (target_emb * context_emb).sum(dim=1)  # [batch]
        pos_loss = F.logsigmoid(pos_score)

        # Negative scores
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(
            2
        )  # [batch, num_neg]
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # [batch]

        return -(pos_loss + neg_loss).mean()


def train_embeddings(texts, vocab, token_counts):
    """Train skip-gram embeddings."""
    print("Generating training pairs...")
    pairs = generate_pairs(texts, vocab)
    print(f"Generated {len(pairs):,} pairs")

    # Build sampling distribution for negative sampling (unigram^0.75)
    vocab_size = len(vocab)
    freq = torch.zeros(vocab_size)
    for token, idx in vocab.items():
        freq[idx] = token_counts.get(token, 0) ** 0.75
    freq = freq / freq.sum()

    # Model
    model = SkipGram(vocab_size, EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        random.shuffle(pairs)
        total_loss = 0
        num_batches = 0

        pbar = tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch = pairs[i : i + BATCH_SIZE]
            target_ids = torch.tensor([p[0] for p in batch], device=DEVICE)
            context_ids = torch.tensor([p[1] for p in batch], device=DEVICE)

            # Sample negatives
            negative_ids = (
                torch.multinomial(freq, len(batch) * NUM_NEGATIVES, replacement=True)
                .view(len(batch), NUM_NEGATIVES)
                .to(DEVICE)
            )

            optimizer.zero_grad()
            loss = model(target_ids, context_ids, negative_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"  Epoch {epoch+1}: avg_loss={total_loss/num_batches:.4f}")

    return model


def main():
    print("Skip-gram Embedding Pretraining")
    print("=" * 50)

    # Load data
    print("Loading texts...")
    texts = load_texts()
    print(f"Loaded {len(texts):,} texts")

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab, token_counts = build_vocab(texts)
    print(f"Vocabulary size: {len(vocab):,}")

    # Train
    model = train_embeddings(texts, vocab, token_counts)

    # Save embeddings
    print(f"\nSaving to {OUTPUT_PATH}...")
    idx_to_token = {i: t for t, i in vocab.items()}
    torch.save(
        {
            "embeddings": model.target_embed.weight.detach().cpu(),
            "vocab": vocab,
            "idx_to_token": idx_to_token,
            "embed_dim": EMBED_DIM,
        },
        OUTPUT_PATH,
    )

    print("Done!")


if __name__ == "__main__":
    main()
