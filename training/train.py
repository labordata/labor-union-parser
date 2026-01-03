#!/usr/bin/env python3
"""
Training script for the CharCNN-based labor union parser.

Uses CharacterCNN instead of BPE tokenization for typo robustness.

Usage:
    python training/train.py                      # With R-Tuning (UNK labels)
    python training/train.py --no-unk             # Without R-Tuning
    python training/train.py --class-weight       # With class weighting for rare affiliations
    python training/train.py --soft-labels        # With soft labels for related affiliations
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.model import (
    CharCNNExtractor,
    create_desig_label,
    extract_desig_from_pred,
)
from labor_union_parser.char_cnn import (
    tokenize_to_chars,
    get_special_token_id,
)
from labor_union_parser.tokenizer import MAX_TOKEN_LEN

# Affiliation relationships for soft labels
# Format: {child: [(related1, weight1), (related2, weight2), ...]}
# Child gets (1 - sum(weights)), each related affiliation gets its weight
# These represent divisions, mergers, and historical affiliations
AFFILIATION_RELATIONS = {
    # Divisions of LIUNA
    "NPMHU": [
        ("LIUNA", 0.1)
    ],  # National Postal Mail Handlers Union is division of LIUNA
    # Historical splits/mergers
    "WU": [
        ("UNITHE", 0.05),
        ("SEIU", 0.05),
    ],  # Workers United: split from UNITE HERE, now affiliated with SEIU
    "IUE": [("CWA", 0.1)],  # IUE merged with CWA
    "PPPWU": [
        ("IBT", 0.1)
    ],  # GCC (Graphic Communications Conference) merged with then split from IBT
    "GMP": [("USW", 0.1)],  # GMP merged into USW in 2016
    # Note: IBFO (Firemen & Oilers) entries relabeled to SEIU since merger was 1995
    # Add more relationships as needed
}


def build_soft_labels(
    aff: str, aff_to_idx: dict, relations: dict = AFFILIATION_RELATIONS
) -> torch.Tensor:
    """Build soft label distribution for an affiliation.

    Returns a probability distribution where the primary affiliation gets most
    of the weight, but related affiliations get partial credit.
    """
    num_affs = len(aff_to_idx)
    labels = torch.zeros(num_affs)

    if aff not in aff_to_idx:
        return labels

    primary_idx = aff_to_idx[aff]

    if aff in relations:
        total_weight = 0.0
        for related, weight in relations[aff]:
            if related in aff_to_idx:
                labels[aff_to_idx[related]] = weight
                total_weight += weight
        labels[primary_idx] = 1.0 - total_weight
    else:
        labels[primary_idx] = 1.0

    return labels


def build_aff_vocab(
    examples: list[dict], use_unk: bool = True
) -> tuple[dict[str, int], dict[int, str]]:
    """Build affiliation vocabulary.

    Args:
        examples: Training examples
        use_unk: If True, use train_aff field (R-Tuning with UNK labels).
                 If False, use aff_abbr field (ground truth only).
    """
    if use_unk:
        # R-Tuning: includes UNK as a learnable class for uncertain examples
        affs = {
            ex.get("train_aff", ex["aff_abbr"])
            for ex in examples
            if ex.get("train_aff", ex["aff_abbr"])
        }
    else:
        # Ground truth only: no UNK class
        affs = {ex["aff_abbr"] for ex in examples if ex["aff_abbr"]}
    aff_list = sorted(affs)
    aff_to_idx = {a: i for i, a in enumerate(aff_list)}
    idx_to_aff = {i: a for a, i in aff_to_idx.items()}
    return aff_to_idx, idx_to_aff


# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "labeled_data.csv"
WEIGHTS_PATH = (
    SCRIPT_DIR.parent / "src" / "labor_union_parser" / "weights" / "char_cnn.pt"
)

# Training config
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOKEN_EMBED_DIM = 64
HIDDEN_DIM = 512
AFF_EMBED_DIM = 64
CHAR_EMBED_DIM = 16
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
                "aff_abbr": row["aff_abbr"],  # Ground truth (for eval)
                "train_aff": row.get(
                    "train_aff", row["aff_abbr"]
                ),  # Training label (may be UNK)
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


def collate_batch(
    batch, aff_to_idx, max_token_len, use_train_aff=False, use_soft_labels=False
):
    """Collate a batch of examples for training.

    Args:
        use_train_aff: If True, use train_aff for labels (R-Tuning).
                       If False, use aff_abbr (ground truth).
        use_soft_labels: If True, return soft label distributions for related affiliations.
    """
    char_ids_list, token_masks, is_number_list = [], [], []
    token_type_list, special_ids_list = [], []
    aff_labels, aff_soft_labels, desig_labels = [], [], []
    texts, true_desigs, true_affs = [], [], []

    for ex in batch:
        text = ex["text"]
        texts.append(text)
        true_desigs.append(ex["desig_num"])
        true_affs.append(ex["aff_abbr"])  # Always store ground truth for eval

        char_ids, tokens, is_number, token_type = tokenize_to_chars(
            text, max_tokens=max_token_len
        )
        seq_len = sum(1 for t in tokens if t)  # Count non-empty tokens

        # Get special token IDs for non-word tokens
        special_ids = [
            get_special_token_id(t) if tt != 0 else 0
            for t, tt in zip(tokens, token_type)
        ]

        char_ids_list.append(char_ids)
        token_masks.append([1.0] * seq_len + [0.0] * (max_token_len - seq_len))
        is_number_list.append(is_number)
        token_type_list.append(token_type)
        special_ids_list.append(special_ids)

        # Use train_aff (may be UNK) or aff_abbr (ground truth)
        label_aff = (
            ex.get("train_aff", ex["aff_abbr"]) if use_train_aff else ex["aff_abbr"]
        )
        aff_labels.append(aff_to_idx.get(label_aff, 0))
        if use_soft_labels:
            aff_soft_labels.append(build_soft_labels(label_aff, aff_to_idx))
        desig_labels.append(create_desig_label(text, ex["desig_num"], max_token_len))

    result = {
        "char_ids": torch.tensor(char_ids_list, dtype=torch.long),
        "token_mask": torch.tensor(token_masks, dtype=torch.float),
        "is_number": torch.tensor(is_number_list, dtype=torch.long),
        "token_type": torch.tensor(token_type_list, dtype=torch.long),
        "special_ids": torch.tensor(special_ids_list, dtype=torch.long),
        "aff_labels": torch.tensor(aff_labels, dtype=torch.long),
        "desig_labels": torch.tensor(desig_labels, dtype=torch.long),
        "texts": texts,
        "true_desigs": true_desigs,
        "true_affs": true_affs,
    }
    if use_soft_labels:
        result["aff_soft_labels"] = torch.stack(aff_soft_labels)
    return result


def evaluate(model, examples, aff_to_idx, idx_to_aff, device):
    """Evaluate model on a set of examples.

    For R-Tuning, tracks:
    - Overall accuracy (UNK predictions counted as wrong vs ground truth)
    - Confident accuracy (only non-UNK predictions)
    - UNK rate (how often model says "I don't know")
    """
    model.eval()
    aff_correct = desig_correct = total = 0
    unk_predictions = 0
    confident_correct = confident_total = 0

    # Get UNK index if it exists
    unk_idx = aff_to_idx.get("UNK", -1)

    with torch.no_grad():
        for i in range(0, len(examples), BATCH_SIZE):
            batch = collate_batch(
                examples[i : i + BATCH_SIZE],
                aff_to_idx,
                MAX_TOKEN_LEN,
                use_train_aff=False,
            )
            char_ids = batch["char_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            is_number = batch["is_number"].to(device)
            token_type = batch["token_type"].to(device)
            special_ids = batch["special_ids"].to(device)
            aff_labels = batch["aff_labels"].to(device)

            results = model(
                char_ids,
                token_mask,
                is_number=is_number,
                token_type=token_type,
                special_ids=special_ids,
            )
            aff_preds = results["aff_idx"].cpu().numpy()
            desig_preds = results["desig_pred"].cpu().numpy()
            aff_labels_np = aff_labels.cpu().numpy()

            for j in range(len(batch["texts"])):
                pred_idx = aff_preds[j]
                true_idx = aff_labels_np[j]

                # Track UNK predictions
                if pred_idx == unk_idx:
                    unk_predictions += 1
                else:
                    # Confident prediction - check if correct
                    confident_total += 1
                    if pred_idx == true_idx:
                        confident_correct += 1

                # Overall accuracy (UNK = wrong)
                if pred_idx == true_idx:
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
        "confident_acc": (
            confident_correct / confident_total if confident_total > 0 else 0
        ),
        "unk_rate": unk_predictions / total if total > 0 else 0,
        "unk_count": unk_predictions,
    }


def compute_class_weights(
    examples: list[dict], aff_to_idx: dict, use_unk: bool = True
) -> torch.Tensor:
    """Compute inverse frequency class weights for affiliation loss.

    Uses sqrt of inverse frequency to avoid over-weighting very rare classes.
    """
    from collections import Counter

    # Count occurrences of each affiliation
    if use_unk:
        counts = Counter(ex.get("train_aff", ex["aff_abbr"]) for ex in examples)
    else:
        counts = Counter(ex["aff_abbr"] for ex in examples)

    # Compute weights as sqrt(max_count / count) for each class
    max_count = max(counts.values())
    weights = torch.zeros(len(aff_to_idx))
    for aff, idx in aff_to_idx.items():
        count = counts.get(aff, 1)
        weights[idx] = (max_count / count) ** 0.5  # sqrt dampening

    return weights


def main():
    parser = argparse.ArgumentParser(description="Train CharCNN union name parser")
    parser.add_argument(
        "--no-unk",
        action="store_true",
        help="Train without UNK labels (use ground truth only, no R-Tuning)",
    )
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="Use inverse frequency class weighting for rare affiliations",
    )
    parser.add_argument(
        "--soft-labels",
        action="store_true",
        help="Use soft labels for related affiliations (e.g., NPMHU shares credit with LIUNA)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    args = parser.parse_args()
    use_unk = not args.no_unk
    num_epochs = args.epochs

    print("CharCNN Union Name Parser Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {WEIGHTS_PATH}")
    print(f"R-Tuning (UNK labels): {'enabled' if use_unk else 'disabled'}")
    print(f"Class weighting: {'enabled' if args.class_weight else 'disabled'}")
    print(f"Soft labels: {'enabled' if args.soft_labels else 'disabled'}")

    # Load data
    print("\nLoading data...")
    train_examples, val_examples, test_examples = load_data(DATA_PATH)
    print(
        f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}"
    )

    # Build affiliation vocab
    aff_to_idx, idx_to_aff = build_aff_vocab(train_examples, use_unk=use_unk)
    if use_unk:
        unk_count = sum(1 for ex in train_examples if ex.get("train_aff") == "UNK")
        print(f"{len(aff_to_idx)} affiliations (including UNK)")
        print(f"R-Tuning: {unk_count} training examples labeled as UNK")
    else:
        print(f"{len(aff_to_idx)} affiliations (ground truth only, no UNK)")

    # Compute class weights if requested
    aff_weight = None
    if args.class_weight:
        aff_weight = compute_class_weights(
            train_examples, aff_to_idx, use_unk=use_unk
        ).to(DEVICE)
        # Show some example weights
        sorted_affs = sorted(
            aff_to_idx.items(), key=lambda x: aff_weight[x[1]], reverse=True
        )
        print("Top 5 weighted affiliations:")
        for aff, idx in sorted_affs[:5]:
            print(f"  {aff}: {aff_weight[idx]:.2f}")
        print("Bottom 5 weighted affiliations:")
        for aff, idx in sorted_affs[-5:]:
            print(f"  {aff}: {aff_weight[idx]:.2f}")

    # Show soft label relationships if enabled
    if args.soft_labels:
        print("Soft label relationships:")
        for child, related_list in AFFILIATION_RELATIONS.items():
            if child in aff_to_idx:
                related_strs = []
                for related, weight in related_list:
                    if related in aff_to_idx:
                        related_strs.append(f"{related}={weight}")
                    else:
                        related_strs.append(f"{related}=skipped")
                print(f"  {child} → {', '.join(related_strs)}")

    # Create model
    model = CharCNNExtractor(
        num_affs=len(aff_to_idx),
        token_embed_dim=TOKEN_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        aff_embed_dim=AFF_EMBED_DIM,
        char_embed_dim=CHAR_EMBED_DIM,
    )

    model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{num_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\nTraining for {num_epochs} epochs with cosine annealing LR...")
    best_combined = 0

    for epoch in range(num_epochs):
        model.train()
        perm = np.random.permutation(len(train_examples))
        train_shuffled = [train_examples[i] for i in perm]

        total_loss = num_batches = 0
        pbar = tqdm(total=len(train_shuffled), desc=f"Epoch {epoch+1}")

        for i in range(0, len(train_shuffled), BATCH_SIZE):
            batch_data = train_shuffled[i : i + BATCH_SIZE]
            batch = collate_batch(
                batch_data,
                aff_to_idx,
                MAX_TOKEN_LEN,
                use_train_aff=use_unk,
                use_soft_labels=args.soft_labels,
            )

            char_ids = batch["char_ids"].to(DEVICE)
            token_mask = batch["token_mask"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            special_ids = batch["special_ids"].to(DEVICE)
            aff_labels = batch["aff_labels"].to(DEVICE)
            desig_labels = batch["desig_labels"].to(DEVICE)
            aff_soft_labels = batch.get("aff_soft_labels")
            if aff_soft_labels is not None:
                aff_soft_labels = aff_soft_labels.to(DEVICE)

            optimizer.zero_grad()
            results = model(
                char_ids,
                token_mask,
                is_number=is_number,
                token_type=token_type,
                special_ids=special_ids,
                aff_labels=aff_labels,
                desig_labels=desig_labels,
                aff_weight=aff_weight,
                aff_soft_labels=aff_soft_labels,
            )
            loss = results["total_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.update(len(batch_data))
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        pbar.close()

        # Step the learning rate scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        val_metrics = evaluate(model, val_examples, aff_to_idx, idx_to_aff, DEVICE)
        print(
            f"  Epoch {epoch+1}: loss={total_loss/num_batches:.4f}, lr={current_lr:.2e}"
        )
        print(
            f"    Val: aff={100*val_metrics['aff_acc']:.1f}%, desig={100*val_metrics['desig_acc']:.1f}%"
        )
        print(
            f"    Val: confident_acc={100*val_metrics['confident_acc']:.1f}%, unk_rate={100*val_metrics['unk_rate']:.1f}% ({val_metrics['unk_count']} UNK)"
        )

        combined = val_metrics["aff_acc"] * val_metrics["desig_acc"]
        if combined > best_combined:
            best_combined = combined
            print("    New best! Saving...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "aff_to_idx": aff_to_idx,
                    "idx_to_aff": idx_to_aff,
                },
                WEIGHTS_PATH,
            )

    # Final test
    print("\n" + "=" * 60)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_examples, aff_to_idx, idx_to_aff, DEVICE)
    print(f"Test aff_acc: {100*test_metrics['aff_acc']:.2f}%")
    print(f"Test desig_acc: {100*test_metrics['desig_acc']:.2f}%")
    print(f"Test confident_acc: {100*test_metrics['confident_acc']:.2f}%")
    print(
        f"Test unk_rate: {100*test_metrics['unk_rate']:.2f}% ({test_metrics['unk_count']} UNK predictions)"
    )
    print(f"\nModel saved to: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
