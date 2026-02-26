#!/usr/bin/env python3
"""Post-hoc temperature scaling for the structured classifier.

Loads a trained checkpoint, freezes all weights, and fits per-head
scalar temperature parameters on the validation set by minimizing NLL
(standard Platt scaling).
"""

import json
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_structured_classifier import (
    StructuredDataset,
    collate_fn,
)

from labor_union_parser.classifier import (
    FIELDS,
    POINTER_FIELDS,
    StructuredClassifier,
)

DATA_DIR = Path(__file__).parent / "data"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"

DEVICE = torch.accelerator.current_accelerator() or torch.device("cpu")


def load_model(ckpt):
    """Load trained structured classifier from checkpoint."""
    model = StructuredClassifier(
        field_sizes=ckpt["field_sizes"],
        d_model=ckpt["d_model"],
        n_heads=4,
        n_layers=ckpt["n_layers"],
        ff_dim=ckpt["d_model"] * 2,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@click.command()
@click.option(
    "--lr", default=0.01, type=float, help="Learning rate for temperature fitting"
)
@click.option("--steps", default=500, type=int, help="Optimization steps")
@click.option("--batch-size", default=256, type=int)
def main(lr, steps, batch_size):
    print(f"Device: {DEVICE}")

    # Load checkpoint
    weights_dir = (
        Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
    )
    ckpt = torch.load(
        weights_dir / "structured_classifier.pt",
        weights_only=False,
        map_location=DEVICE,
    )

    model = load_model(ckpt)
    field_vocabs = ckpt["field_vocabs"]

    # Freeze all model weights
    for param in model.parameters():
        param.requires_grad = False

    # Load data
    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    val_examples = [ex for ex in all_examples if ex["split"] == "val" and ex["records"]]
    print(f"Val examples: {len(val_examples)}")

    val_ds = StructuredDataset(val_examples, field_vocabs)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Collect all logits on val set (frozen model)
    print("Collecting logits on validation set...")
    all_logits = {f: [] for f in FIELDS}
    all_labels = {f: [] for f in FIELDS}

    with torch.no_grad():
        for inputs, labels in val_loader:
            char_ids = inputs["char_ids"].to(DEVICE)
            mask = inputs["mask"].to(DEVICE)
            logits = model(char_ids, mask)

            for f in FIELDS:
                all_logits[f].append(logits[f])
                all_labels[f].append(labels[f].to(DEVICE))

    all_logits = {f: torch.cat(v) for f, v in all_logits.items()}
    all_labels = {f: torch.cat(v) for f, v in all_labels.items()}

    # Replace -inf in pointer logits with large finite negative
    # so that dividing by temperature doesn't produce NaN gradients
    for f in POINTER_FIELDS:
        all_logits[f] = all_logits[f].clamp(min=-1e9)

    # Learnable parameters: scalar log-temperature per head
    log_temperatures = {
        f: torch.nn.Parameter(torch.zeros(1, device=DEVICE)) for f in FIELDS
    }

    all_params = list(log_temperatures.values())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    print(f"\nFitting temperatures (Adam, {steps} steps)...")
    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for f in FIELDS:
            temp = log_temperatures[f].exp()
            scaled_logits = all_logits[f] / temp
            loss = F.cross_entropy(scaled_logits, all_labels[f])
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: loss={total_loss.item():.4f}")
            for f in FIELDS:
                print(f"    {f}: T={log_temperatures[f].exp().item():.4f}")

    # Final results
    print("\n" + "=" * 60)
    print("Final temperatures:")
    print("=" * 60)
    final_temps = {}
    for f in FIELDS:
        t = log_temperatures[f].exp().item()
        final_temps[f] = t
        print(f"  {f}: T={t:.4f}")

    # Per-head NLL before/after
    print("\n" + "=" * 60)
    print("Per-head NLL (before / after temperature scaling):")
    print("=" * 60)
    for f in FIELDS:
        nll_before = F.cross_entropy(all_logits[f], all_labels[f]).item()
        temp = log_temperatures[f].exp()
        nll_after = F.cross_entropy(all_logits[f] / temp, all_labels[f]).item()
        print(
            f"  {f}: {nll_before:.4f} -> {nll_after:.4f} (delta={nll_after - nll_before:+.4f})"
        )

    # Save temperatures
    out_path = DATA_DIR / "temperatures.json"
    with open(out_path, "w") as f_out:
        json.dump(final_temps, f_out, indent=2)
    print(f"\nTemperatures saved to {out_path}")


if __name__ == "__main__":
    main()
