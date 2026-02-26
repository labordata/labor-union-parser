#!/usr/bin/env python3
"""Bundle trained model, gazetteer, and fnum counts into a single checkpoint.

Reads the Lightning checkpoint from training, extracts model weights and
hyperparameters, then combines with the gazetteer and fnum training counts
so the production Extractor only needs to load one file.
"""

import json
from collections import Counter
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"


def main():
    # Load Lightning checkpoint
    # Find the latest versioned checkpoint
    ckpt_files = sorted(
        DATA_DIR.glob("structured_classifier-v*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if ckpt_files:
        ckpt_path = ckpt_files[-1]
    else:
        ckpt_path = DATA_DIR / "structured_classifier.ckpt"
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, weights_only=False)

    # Extract model weights (strip "model." prefix from Lightning state_dict)
    model_state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }

    # Extract hyperparameters
    hparams = ckpt["hyper_parameters"]

    # Load gazetteer
    with open(DATA_DIR / "gazetteer.json") as f:
        gazetteer = json.load(f)

    # Compute fnum counts across all splits
    with open(DATA_DIR / "training_examples.json") as f:
        examples = json.load(f)
    fnum_train_counts = {
        str(k): v
        for k, v in Counter(
            ex["records"][0]["f_num"]
            for ex in examples
            if ex["records"] and ex["records"][0]["f_num"] != -100
        ).items()
    }

    # Save bundled checkpoint
    save_dict = {
        "model_state": model_state,
        "field_vocabs": hparams["field_vocabs"],
        "field_sizes": hparams["field_sizes"],
        "d_model": hparams["d_model"],
        "n_layers": hparams["n_layers"],
        "gazetteer": gazetteer,
        "fnum_train_counts": fnum_train_counts,
    }

    output = WEIGHTS_DIR / "structured_classifier.pt"
    torch.save(save_dict, output)
    print(f"Bundled checkpoint saved to {output}")
    print(f"  Keys: {list(save_dict.keys())}")


if __name__ == "__main__":
    main()
