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
    ckpt = torch.load(DATA_DIR / "structured_classifier.ckpt", weights_only=False)

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

    # Compute fnum training counts
    with open(DATA_DIR / "training_examples.json") as f:
        examples = json.load(f)
    fnum_train_counts = {
        str(k): v
        for k, v in Counter(
            ex["records"][0]["f_num"]
            for ex in examples
            if ex["split"] == "train" and ex["records"]
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
