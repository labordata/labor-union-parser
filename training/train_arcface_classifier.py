#!/usr/bin/env python3
"""Train the factored ArcFace classifier for production.

This is a thin wrapper around arcface_fasttext_bloom_unionhead_spike.py
with the recommended hyperparameters for the best model.

Usage:
    python training/train_arcface_classifier.py
    python training/train_arcface_classifier.py --epochs 30 --patience 10
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CHECKPOINT_PATH = DATA_DIR / "arcface_classifier.ckpt"
SPIKE_SCRIPT = SCRIPT_DIR / "arcface_fasttext_bloom_unionhead_spike.py"

# Default hyperparameters (best configuration)
DEFAULTS = {
    "--data": str(DATA_DIR / "training_examples.json"),
    "--epochs": "50",
    "--patience": "15",
    "--batch-size": "256",
    "--lr": "1e-3",
    "--d-model": "128",
    "--n-layers": "3",
    "--n-heads": "4",
    "--n-buckets": "50000",
    "--arcface-scale": "30.0",
    "--arcface-margin": "0.0",
    "--union-weight": "1.0",
    "--disagree-penalty": "1.0",
    "--save-checkpoint": str(CHECKPOINT_PATH),
}


def main():
    # Build command with defaults, allowing CLI overrides
    cmd = [sys.executable, str(SPIKE_SCRIPT)]

    # Parse user args to detect overrides
    user_args = set()
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            user_args.add(sys.argv[i].split("=")[0])
        i += 1

    # Add defaults that aren't overridden
    for key, value in DEFAULTS.items():
        if key not in user_args:
            cmd.extend([key, value])

    # Add any user-provided args
    cmd.extend(sys.argv[1:])

    print("Training ArcFace classifier...")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Command: {' '.join(cmd[:6])}...")
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
