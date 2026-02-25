#!/usr/bin/env python3
"""Bundle learned scoring layer weights for production use.

Reads temperatures.json and the best scoring layer checkpoint,
writes a compact scoring_weights.pt for the Extractor.
"""

import json
import math
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"


def main():
    # Load temperatures
    with open(DATA_DIR / "temperatures.json") as f:
        temps = json.load(f)

    # Load scoring layer checkpoint (find latest versioned checkpoint)
    ckpt_files = sorted(
        WEIGHTS_DIR.glob("scoring_layer*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    if not ckpt_files:
        raise FileNotFoundError("No scoring_layer*.ckpt found in weights dir")
    ckpt_path = ckpt_files[-1]
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    scoring_state = {
        k.removeprefix("scoring."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("scoring.")
    }

    penalty_weights = scoring_state["w_penalty.weight"][0]  # (6,)
    penalty_bias = scoring_state["w_penalty.bias"][0].item()
    gate_ceil = scoring_state["gate_ceil"].item()
    gate_log_k = scoring_state["gate_log_k"].item()

    scoring_temp = temps.get("scoring", 1.0)

    save_dict = {
        "temperatures": {
            "union_name": temps["union_name"],
            "desig_name": temps["desig_name"],
            "desig_num": temps["desig_num"],
            "prefix": temps["prefix"],
            "suffix": temps["suffix"],
            "f_num": temps["f_num"],
        },
        "penalty_weights": penalty_weights,
        "penalty_bias": penalty_bias,
        "gate_ceil": gate_ceil,
        "gate_log_k": gate_log_k,
        "scoring_temperature": scoring_temp,
    }

    output = WEIGHTS_DIR / "scoring_weights.pt"
    torch.save(save_dict, output)
    print(f"Scoring weights saved to {output}")
    print(
        f"  Temperatures: { {k: f'{v:.4f}' for k, v in save_dict['temperatures'].items()} }"
    )
    print(f"  Penalty weights: {penalty_weights.tolist()}")
    print(f"  Penalty bias: {penalty_bias:.4f}")
    print(f"  Gate ceil: {gate_ceil:.4f}")
    print(f"  Gate k: {math.exp(gate_log_k):.1f}")
    print(f"  Scoring temperature: {scoring_temp:.4f}")


if __name__ == "__main__":
    main()
