#!/usr/bin/env python3
"""Bundle learned scoring layer weights for production use.

Reads temperatures.json and the best scoring layer checkpoint,
writes a compact scoring_weights.pt for the Extractor.
"""

import json
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"


def main():
    # Load temperatures
    with open(DATA_DIR / "temperatures.json") as f:
        temps = json.load(f)

    # Load scoring layer checkpoint
    ckpt_path = WEIGHTS_DIR / "scoring_layer.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No {ckpt_path} found")
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    scoring_state = {
        k.removeprefix("scoring."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("scoring.")
    }

    scoring_weight = scoring_state["linear.weight"][0, :12]  # (12,) drop count
    scoring_bias = scoring_state["linear.bias"][0].item()

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
        "scoring_weight": scoring_weight,
        "scoring_bias": scoring_bias,
        "scoring_temperature": scoring_temp,
    }

    output = WEIGHTS_DIR / "scoring_weights.pt"
    torch.save(save_dict, output)
    print(f"Scoring weights saved to {output}")
    print(
        f"  Temperatures: { {k: f'{v:.4f}' for k, v in save_dict['temperatures'].items()} }"
    )

    feature_names = [
        "lp_union",
        "lp_desig",
        "lp_fnum",
        "lp_designum",
        "lp_prefix",
        "lp_suffix",
        "unk_union",
        "unk_desig",
        "unk_fnum",
        "nf_designum",
        "nf_prefix",
        "nf_suffix",
    ]
    for name, w in zip(feature_names, scoring_weight.tolist()):
        print(f"  {name:>15s}: {w:+.4f}")
    print(f"  {'bias':>15s}: {scoring_bias:+.4f}")
    print(f"  Scoring temperature: {scoring_temp:.4f}")


if __name__ == "__main__":
    main()
