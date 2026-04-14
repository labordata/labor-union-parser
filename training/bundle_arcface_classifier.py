"""Bundle trained ArcFace checkpoint into production weights.

Reads:
  - training/data/arcface_classifier.ckpt
  - training/data/gazetteer.json

Writes:
  - src/labor_union_parser/weights/arcface_classifier.pt

The checkpoint already includes OOV prototypes (built during training).
This script filters the state dict to inference-only keys and packages
the bundle for production.
"""

import json
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
WEIGHTS_DIR = SCRIPT_DIR.parent / "src" / "labor_union_parser" / "weights"


def main():
    ckpt_path = DATA_DIR / "arcface_classifier.ckpt"
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Filter state dict to inference keys (encoder + classifier only)
    mapped_state = {}
    for key, value in ckpt["state_dict"].items():
        if (
            key.startswith("encoder.")
            or key.startswith("classifier.")
            or key == "union_scale"
        ):
            mapped_state[key] = value
        # Skip: desig_scale, tag_head.*, _crf_*, class_to_union (training only)

    # Build union_names list (ordered by head output index)
    # field_vocabs["union_name"] maps name -> 1-indexed id
    # W_union[1:] is 0-indexed, so union_names[0] = first union
    field_vocabs = ckpt["field_vocabs"]
    union_vocab = field_vocabs["union_name"]
    union_names = [""] * len(union_vocab)
    for name, idx in union_vocab.items():
        union_names[idx - 1] = name

    with open(DATA_DIR / "gazetteer.json") as f:
        gazetteer = json.load(f)

    n_classes = ckpt["n_classes"]
    n_train_classes = ckpt["n_train_classes"]

    bundle = {
        "state_dict": mapped_state,
        "vocab": ckpt["vocab"],
        "fnum_to_idx": ckpt["fnum_to_idx"],
        "idx_to_fnum": ckpt["idx_to_fnum"],
        "field_vocabs": field_vocabs,
        "field_sizes": ckpt["field_sizes"],
        "field_map": ckpt["field_map"],
        "desig_bloom": ckpt["desig_bloom"],
        "proto_to_class": ckpt["proto_to_class"],
        "n_classes": n_classes,
        "n_train_classes": n_train_classes,
        "d_model": ckpt["d_model"],
        "n_heads": ckpt["n_heads"],
        "n_layers": ckpt["n_layers"],
        "n_buckets": ckpt["n_buckets"],
        "arcface_scale": ckpt["arcface_scale"],
        "union_names": union_names,
        "gazetteer": gazetteer,
    }

    out_path = WEIGHTS_DIR / "arcface_classifier.pt"
    torch.save(bundle, out_path)
    print(f"Saved to {out_path}")
    print(f"  n_classes: {n_classes}")
    print(f"  n_unions: {len(union_names)}")
    print(f"  vocab_size: {len(ckpt['vocab'])}")
    print(f"  d_model: {bundle['d_model']}, n_layers: {bundle['n_layers']}")
    print(f"  gazetteer records: {sum(len(v) for v in gazetteer.values())}")


if __name__ == "__main__":
    main()
