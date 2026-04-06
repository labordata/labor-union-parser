"""Bundle trained ArcFace checkpoint + gazetteer into production weights.

Reads:
  - training/data/arcface_classifier.ckpt (or --checkpoint path)
  - training/data/gazetteer.json

Writes:
  - src/labor_union_parser/weights/arcface_classifier.pt
"""

import json
from pathlib import Path

import click
import torch

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
WEIGHTS_DIR = SCRIPT_DIR.parent / "src" / "labor_union_parser" / "weights"


@click.command()
@click.option(
    "--checkpoint",
    default=str(DATA_DIR / "arcface_classifier.ckpt"),
    help="Path to trained checkpoint",
)
def main(checkpoint):
    print(f"Loading checkpoint: {Path(checkpoint).name}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Load gazetteer
    with open(DATA_DIR / "gazetteer.json") as f:
        gazetteer = json.load(f)

    # Build idx_to_fnum mapping
    fnum_to_idx = ckpt["fnum_to_idx"]
    idx_to_fnum = {v: k for k, v in fnum_to_idx.items()}

    # Build union_names list (ordered by head output index)
    # field_vocabs["union_name"] maps name -> 1-indexed id
    # W_union[1:] is 0-indexed, so union_names[0] = first union
    field_vocabs = ckpt["field_vocabs"]
    union_vocab = field_vocabs["union_name"]
    union_names = [""] * len(union_vocab)
    for name, idx in union_vocab.items():
        # field_vocabs is 1-indexed, union head output is 0-indexed (W_union[1:])
        union_names[idx - 1] = name

    # Extract only the encoder + classifier weights (no auxiliary training heads)
    state_dict = ckpt["state_dict"]

    # Map training model keys to inference model keys
    # Training: encoder.*, arcface.* (+ union_scale, desig_scale, class_to_*)
    # Inference: encoder.*, classifier.* (+ union_scale)
    mapped_state = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            mapped_state[key] = value
        elif key.startswith("arcface."):
            # arcface.* -> classifier.*
            new_key = key.replace("arcface.", "classifier.", 1)
            mapped_state[new_key] = value
        elif key == "union_scale":
            mapped_state[key] = value
        # Skip: desig_scale, class_to_union, class_to_desig (training only)

    bundle = {
        "state_dict": mapped_state,
        "vocab": ckpt["vocab"],
        "fnum_to_idx": fnum_to_idx,
        "idx_to_fnum": idx_to_fnum,
        "field_vocabs": field_vocabs,
        "field_sizes": ckpt["field_sizes"],
        "field_map": ckpt["field_map"],
        "desig_bloom": ckpt["desig_bloom"],
        "proto_to_class": ckpt["proto_to_class"],
        "n_classes": len(fnum_to_idx),
        "d_model": ckpt.get("d_model", 128),
        "n_heads": ckpt.get("n_heads", 4),
        "n_layers": ckpt.get("n_layers", 3),
        "n_buckets": ckpt.get("n_buckets", 50000),
        "arcface_scale": ckpt.get("arcface_scale", 30.0),
        "union_names": union_names,
        "match_threshold": ckpt.get("match_threshold", 0.0),
        "gazetteer": gazetteer,
    }

    out_path = WEIGHTS_DIR / "arcface_classifier.pt"
    torch.save(bundle, out_path)
    print(f"Saved to {out_path}")
    print(f"  n_classes: {len(fnum_to_idx)}")
    print(f"  n_unions: {len(union_names)}")
    print(f"  vocab_size: {len(ckpt['vocab'])}")
    print(f"  d_model: {bundle['d_model']}, n_layers: {bundle['n_layers']}")
    print(f"  gazetteer records: {sum(len(v) for v in gazetteer.values())}")


if __name__ == "__main__":
    main()
