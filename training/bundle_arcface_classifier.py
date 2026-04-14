"""Bundle trained ArcFace checkpoint + gazetteer into production weights.

Reads:
  - training/data/arcface_classifier.ckpt (or --checkpoint path)
  - training/data/gazetteer.json

Writes:
  - src/labor_union_parser/weights/arcface_classifier.pt

Zero-shot prototypes: For gazetteer f_nums not seen in training, we build
prototypes from field embeddings alone (W_fnum = 0). This works because
training with --fnum-reg pushes W_fnum toward zero, so the shared field
embeddings carry most of the discriminative information.
"""

import json
from pathlib import Path

import click
import torch

from labor_union_parser.tokenizer import NUM_BLOOM_HASHES, bloom_hash_ids

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
    fnum_to_idx = dict(ckpt["fnum_to_idx"])
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

    # --- Build zero-shot prototypes for OOV gazetteer f_nums ---
    train_field_map = ckpt["field_map"]
    train_desig_bloom = ckpt["desig_bloom"]
    train_proto_to_class = ckpt["proto_to_class"]
    n_train_classes = ckpt.get("n_train_classes", len(fnum_to_idx))
    n_train_protos = train_field_map.shape[0]
    d_model = ckpt.get("d_model", 128)

    oov_proto_rows = []  # (class_idx, [union, desig, prefix, suffix], [bloom_hashes])
    n_oov_classes = 0

    for fnum_str, gaz_records in sorted(gazetteer.items(), key=lambda x: int(x[0])):
        fn = int(fnum_str)
        if fn in fnum_to_idx:
            continue

        class_idx = n_train_classes + n_oov_classes
        fnum_to_idx[fn] = class_idx
        idx_to_fnum[class_idx] = fn
        n_oov_classes += 1

        seen_hashes = set()
        for rec in gaz_records:
            fields = [
                field_vocabs["union_name"].get(rec.get("union_name", ""), 0),
                field_vocabs["desig_name"].get(rec.get("desig_name", ""), 0),
                field_vocabs["prefix"].get(rec.get("prefix", 0), 0),
                field_vocabs["suffix"].get(str(rec.get("suffix", "")), 0),
            ]
            dnum = rec.get("desig_num", 0)
            if dnum and dnum not in (0, -100, None):
                hashes = bloom_hash_ids(str(int(dnum)))
            else:
                hashes = [0] * NUM_BLOOM_HASHES

            hashes_key = (tuple(fields), tuple(hashes))
            if hashes_key in seen_hashes:
                continue
            seen_hashes.add(hashes_key)
            oov_proto_rows.append((class_idx, fields, hashes))

    # Extend prototype tensors with OOV rows
    n_total_classes = max(len(fnum_to_idx), n_train_classes + n_oov_classes)
    n_total_protos = n_train_protos + len(oov_proto_rows)

    field_map = torch.zeros(n_total_protos, 4, dtype=torch.long)
    desig_bloom = torch.zeros(n_total_protos, NUM_BLOOM_HASHES, dtype=torch.long)
    proto_to_class = torch.zeros(n_total_protos, dtype=torch.long)

    field_map[:n_train_protos] = train_field_map
    desig_bloom[:n_train_protos] = train_desig_bloom
    proto_to_class[:n_train_protos] = train_proto_to_class

    for i, (cls_idx, fields, hashes) in enumerate(oov_proto_rows):
        p = n_train_protos + i
        proto_to_class[p] = cls_idx
        field_map[p] = torch.tensor(fields, dtype=torch.long)
        desig_bloom[p] = torch.tensor(hashes, dtype=torch.long)

    # Extend W_fnum with zeros for OOV classes (if not already extended)
    W_fnum_key = "classifier.W_fnum"
    train_W_fnum = mapped_state[W_fnum_key]
    if train_W_fnum.shape[0] < n_total_classes:
        extended_W_fnum = torch.zeros(n_total_classes, d_model)
        extended_W_fnum[: train_W_fnum.shape[0]] = train_W_fnum
        mapped_state[W_fnum_key] = extended_W_fnum

    print(f"  Train classes: {n_train_classes}, OOV classes: {n_oov_classes}")
    print(f"  Train protos: {n_train_protos}, OOV protos: {len(oov_proto_rows)}")

    bundle = {
        "state_dict": mapped_state,
        "vocab": ckpt["vocab"],
        "fnum_to_idx": fnum_to_idx,
        "idx_to_fnum": idx_to_fnum,
        "field_vocabs": field_vocabs,
        "field_sizes": ckpt["field_sizes"],
        "field_map": field_map,
        "desig_bloom": desig_bloom,
        "proto_to_class": proto_to_class,
        "n_classes": n_total_classes,
        "n_train_classes": n_train_classes,
        "d_model": d_model,
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
    print(f"  n_classes: {n_total_classes}")
    print(f"  n_unions: {len(union_names)}")
    print(f"  vocab_size: {len(ckpt['vocab'])}")
    print(f"  d_model: {bundle['d_model']}, n_layers: {bundle['n_layers']}")
    print(f"  gazetteer records: {sum(len(v) for v in gazetteer.values())}")


if __name__ == "__main__":
    main()
