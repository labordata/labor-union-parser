#!/usr/bin/env python3
"""Train the factored ArcFace f_num classifier.

Trains a FastText+RoPE encoder with factored ArcFace prototypes, shared
union classification head, disagree penalty, frozen OOV distractors,
W_fnum L2 regularization, and CRF latent alignment for number roles.

Usage:
    python training/train_arcface_classifier.py
    python training/train_arcface_classifier.py --epochs 30 --patience 10

Output:
    training/data/arcface_classifier.ckpt (or --save-checkpoint path)
"""

import json
import random
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import click
import torch

print = partial(print, flush=True)  # noqa: A001

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import model, tokenization, and CRF utilities from the spike
# (these will be refactored into the package eventually)
from arcface_fasttext_bloom_unionhead_spike import (  # noqa: E402
    NUM_BLOOM_HASHES,
    ArcFaceFastTextModel,
    bloom_hash_ids,
    build_vocab,
    collate_batch,
    compute_fnum_freq,
    evaluate,
    load_data,
    print_results,
)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Fixed hyperparameters (best config from experiments)
# ---------------------------------------------------------------------------
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
N_BUCKETS = 50000
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.0
FNUM_REG = 100.0
UNION_WEIGHT = 1.0
DISAGREE_PENALTY = 1.0
TAG_WEIGHT = 1.0


def encode_examples(data, vocab, fnum_to_idx, field_vocabs_aux=None):
    """Encode examples with target indices and auxiliary head targets."""
    for ex in data:
        ex["token_ids"] = [vocab.get(tok, 1) for tok in ex["tokens"]]
        ex["is_num_f"] = [float(n) for n in ex["is_num"]]
        ex["target"] = fnum_to_idx.get(ex["f_num"], -100)

        rec = ex.get("record", {})
        if field_vocabs_aux:
            uv = field_vocabs_aux.get("union_name", {})
            ex["union_target"] = uv.get(ex.get("union_name", ""), -1)
            for field in ["desig_name", "prefix", "suffix"]:
                fv = field_vocabs_aux.get(field, {})
                val = rec.get(field, -100)
                if val in (-100, 0, "", None):
                    ex[f"{field}_target"] = -1
                else:
                    ex[f"{field}_target"] = fv.get(val, -1)
        else:
            ex["union_target"] = -1
            for field in ["desig_name", "prefix", "suffix"]:
                ex[f"{field}_target"] = -1


def build_fnum_mapping(data):
    return {
        f: i
        for i, f in enumerate(
            sorted(
                set(
                    ex["f_num"]
                    for ex in data
                    if ex["split"] == "train" and ex["f_num"] != -100
                )
            )
        )
    }


@click.command()
@click.option("--epochs", default=50, help="Max training epochs")
@click.option("--patience", default=15, help="Early stopping patience")
@click.option("--batch-size", default=256)
@click.option("--lr", default=1e-3, type=float)
@click.option(
    "--save-checkpoint",
    default=str(DATA_DIR / "arcface_classifier.ckpt"),
    help="Output checkpoint path",
)
def main(epochs, patience, batch_size, lr, save_checkpoint):
    random.seed(42)
    torch.manual_seed(42)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # --- Data ---
    print("Loading data...")
    data, skipped, n_nofnum = load_data(
        str(DATA_DIR / "training_examples.json"), N_BUCKETS
    )
    n_with_fnum = len(data) - n_nofnum
    print(f"Loaded {n_with_fnum} with f_num, {n_nofnum} union-only ({skipped} skipped)")

    train_data = [ex for ex in data if ex["split"] == "train"]
    val_data = [ex for ex in data if ex["split"] == "val"]
    test_data = [ex for ex in data if ex["split"] == "test"]

    fnum_to_idx = build_fnum_mapping(data)
    n_classes = len(fnum_to_idx)
    n_train_classes = n_classes
    fnum_freq = compute_fnum_freq(data)
    idx_to_fnum_map = {v: k for k, v in fnum_to_idx.items()}

    val_data = [ex for ex in val_data if ex["f_num"] in fnum_to_idx]
    test_data = [ex for ex in test_data if ex["f_num"] in fnum_to_idx]

    print(f"Classes: {n_classes}, Train: {len(train_data)}, Val: {len(val_data)}")

    vocab = build_vocab(data)
    print(f"Vocab: {len(vocab)} tokens")

    # --- Union vocab ---
    union_names = sorted(
        set(ex.get("union_name", "") for ex in train_data if ex.get("union_name"))
    )
    union_vocab = {name: i for i, name in enumerate(union_names)}
    n_unions = len(union_vocab)
    print(f"Union vocab: {n_unions}")

    # --- Field vocabs and prototypes ---
    field_vocabs = {}
    fnum_records = {}
    fnum_all_records = defaultdict(list)

    for ex in train_data:
        fn = ex["f_num"]
        if fn == -100 or not ex.get("union_name"):
            continue
        raw_rec = ex.get("record", {})
        rec = {
            "union_name": ex["union_name"],
            "desig_name": raw_rec.get("desig_name", -100),
            "desig_num": raw_rec.get("desig_num", -100),
            "prefix": raw_rec.get("prefix", -100),
            "suffix": raw_rec.get("suffix", -100),
        }
        if fn not in fnum_records:
            fnum_records[fn] = rec
        sig = tuple(
            rec[k]
            for k in ["union_name", "desig_name", "desig_num", "prefix", "suffix"]
        )
        fnum_all_records[fn].append(sig)

    for field in ["union_name", "desig_name", "prefix", "suffix"]:
        vals = sorted(
            set(
                rec[field]
                for rec in fnum_records.values()
                if rec[field] not in (-100, 0, None, "")
            )
        )
        field_vocabs[field] = {v: i + 1 for i, v in enumerate(vals)}

    field_sizes = {f: len(v) for f, v in field_vocabs.items()}
    print(f"Field sizes: {field_sizes}")

    field_vocabs_aux = {
        field: {v: idx - 1 for v, idx in field_vocabs[field].items()}
        for field in field_vocabs
    }
    encode_examples(train_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(val_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(test_data, vocab, fnum_to_idx, field_vocabs_aux)

    # --- Build prototypes ---
    proto_rows = []
    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        all_recs = fnum_all_records.get(fn, [])
        if not all_recs:
            rec = fnum_records.get(fn, {})
            fields = [0, 0, 0, 0]
            for col, field_name in enumerate(
                ["union_name", "desig_name", "prefix", "suffix"]
            ):
                val = rec.get(field_name, "")
                if val and val not in (-100, 0, None, ""):
                    fields[col] = field_vocabs[field_name].get(val, 0)
            hashes = [0] * NUM_BLOOM_HASHES
            dnum = rec.get("desig_num", 0)
            if dnum not in (-100, 0, None):
                hashes = bloom_hash_ids(str(int(dnum)))
            proto_rows.append((i, fields, hashes))
        else:
            seen_hashes = set()
            for sig in all_recs:
                un, dn_name, dnum, pfx, sfx = sig
                fields = [0, 0, 0, 0]
                for col, (field_name, val) in enumerate(
                    zip(
                        ["union_name", "desig_name", "prefix", "suffix"],
                        [un, dn_name, pfx, sfx],
                    )
                ):
                    if val and val not in (-100, 0, None, ""):
                        fields[col] = field_vocabs[field_name].get(val, 0)
                hashes = [0] * NUM_BLOOM_HASHES
                if dnum not in (-100, 0, None):
                    hashes = bloom_hash_ids(str(int(dnum)))
                hashes_key = tuple(hashes)
                if hashes_key not in seen_hashes:
                    seen_hashes.add(hashes_key)
                    proto_rows.append((i, fields, hashes))

    n_train_protos = len(proto_rows)
    print(f"Train prototypes: {n_train_protos}")

    # --- Frozen OOV prototypes from gazetteer ---
    gaz_path = DATA_DIR / "gazetteer.json"
    with open(gaz_path) as f:
        gazetteer_data = json.load(f)

    n_oov = 0
    for fnum_str, gaz_records in sorted(
        gazetteer_data.items(), key=lambda x: int(x[0])
    ):
        fn = int(fnum_str)
        if fn in fnum_to_idx:
            continue
        class_idx = len(fnum_to_idx)
        fnum_to_idx[fn] = class_idx
        idx_to_fnum_map[class_idx] = fn

        seen_hashes = set()
        for rec in gaz_records:
            fields = [0, 0, 0, 0]
            for col, field_name in enumerate(
                ["union_name", "desig_name", "prefix", "suffix"]
            ):
                val = rec.get(field_name, "")
                if val and val not in (0, -100, None, ""):
                    fields[col] = field_vocabs[field_name].get(val, 0)
            dnum = rec.get("desig_num", 0)
            hashes = [0] * NUM_BLOOM_HASHES
            if dnum and dnum not in (0, -100, None):
                hashes = bloom_hash_ids(str(int(dnum)))
            hashes_key = (tuple(fields), tuple(hashes))
            if hashes_key not in seen_hashes:
                seen_hashes.add(hashes_key)
                proto_rows.append((class_idx, fields, hashes))
        n_oov += 1

    n_classes = len(fnum_to_idx)
    print(f"Frozen OOV: {n_oov} f_nums ({len(proto_rows) - n_train_protos} protos)")
    print(f"Total classes: {n_classes}")

    # Build prototype tensors
    n_protos = len(proto_rows)
    field_map = torch.zeros(n_protos, 4, dtype=torch.long)
    desig_bloom_t = torch.zeros(n_protos, NUM_BLOOM_HASHES, dtype=torch.long)
    proto_to_class = torch.zeros(n_protos, dtype=torch.long)

    for p, (class_idx, fields, hashes) in enumerate(proto_rows):
        proto_to_class[p] = class_idx
        for col in range(4):
            field_map[p, col] = fields[col]
        for j in range(NUM_BLOOM_HASHES):
            desig_bloom_t[p, j] = hashes[j]

    factored_info = {
        "field_vocabs": field_vocabs,
        "field_sizes": field_sizes,
        "field_map": field_map,
        "desig_bloom": desig_bloom_t,
        "proto_to_class": proto_to_class,
    }

    # --- Model ---
    model = ArcFaceFastTextModel(
        n_classes=n_classes,
        n_unions=n_unions,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_buckets=N_BUCKETS,
        vocab_size=len(vocab),
        scale=ARCFACE_SCALE,
        margin=ARCFACE_MARGIN,
        factored_info=factored_info,
    ).to(device)

    # Freeze OOV W_fnum
    with torch.no_grad():
        model.arcface.W_fnum.data[n_train_classes:].zero_()

    def _zero_oov_grad(grad):
        grad[n_train_classes:] = 0
        return grad

    model.arcface.W_fnum.register_hook(_zero_oov_grad)
    print(f"Frozen W_fnum for OOV classes {n_train_classes}..{n_classes}")

    # Class→union mapping for disagree penalty
    class_to_union = torch.zeros(n_classes, dtype=torch.long)
    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        if fn in fnum_records:
            rec = fnum_records[fn]
        else:
            gaz_recs = gazetteer_data.get(str(fn), [{}])
            rec = gaz_recs[0] if gaz_recs else {}
        un = rec.get("union_name", "")
        proto_un_idx = field_vocabs["union_name"].get(un, 0)
        class_to_union[i] = max(proto_un_idx - 1, 0)
    model.class_to_union = class_to_union.to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'Epoch':>7} | {'Loss':>8} | {'Top-1':>7} | {'Top-5':>7} | {'Time':>6}")
    print("-" * 48)

    best_val_top1 = 0.0
    best_state = None
    wait = 0
    n_train = len(train_data)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()

        indices = list(range(n_train))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = [train_data[i] for i in batch_indices]

            (
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
                field_targets,
            ) = collate_batch(batch, device)

            _, arcface_loss, field_losses, disagree_loss = model(
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
                field_targets,
            )

            loss = arcface_loss
            for fname, fl in field_losses.items():
                w = TAG_WEIGHT if fname == "crf_tags" else UNION_WEIGHT
                loss = loss + w * fl
            if disagree_loss is not None:
                loss = loss + DISAGREE_PENALTY * disagree_loss
            loss = loss + FNUM_REG * model.arcface.W_fnum.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_results = evaluate(model, val_data, fnum_freq, device)
        elapsed = time.time() - t0

        marker = ""
        if val_results["top1"] > best_val_top1:
            best_val_top1 = val_results["top1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(
            f"  {epoch + 1:>2}/{epochs} | {avg_loss:>8.4f} | {val_results['top1']:>6.1%} | "
            f"{val_results['top5']:>6.1%} | {elapsed:>5.1f}s{marker}"
        )

        if wait >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break

    # --- Restore best and evaluate ---
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    print(f"\nRestored best model (val top1={best_val_top1:.1%})")

    # Save checkpoint
    checkpoint = {
        "state_dict": best_state,
        "fnum_to_idx": fnum_to_idx,
        "vocab": vocab,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "n_classes": n_classes,
        "n_train_classes": n_train_classes,
        "n_buckets": N_BUCKETS,
        "arcface_scale": ARCFACE_SCALE,
        "arcface_margin": ARCFACE_MARGIN,
        "field_vocabs": field_vocabs,
        "field_sizes": field_sizes,
        "field_map": field_map,
        "desig_bloom": desig_bloom_t,
        "proto_to_class": proto_to_class,
        "idx_to_fnum": idx_to_fnum_map,
        "union_vocab": union_vocab,
        "n_unions": n_unions,
    }
    torch.save(checkpoint, save_checkpoint)
    print(f"Checkpoint saved to {save_checkpoint}")

    # Final test evaluation
    test_results = evaluate(model, test_data, fnum_freq, device)
    print_results(test_results, "Test Set")

    val_results = evaluate(model, val_data, fnum_freq, device)
    print_results(val_results, "Val Set")


if __name__ == "__main__":
    main()
