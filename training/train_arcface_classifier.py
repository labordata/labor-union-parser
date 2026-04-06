#!/usr/bin/env python3
"""Train the factored ArcFace classifier.

Trains a FastText+RoPE encoder with factored ArcFace prototypes and a
shared union classification head with disagree penalty.

Usage:
    python training/train_arcface_classifier.py
    python training/train_arcface_classifier.py --epochs 30 --patience 10

Output:
    training/data/arcface_classifier.ckpt
"""

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import click
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.arcface_model import ArcFaceModel
from labor_union_parser.tokenizer import (
    NUM_BLOOM_HASHES,
    bloom_hash_ids,
    tokenize_for_arcface,
)

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path, n_buckets=50000):
    """Load training examples and precompute features."""
    with open(path) as f:
        raw = json.load(f)

    raw = [ex for ex in raw if ex.get("source") != "synthetic"]

    data = []
    skipped = 0
    for ex in raw:
        f_num = ex.get("f_num")
        if not f_num or f_num == -100:
            skipped += 1
            continue
        if not ex.get("records"):
            skipped += 1
            continue

        tokens, is_num, ngram_ids, ngram_counts, bloom_ids = tokenize_for_arcface(
            ex["query"], n_buckets=n_buckets
        )
        if not tokens:
            skipped += 1
            continue

        data.append(
            {
                "tokens": tokens,
                "is_num": is_num,
                "length": len(tokens),
                "f_num": int(f_num),
                "split": ex["split"],
                "union_name": ex.get("union_name"),
                "record": ex["records"][0] if ex.get("records") else {},
                "ngram_ids": ngram_ids,
                "ngram_counts": ngram_counts,
                "bloom_ids": bloom_ids,
            }
        )

    return data, skipped


def build_fnum_mapping(data):
    fnums = sorted(set(ex["f_num"] for ex in data if ex["split"] == "train"))
    return {f: i for i, f in enumerate(fnums)}


def encode_examples(data, vocab, fnum_to_idx, field_vocabs_aux=None):
    """Encode examples with target indices."""
    for ex in data:
        ex["token_ids"] = [vocab.get(tok, 1) for tok in ex["tokens"]]
        ex["is_num_f"] = [float(n) for n in ex["is_num"]]
        ex["target"] = fnum_to_idx[ex["f_num"]]

        if field_vocabs_aux:
            uv = field_vocabs_aux.get("union_name", {})
            ex["union_target"] = uv.get(ex.get("union_name", ""), -1)


def build_vocab(data):
    counter = Counter()
    for ex in data:
        if ex["split"] == "train":
            for tok in ex["tokens"]:
                counter[tok] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, count in counter.most_common():
        if count >= 2:
            vocab[tok] = len(vocab)
    return vocab


def collate_batch(batch, device):
    max_len = max(ex["length"] for ex in batch)
    max_ngrams = len(batch[0]["ngram_ids"][0])
    B = len(batch)

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids_t = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num_t = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)
    union_targets = torch.full((B,), -1, dtype=torch.long)

    for i, ex in enumerate(batch):
        L = ex["length"]
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(ex["token_ids"][:L], dtype=torch.long)
        ngram_ids[i, :L] = torch.tensor(ex["ngram_ids"][:L], dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ex["ngram_counts"][:L], dtype=torch.long)
        bloom_ids_t[i, :L] = torch.tensor(ex["bloom_ids"][:L], dtype=torch.long)
        is_num_t[i, :L] = torch.tensor(ex["is_num_f"], dtype=torch.float)
        targets[i] = ex["target"]
        union_targets[i] = ex.get("union_target", -1)

    return (
        token_ids.to(device),
        ngram_ids.to(device),
        ngram_counts.to(device),
        bloom_ids_t.to(device),
        is_num_t.to(device),
        lengths.to(device),
        targets.to(device),
        union_targets.to(device),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def evaluate(model, data, device, fnum_to_idx, batch_size=512):
    """Evaluate top-1 and top-5 accuracy."""
    model.eval()
    top1 = 0
    top5 = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            (
                token_ids,
                ngram_ids,
                ngram_counts,
                bloom_ids,
                is_num_t,
                lengths,
                targets,
                _union_targets,
            ) = collate_batch(batch, device)

            class_logits, _union_logits = model(
                token_ids, ngram_ids, ngram_counts, bloom_ids, is_num_t, lengths
            )

            _, top5_preds = class_logits.topk(5, dim=1)
            top1_correct = (top5_preds[:, 0] == targets).sum().item()
            top5_correct = (top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
            top1 += top1_correct
            top5 += top5_correct
            total += len(batch)

    return {"top1": top1 / total, "top5": top5 / total}


def train_step(
    model,
    batch,
    device,
    arcface_margin,
    arcface_scale,
    union_weight,
    disagree_penalty,
    class_to_union,
    n_classes,
):
    """Single training step. Returns loss."""
    (
        token_ids,
        ngram_ids,
        ngram_counts,
        bloom_ids,
        is_num_t,
        lengths,
        targets,
        union_targets,
    ) = batch

    # Forward
    embeddings = model.encode(
        token_ids, ngram_ids, ngram_counts, bloom_ids, is_num_t, lengths
    )
    class_logits, _union_logits = model(
        token_ids, ngram_ids, ngram_counts, bloom_ids, is_num_t, lengths
    )

    # ArcFace margin
    if arcface_margin > 0:
        cos_theta = class_logits / arcface_scale
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(targets, n_classes).float()
        class_logits = arcface_scale * torch.cos(theta + one_hot * arcface_margin)

    loss = F.cross_entropy(class_logits, targets)

    # Union head loss
    W_u = model.classifier.W_union.weight[1:]
    union_logits = model.union_scale * F.linear(embeddings, F.normalize(W_u, dim=1))
    valid = union_targets >= 0
    if valid.any():
        loss = loss + union_weight * F.cross_entropy(
            union_logits[valid], union_targets[valid]
        )

    # Disagree penalty
    if disagree_penalty > 0 and class_to_union is not None:
        fnum_probs = F.softmax(
            class_logits.detach() if arcface_margin > 0 else class_logits, dim=1
        )
        union_log_probs = F.log_softmax(union_logits, dim=1)
        union_per_class = union_log_probs[:, class_to_union]
        loss = loss + disagree_penalty * (
            -(fnum_probs * union_per_class).sum(dim=1).mean()
        )

    return loss


@click.command()
@click.option("--data", default=str(DATA_DIR / "training_examples.json"))
@click.option("--epochs", default=50)
@click.option("--batch-size", default=256)
@click.option("--lr", default=1e-3)
@click.option("--patience", default=15)
@click.option("--d-model", default=128)
@click.option("--n-layers", default=3)
@click.option("--n-heads", default=4)
@click.option("--n-buckets", default=50000)
@click.option("--arcface-scale", default=30.0)
@click.option("--arcface-margin", default=0.0)
@click.option("--union-weight", default=1.0)
@click.option("--disagree-penalty", default=1.0)
@click.option("--output", default=str(DATA_DIR / "arcface_classifier.ckpt"))
def main(
    data,
    epochs,
    batch_size,
    lr,
    patience,
    d_model,
    n_layers,
    n_heads,
    n_buckets,
    arcface_scale,
    arcface_margin,
    union_weight,
    disagree_penalty,
    output,
):
    random.seed(42)
    torch.manual_seed(42)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    all_data, skipped = load_data(data, n_buckets)
    print(f"Loaded {len(all_data)} examples ({skipped} skipped)")

    train_data = [ex for ex in all_data if ex["split"] == "train"]
    val_data = [ex for ex in all_data if ex["split"] == "val"]
    test_data = [ex for ex in all_data if ex["split"] == "test"]

    fnum_to_idx = build_fnum_mapping(all_data)
    n_classes = len(fnum_to_idx)
    idx_to_fnum_map = {v: k for k, v in fnum_to_idx.items()}
    print(f"Classes: {n_classes} f_nums")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Filter val/test to known f_nums
    val_data = [ex for ex in val_data if ex["f_num"] in fnum_to_idx]
    test_data = [ex for ex in test_data if ex["f_num"] in fnum_to_idx]

    # Build vocab
    vocab = build_vocab(all_data)
    print(f"Vocab: {len(vocab)} tokens")

    # Build field vocabs from records
    fnum_records = {}
    fnum_all_records = defaultdict(list)
    for ex in all_data:
        fn = ex["f_num"]
        if ex["split"] != "train":
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
        dnum = rec["desig_num"]
        if dnum not in (-100, 0, None):
            existing = {r["desig_num"] for r in fnum_all_records[fn]}
            if dnum not in existing:
                fnum_all_records[fn].append(rec)

    field_vocabs = {}
    for field in ["union_name", "desig_name", "prefix", "suffix"]:
        vals = sorted(
            set(
                r[field]
                for r in fnum_records.values()
                if r[field] not in (-100, 0, "", None)
            ),
            key=str,
        )
        field_vocabs[field] = {v: i + 1 for i, v in enumerate(vals)}

    field_sizes = {f: len(v) for f, v in field_vocabs.items()}
    n_unions = field_sizes["union_name"]
    print(f"Union vocab: {n_unions} unions")
    print(f"Field sizes: {field_sizes}")

    # Build aux vocabs (0-indexed for classification heads)
    field_vocabs_aux = {}
    for field in ["union_name"]:
        field_vocabs_aux[field] = {v: idx - 1 for v, idx in field_vocabs[field].items()}

    encode_examples(train_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(val_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(test_data, vocab, fnum_to_idx, field_vocabs_aux)

    # Build prototypes
    proto_rows = []
    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        variants = fnum_all_records.get(fn, [])
        if not variants:
            variants = [fnum_records.get(fn, {})]
        seen_hashes = set()
        for rec in variants:
            fields = [0, 0, 0, 0]
            for col, field in enumerate(
                ["union_name", "desig_name", "prefix", "suffix"]
            ):
                val = rec.get(field, -100)
                if val not in (-100, 0, "", None):
                    fields[col] = field_vocabs[field].get(val, 0)
            dnum = rec.get("desig_num", -100)
            hashes = [0] * NUM_BLOOM_HASHES
            if dnum not in (-100, 0, None):
                hashes = bloom_hash_ids(str(int(dnum)))
            hashes_key = tuple(hashes)
            if hashes_key not in seen_hashes:
                seen_hashes.add(hashes_key)
                proto_rows.append((i, fields, hashes))

    n_protos = len(proto_rows)
    print(f"Prototypes: {n_protos} ({n_protos - n_classes} variant aliases)")

    field_map = torch.zeros(n_protos, 4, dtype=torch.long)
    desig_bloom = torch.zeros(n_protos, NUM_BLOOM_HASHES, dtype=torch.long)
    proto_to_class = torch.zeros(n_protos, dtype=torch.long)
    for pi, (ci, fields, hashes) in enumerate(proto_rows):
        proto_to_class[pi] = ci
        for col in range(4):
            field_map[pi, col] = fields[col]
        for j in range(NUM_BLOOM_HASHES):
            desig_bloom[pi, j] = hashes[j]

    # Build model
    model = ArcFaceModel(
        n_classes=n_classes,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_buckets=n_buckets,
        vocab_size=len(vocab),
        scale=arcface_scale,
        field_sizes=field_sizes,
    ).to(device)

    # Set prototype buffers
    model.classifier.field_map = field_map.to(device)
    model.classifier.desig_bloom = desig_bloom.to(device)
    model.classifier.proto_to_class = proto_to_class.to(device)

    # Build class→union mapping for disagree penalty
    class_to_union = torch.zeros(n_classes, dtype=torch.long)
    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        rec = fnum_records.get(fn, {})
        un = rec.get("union_name", "")
        proto_un_idx = field_vocabs["union_name"].get(un, 0)
        class_to_union[i] = max(proto_un_idx - 1, 0)
    class_to_union = class_to_union.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\n{'Epoch':>7} | {'Loss':>8} | {'Top-1':>7} | {'Top-5':>7} | {'Time':>6}")
    print("-" * 48)

    best_val_top1 = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = [train_data[i] for i in batch_indices]
            collated = collate_batch(batch, device)

            loss = train_step(
                model,
                collated,
                device,
                arcface_margin,
                arcface_scale,
                union_weight,
                disagree_penalty,
                class_to_union,
                n_classes,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_results = evaluate(model, val_data, device, fnum_to_idx)
        elapsed = time.time() - t0

        marker = ""
        if val_results["top1"] > best_val_top1:
            best_val_top1 = val_results["top1"]
            import copy

            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(
            f"  {epoch+1:2d}/{epochs:2d} | {avg_loss:8.4f} | "
            f"{val_results['top1']:6.1%} | {val_results['top5']:6.1%} | "
            f"{elapsed:5.1f}s{marker}"
        )

        if wait >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break

    # Restore best model
    print(f"\nRestored best model (val top1={best_val_top1:.1%})")
    model.load_state_dict(best_state)

    # Save checkpoint
    checkpoint = {
        "state_dict": best_state,
        "fnum_to_idx": fnum_to_idx,
        "idx_to_fnum": idx_to_fnum_map,
        "vocab": vocab,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_classes": n_classes,
        "n_buckets": n_buckets,
        "n_unions": n_unions,
        "arcface_scale": arcface_scale,
        "arcface_margin": arcface_margin,
        "field_vocabs": field_vocabs,
        "field_sizes": field_sizes,
        "field_map": field_map,
        "desig_bloom": desig_bloom,
        "proto_to_class": proto_to_class,
    }
    torch.save(checkpoint, output)
    print(f"Checkpoint saved to {output}")

    # Final test evaluation
    model.eval()
    test_results = evaluate(model, test_data, device, fnum_to_idx)
    print("\n--- Test Set ---")
    print(
        f"  Overall: top1={test_results['top1']:.1%}  top5={test_results['top5']:.1%}  (n={len(test_data)})"
    )

    val_results = evaluate(model, val_data, device, fnum_to_idx)
    print("\n--- Val Set ---")
    print(
        f"  Overall: top1={val_results['top1']:.1%}  top5={val_results['top5']:.1%}  (n={len(val_data)})"
    )


if __name__ == "__main__":
    main()
