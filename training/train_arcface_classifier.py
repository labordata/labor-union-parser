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
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from crf_tagger import CRFTaggerMixin, build_field_tensors, find_valid_positions

from labor_union_parser.arcface_model import ArcFaceModel
from labor_union_parser.tokenizer import (
    DEFAULT_N_BUCKETS,
    NUM_BLOOM_HASHES,
    bloom_hash_ids,
    tokenize_for_arcface,
)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
N_BUCKETS = DEFAULT_N_BUCKETS
ARCFACE_SCALE = 30.0
ARCFACE_MARGIN = 0.0
FNUM_REG = 100.0

# ---------------------------------------------------------------------------
# Model (training-specific classes; encoder imported from arcface_model.py)
# ---------------------------------------------------------------------------


class TrainingModel(CRFTaggerMixin, ArcFaceModel):
    """ArcFaceModel extended with CRF tag head and disagree penalty for training."""

    def __init__(self, n_classes, n_unions, vocab_size, factored_info):
        super().__init__(
            n_classes=n_classes,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            n_buckets=N_BUCKETS,
            vocab_size=vocab_size,
            scale=ARCFACE_SCALE,
            field_sizes=factored_info["field_sizes"],
        )
        self.classifier.margin = ARCFACE_MARGIN
        self.classifier.field_map = factored_info["field_map"]
        self.classifier.desig_bloom = factored_info["desig_bloom"]
        self.classifier.proto_to_class = factored_info["proto_to_class"]
        self.desig_scale = nn.Parameter(torch.tensor(10.0))
        CRFTaggerMixin.init_crf_params(self, D_MODEL)
        self.class_to_union = None

    def forward(
        self,
        token_ids,
        ngram_ids,
        ngram_counts,
        bloom_ids,
        is_num,
        lengths,
        targets=None,
        field_targets=None,
    ):
        embeddings, hidden = self.encode(
            token_ids, ngram_ids, ngram_counts, bloom_ids, is_num, lengths
        )
        logits, arcface_loss = self.classifier(embeddings, targets)

        W_u = self.classifier.W_union.weight[1:]
        union_logits = self.union_scale * F.linear(embeddings, F.normalize(W_u, dim=1))
        W_dn = self.classifier.W_desig_name.weight[1:]
        desig_logits = self.desig_scale * F.linear(embeddings, F.normalize(W_dn, dim=1))
        tag_logits = self.tag_head(hidden)

        field_losses = {}
        if field_targets is not None:
            for field, flogits in [
                ("union_name", union_logits),
                ("desig_name", desig_logits),
            ]:
                ft = field_targets.get(field)
                if ft is not None:
                    valid = ft >= 0
                    if valid.any():
                        field_losses[field] = F.cross_entropy(flogits[valid], ft[valid])

            crf_loss_val = self.crf_loss(
                tag_logits, lengths, field_targets["crf_fields"]
            )
            if crf_loss_val is not None:
                field_losses["crf_tags"] = crf_loss_val

        disagree_loss = torch.tensor(0.0, device=logits.device)
        if logits is not None and targets is not None:
            fnum_valid = targets >= 0
            if fnum_valid.any() and self.class_to_union is not None:
                fnum_probs = F.softmax(logits[fnum_valid], dim=1)
                union_lp = F.log_softmax(union_logits[fnum_valid], dim=1)
                disagree_loss = (
                    disagree_loss
                    - (fnum_probs * union_lp[:, self.class_to_union]).sum(dim=1).mean()
                )

        return logits, arcface_loss, field_losses, disagree_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    raw = [ex for ex in raw if ex.get("source") != "synthetic"]

    data, skipped, n_nofnum = [], 0, 0
    for ex in raw:
        f_num = ex.get("f_num")
        has_fnum = f_num and f_num != -100
        if not has_fnum:
            union_name = ex.get("union_name")
            if not union_name or union_name == -100:
                skipped += 1
                continue
        elif not ex.get("records"):
            skipped += 1
            continue

        result = tokenize_for_arcface(ex["query"])
        tokens, is_num, ngram_ids, ngram_counts, bloom_ids = result
        if not tokens:
            skipped += 1
            continue

        record = ex["records"][0] if ex.get("records") else {}
        valid_dnum, valid_pfx, valid_sfx = find_valid_positions(tokens, record)

        data.append(
            {
                "tokens": tokens,
                "is_num": is_num,
                "length": len(tokens),
                "f_num": int(f_num) if has_fnum else -100,
                "split": ex["split"],
                "source": ex.get("source"),
                "union_name": ex.get("union_name"),
                "record": record,
                "ngram_ids": ngram_ids,
                "ngram_counts": ngram_counts,
                "bloom_ids": bloom_ids,
                "valid_dnum": valid_dnum,
                "valid_pfx": valid_pfx,
                "valid_sfx": valid_sfx,
            }
        )
        if not has_fnum:
            n_nofnum += 1
    return data, skipped, n_nofnum


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


def encode_examples(data, vocab, fnum_to_idx, field_vocabs_aux):
    for ex in data:
        ex["token_ids"] = [vocab.get(tok, 1) for tok in ex["tokens"]]
        ex["is_num_f"] = [float(n) for n in ex["is_num"]]
        ex["target"] = fnum_to_idx.get(ex["f_num"], -100)
        uv = field_vocabs_aux.get("union_name", {})
        ex["union_target"] = uv.get(ex.get("union_name", ""), -1)
        rec = ex.get("record", {})
        for field in ["desig_name", "prefix", "suffix"]:
            fv = field_vocabs_aux.get(field, {})
            val = rec.get(field, -100)
            ex[f"{field}_target"] = (
                -1 if val in (-100, 0, "", None) else fv.get(val, -1)
            )


def collate_batch(batch, device):
    B = len(batch)
    max_len = max(ex["length"] for ex in batch)
    max_ngrams = len(batch[0]["ngram_ids"][0])

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num_t = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)
    crf_dnum, crf_pfx, crf_sfx = [], [], []
    union_tgt = torch.full((B,), -1, dtype=torch.long)
    desig_tgt = torch.full((B,), -1, dtype=torch.long)
    prefix_tgt = torch.full((B,), -1, dtype=torch.long)
    suffix_tgt = torch.full((B,), -1, dtype=torch.long)

    for i, ex in enumerate(batch):
        L = ex["length"]
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(ex["token_ids"][:L], dtype=torch.long)
        ngram_ids[i, :L] = torch.tensor(ex["ngram_ids"][:L], dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ex["ngram_counts"][:L], dtype=torch.long)
        bloom_ids[i, :L] = torch.tensor(ex["bloom_ids"][:L], dtype=torch.long)
        is_num_t[i, :L] = torch.tensor(ex["is_num_f"], dtype=torch.float)
        targets[i] = ex["target"]
        crf_dnum.append(ex.get("valid_dnum", []))
        crf_pfx.append(ex.get("valid_pfx", []))
        crf_sfx.append(ex.get("valid_sfx", []))
        union_tgt[i] = ex.get("union_target", -1)
        desig_tgt[i] = ex.get("desig_name_target", -1)
        prefix_tgt[i] = ex.get("prefix_target", -1)
        suffix_tgt[i] = ex.get("suffix_target", -1)

    field_targets = {
        "union_name": union_tgt.to(device),
        "desig_name": desig_tgt.to(device),
        "prefix": prefix_tgt.to(device),
        "suffix": suffix_tgt.to(device),
        "crf_fields": [
            build_field_tensors(crf_dnum, B, device),
            build_field_tensors(crf_pfx, B, device),
            build_field_tensors(crf_sfx, B, device),
        ],
    }
    return (
        token_ids.to(device),
        ngram_ids.to(device),
        ngram_counts.to(device),
        bloom_ids.to(device),
        is_num_t.to(device),
        lengths.to(device),
        targets.to(device),
        field_targets,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def bucket_label(count):
    if count == 1:
        return "1"
    elif count <= 5:
        return "2-5"
    elif count <= 15:
        return "6-15"
    else:
        return "16+"


def evaluate(model, data, fnum_freq, device, batch_size=512):
    model.eval()
    buckets = {"1": [0, 0, 0], "2-5": [0, 0, 0], "6-15": [0, 0, 0], "16+": [0, 0, 0]}
    total_top1 = total_top5 = total = 0

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            if not batch:
                continue
            tk, ng, nc, bl, isn, ln, tg, ft = collate_batch(batch, device)
            logits, _, _, _ = model(tk, ng, nc, bl, isn, ln)
            _, top5_preds = logits.topk(5, dim=1)
            top1_correct = (top5_preds[:, 0] == tg).cpu()
            top5_correct = (top5_preds == tg.unsqueeze(1)).any(dim=1).cpu()
            for i, ex in enumerate(batch):
                freq = fnum_freq.get(ex["f_num"], 0)
                b = bucket_label(freq)
                buckets[b][2] += 1
                if top1_correct[i]:
                    buckets[b][0] += 1
                if top5_correct[i]:
                    buckets[b][1] += 1
                total += 1
                total_top1 += int(top1_correct[i])
                total_top5 += int(top5_correct[i])

    return {
        "top1": total_top1 / max(total, 1),
        "top5": total_top5 / max(total, 1),
        "total": total,
        "buckets": buckets,
    }


def print_results(results, label=""):
    if label:
        print(f"\n--- {label} ---")
    print(
        f"  Overall: top1={results['top1']:.1%}  top5={results['top5']:.1%}  (n={results['total']})"
    )
    print(f"    {'Bucket':>8} | {'Top-1':>8} | {'Top-5':>8} | {'Count':>6}")
    print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | {'------':>6}")
    for b in ["1", "2-5", "6-15", "16+"]:
        t1, t5, n = results["buckets"][b]
        if n > 0:
            print(f"  {b:>8} | {t1/n:>7.1%} | {t5/n:>7.1%} | {n:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    epochs = 50
    patience = 15
    batch_size = 256
    lr = 1e-3
    save_checkpoint = str(DATA_DIR / "arcface_classifier.ckpt")
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
    data, skipped, n_nofnum = load_data(str(DATA_DIR / "training_examples.json"))
    print(
        f"Loaded {len(data) - n_nofnum} with f_num, {n_nofnum} union-only ({skipped} skipped)"
    )

    train_data = [ex for ex in data if ex["split"] == "train"]
    val_data = [ex for ex in data if ex["split"] == "val"]
    test_data = [ex for ex in data if ex["split"] == "test"]

    fnum_to_idx = {
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
    n_train_classes = len(fnum_to_idx)
    idx_to_fnum_map = {v: k for k, v in fnum_to_idx.items()}
    fnum_freq = Counter(
        ex["f_num"]
        for ex in data
        if ex["split"] == "train"
        and ex["f_num"] != -100
        and ex.get("source") != "synthetic_mdlm"
    )

    val_data = [ex for ex in val_data if ex["f_num"] in fnum_to_idx]
    test_data = [ex for ex in test_data if ex["f_num"] in fnum_to_idx]

    vocab = build_vocab(data)
    union_names = sorted(
        set(ex.get("union_name", "") for ex in train_data if ex.get("union_name"))
    )
    n_unions = len(union_names)
    print(f"Classes: {n_train_classes}, Vocab: {len(vocab)}, Unions: {n_unions}")

    # Build field vocabs
    field_vocabs = {}
    fnum_records = {}
    fnum_all_records = defaultdict(list)
    for ex in train_data:
        fn = ex["f_num"]
        if fn == -100 or not ex.get("union_name"):
            continue
        raw_rec = ex.get("record", {})
        rec = {
            f: raw_rec.get(f, -100) if f != "union_name" else ex["union_name"]
            for f in ["union_name", "desig_name", "desig_num", "prefix", "suffix"]
        }
        if fn not in fnum_records:
            fnum_records[fn] = rec
        fnum_all_records[fn].append(
            tuple(
                rec[k]
                for k in ["union_name", "desig_name", "desig_num", "prefix", "suffix"]
            )
        )

    for field in ["union_name", "desig_name", "prefix", "suffix"]:
        vals = sorted(
            set(
                r[field]
                for r in fnum_records.values()
                if r[field] not in (-100, 0, None, "")
            )
        )
        field_vocabs[field] = {v: i + 1 for i, v in enumerate(vals)}
    field_sizes = {f: len(v) for f, v in field_vocabs.items()}

    field_vocabs_aux = {
        f: {v: idx - 1 for v, idx in field_vocabs[f].items()} for f in field_vocabs
    }
    encode_examples(train_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(val_data, vocab, fnum_to_idx, field_vocabs_aux)
    encode_examples(test_data, vocab, fnum_to_idx, field_vocabs_aux)

    # Build prototypes
    proto_rows = []
    for i in range(n_train_classes):
        fn = idx_to_fnum_map[i]
        all_recs = fnum_all_records.get(fn, [])
        if not all_recs:
            rec = fnum_records.get(fn, {})
            fields = [
                field_vocabs[f].get(rec.get(f, ""), 0)
                for f in ["union_name", "desig_name", "prefix", "suffix"]
            ]
            dnum = rec.get("desig_num", 0)
            hashes = (
                bloom_hash_ids(str(int(dnum)))
                if dnum and dnum not in (-100, 0, None)
                else [0] * NUM_BLOOM_HASHES
            )
            proto_rows.append((i, fields, hashes))
        else:
            seen = set()
            for un, dn_name, dnum, pfx, sfx in all_recs:
                fields = [0, 0, 0, 0]
                for col, (f, v) in enumerate(
                    zip(
                        ["union_name", "desig_name", "prefix", "suffix"],
                        [un, dn_name, pfx, sfx],
                    )
                ):
                    if v and v not in (-100, 0, None, ""):
                        fields[col] = field_vocabs[f].get(v, 0)
                hashes = (
                    bloom_hash_ids(str(int(dnum)))
                    if dnum and dnum not in (-100, 0, None)
                    else [0] * NUM_BLOOM_HASHES
                )
                key = (tuple(fields), tuple(hashes))
                if key not in seen:
                    seen.add(key)
                    proto_rows.append((i, fields, hashes))

    n_train_protos = len(proto_rows)
    print(f"Train prototypes: {n_train_protos}")

    # Frozen OOV from gazetteer
    with open(DATA_DIR / "gazetteer.json") as f:
        gazetteer_data = json.load(f)

    n_oov = 0
    for fnum_str, gaz_records in sorted(
        gazetteer_data.items(), key=lambda x: int(x[0])
    ):
        fn = int(fnum_str)
        if fn in fnum_to_idx:
            continue
        ci = len(fnum_to_idx)
        fnum_to_idx[fn] = ci
        idx_to_fnum_map[ci] = fn
        seen = set()
        for rec in gaz_records:
            fields = [0, 0, 0, 0]
            for col, f in enumerate(["union_name", "desig_name", "prefix", "suffix"]):
                val = rec.get(f, "")
                if val and val not in (0, -100, None, ""):
                    fields[col] = field_vocabs[f].get(val, 0)
            dnum = rec.get("desig_num", 0)
            hashes = (
                bloom_hash_ids(str(int(dnum)))
                if dnum and dnum not in (0, -100, None)
                else [0] * NUM_BLOOM_HASHES
            )
            key = (tuple(fields), tuple(hashes))
            if key not in seen:
                seen.add(key)
                proto_rows.append((ci, fields, hashes))
        n_oov += 1

    n_classes = len(fnum_to_idx)
    print(f"Frozen OOV: {n_oov}, Total classes: {n_classes}")

    n_protos = len(proto_rows)
    field_map = torch.zeros(n_protos, 4, dtype=torch.long)
    desig_bloom_t = torch.zeros(n_protos, NUM_BLOOM_HASHES, dtype=torch.long)
    proto_to_class = torch.zeros(n_protos, dtype=torch.long)
    for p, (ci, fields, hashes) in enumerate(proto_rows):
        proto_to_class[p] = ci
        field_map[p] = torch.tensor(fields)
        desig_bloom_t[p] = torch.tensor(hashes)

    # Model
    model = TrainingModel(
        n_classes,
        n_unions,
        len(vocab),
        {
            "field_sizes": field_sizes,
            "field_map": field_map,
            "desig_bloom": desig_bloom_t,
            "proto_to_class": proto_to_class,
        },
    ).to(device)

    with torch.no_grad():
        model.arcface.W_fnum.data[n_train_classes:].zero_()
    model.arcface.W_fnum.register_hook(
        lambda grad: grad.__setitem__(slice(n_train_classes, None), 0) or grad
    )

    # Class→union for disagree penalty
    class_to_union = torch.zeros(n_classes, dtype=torch.long)
    for i in range(n_classes):
        fn = idx_to_fnum_map[i]
        rec = fnum_records.get(fn) or (
            gazetteer_data.get(str(fn), [{}])[0] if str(fn) in gazetteer_data else {}
        )
        un = rec.get("union_name", "")
        class_to_union[i] = max(field_vocabs["union_name"].get(un, 0) - 1, 0)
    model.class_to_union = class_to_union.to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'Epoch':>7} | {'Loss':>8} | {'Top-1':>7} | {'Top-5':>7} | {'Time':>6}")
    print("-" * 48)

    best_val_top1, best_state, wait = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(train_data), batch_size):
            batch = [train_data[i] for i in indices[start : start + batch_size]]
            tk, ng, nc, bl, isn, ln, tg, ft = collate_batch(batch, device)
            _, arcface_loss, field_losses, disagree_loss = model(
                tk, ng, nc, bl, isn, ln, tg, ft
            )

            loss = arcface_loss + disagree_loss
            for fl in field_losses.values():
                loss = loss + fl
            loss = loss + FNUM_REG * model.classifier.W_fnum.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
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
            f"  {epoch+1:>2}/{epochs} | {total_loss/max(n_batches,1):>8.4f} | "
            f"{val_results['top1']:>6.1%} | {val_results['top5']:>6.1%} | {elapsed:>5.1f}s{marker}"
        )

        if wait >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break

    # Save
    model.load_state_dict(best_state)
    print(f"\nRestored best model (val top1={best_val_top1:.1%})")

    torch.save(
        {
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
            "field_vocabs": field_vocabs,
            "field_sizes": field_sizes,
            "field_map": field_map,
            "desig_bloom": desig_bloom_t,
            "proto_to_class": proto_to_class,
            "idx_to_fnum": idx_to_fnum_map,
            "union_vocab": {name: i for i, name in enumerate(union_names)},
            "n_unions": n_unions,
        },
        save_checkpoint,
    )
    print(f"Checkpoint saved to {save_checkpoint}")

    test_results = evaluate(model, test_data, fnum_freq, device)
    print_results(test_results, "Test Set")
    val_results = evaluate(model, val_data, fnum_freq, device)
    print_results(val_results, "Val Set")


if __name__ == "__main__":
    main()
