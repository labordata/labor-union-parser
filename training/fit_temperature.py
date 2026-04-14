#!/usr/bin/env python3
"""Post-hoc temperature scaling for ArcFace classifier and union head.

Fits separate scalar temperatures for f_num logits and union logits
on the validation set by minimizing NLL (standard Platt scaling).

Saves temperatures to the production weights bundle.
"""

import json
import random
from pathlib import Path

import click
import torch
import torch.nn.functional as F

from labor_union_parser import Extractor
from labor_union_parser.tokenizer import tokenize_for_arcface

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"


def fit_temperature(logits, targets, name, lr=0.01, steps=500):
    """Fit a scalar temperature by minimizing NLL."""
    nll_before = F.cross_entropy(logits, targets).item()

    log_temp = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_temp], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        temp = log_temp.exp()
        loss = F.cross_entropy(logits / temp, targets)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            print(f"  Step {step:3d}: NLL={loss.item():.4f}  T={temp.item():.4f}")

    temp = log_temp.exp().item()
    nll_after = F.cross_entropy(logits / temp, targets).item()
    print(f"  {name}: T={temp:.4f}, NLL {nll_before:.4f} -> {nll_after:.4f}")
    return temp


def show_calibration(logits, targets, temp, name):
    """Show calibration table for a head."""
    probs = F.softmax(logits / temp, dim=1)
    top_probs, top_preds = probs.max(dim=1)
    correct = (top_preds == targets).float()

    bins = [
        (0, 0.5),
        (0.5, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 0.95),
        (0.95, 0.99),
        (0.99, 1.0),
    ]
    print(f"\n  {name} calibration (T={temp:.4f}):")
    print(f"  {'Conf range':>12}  {'Count':>6}  {'Accuracy':>8}  {'Avg conf':>8}")
    for lo, hi in bins:
        mask = (top_probs >= lo) & (top_probs < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean().item()
        avg_conf = top_probs[mask].mean().item()
        print(
            f"  [{lo:.2f}, {hi:.2f})  {mask.sum().item():>6}  {acc:>8.1%}  {avg_conf:>8.4f}"
        )


@click.command()
@click.option("--lr", default=0.01, type=float, help="Learning rate")
@click.option("--steps", default=500, type=int, help="Optimization steps")
def main(lr, steps):
    ext = Extractor()
    model = ext.arcface_model
    model.eval()

    with open(DATA_DIR / "training_examples.json") as f:
        all_examples = json.load(f)

    fnum_to_idx = {v: k for k, v in ext.idx_to_fnum.items()}

    # Build union_name -> index mapping from bundle
    bundle = torch.load(
        WEIGHTS_DIR / "arcface_classifier.pt", map_location="cpu", weights_only=False
    )
    field_vocabs = bundle["field_vocabs"]
    union_vocab = {v: idx - 1 for v, idx in field_vocabs["union_name"].items()}

    # Val examples with known f_num (for f_num head)
    val_fnum = [
        ex
        for ex in all_examples
        if ex["split"] == "val"
        and ex.get("records")
        and ex["records"][0]["f_num"] != -100
        and ex["records"][0]["f_num"] in fnum_to_idx
    ]

    # Val examples with known union_name (for union head)
    val_union = [
        ex
        for ex in all_examples
        if ex["split"] == "val"
        and ex.get("records")
        and ex["records"][0].get("union_name", "")
        and ex["records"][0]["union_name"] in union_vocab
    ]

    print(f"Val examples: {len(val_fnum)} with f_num, {len(val_union)} with union_name")

    # Collect f_num logits
    print("\nCollecting f_num logits...")
    fnum_logits_list = []
    fnum_targets_list = []

    for i in range(0, len(val_fnum), 256):
        batch_ex = val_fnum[i : i + 256]
        texts = [ex["query"] for ex in batch_ex]
        targets = [fnum_to_idx[ex["records"][0]["f_num"]] for ex in batch_ex]

        batch_features = [tokenize_for_arcface(text) for text in texts]
        collated = ext._collate(batch_features, ext.vocab)

        with torch.no_grad():
            class_logits, _ = model(*collated)

        fnum_logits_list.append(class_logits.cpu())
        fnum_targets_list.append(torch.tensor(targets, dtype=torch.long))

    all_fnum_logits = torch.cat(fnum_logits_list)
    all_fnum_targets = torch.cat(fnum_targets_list)

    # Collect union logits
    print("Collecting union logits...")
    union_logits_list = []
    union_targets_list = []

    for i in range(0, len(val_union), 256):
        batch_ex = val_union[i : i + 256]
        texts = [ex["query"] for ex in batch_ex]
        targets = [union_vocab[ex["records"][0]["union_name"]] for ex in batch_ex]

        batch_features = [tokenize_for_arcface(text) for text in texts]
        collated = ext._collate(batch_features, ext.vocab)

        with torch.no_grad():
            _, union_logits = model(*collated)

        union_logits_list.append(union_logits.cpu())
        union_targets_list.append(torch.tensor(targets, dtype=torch.long))

    all_union_logits = torch.cat(union_logits_list)
    all_union_targets = torch.cat(union_targets_list)

    print(f"f_num logits: {all_fnum_logits.shape}")
    print(f"union logits: {all_union_logits.shape}")

    # Fit temperatures
    print("\n--- f_num temperature ---")
    fnum_temp = fit_temperature(all_fnum_logits, all_fnum_targets, "f_num", lr, steps)

    print("\n--- union_name temperature ---")
    union_temp = fit_temperature(
        all_union_logits, all_union_targets, "union_name", lr, steps
    )

    # Calibration tables
    show_calibration(all_fnum_logits, all_fnum_targets, fnum_temp, "f_num")
    show_calibration(all_union_logits, all_union_targets, union_temp, "union_name")

    # Save to arcface bundle
    bundle["fnum_temperature"] = fnum_temp
    bundle["union_temperature"] = union_temp
    torch.save(bundle, WEIGHTS_DIR / "arcface_classifier.pt")
    print(
        f"\nSaved fnum_temperature={fnum_temp:.4f}, union_temperature={union_temp:.4f}"
    )

    # --- Union detector Platt scaling ---
    # Fit sigmoid(a * cos_sim + b) to calibrate union_score
    print("\n--- Union detector Platt scaling ---")

    # Collect union detector cosine similarities on val set
    ud_sims = []

    # Union examples
    val_union = [
        ex
        for ex in all_examples
        if ex["split"] in ("val", "test")
        and ex.get("records")
        and ex.get("reason_missing_fnum") not in ("multi-union", "multi-local")
    ]
    # Non-union examples
    val_nonunion = [
        ex
        for ex in all_examples
        if ex["split"] in ("val", "test")
        and (
            not ex.get("records")
            or ex.get("reason_missing_fnum") in ("multi-union", "multi-local")
        )
    ]

    # Add F7 employer negatives (same as training script)
    import sqlite3

    f7_path = (DATA_DIR / "f7.db").resolve()
    if f7_path.exists():
        conn = sqlite3.connect(str(f7_path))
        rows = conn.execute(
            "SELECT DISTINCT employer FROM f7 "
            "WHERE employer IS NOT NULL AND employer != '' "
            "ORDER BY employer"
        ).fetchall()
        conn.close()
        rng = random.Random(42)
        employers = rng.sample([r[0] for r in rows], min(20000, len(rows)))
        # Use same 10% val split as training script
        n_val = len(employers) // 10
        employer_val = employers[:n_val]
        val_nonunion_texts = [ex["query"] for ex in val_nonunion] + employer_val
    else:
        val_nonunion_texts = [ex["query"] for ex in val_nonunion]

    print(
        f"Union detector val: {len(val_union)} union, {len(val_nonunion_texts)} non-union"
    )

    all_ud_texts = [ex["query"] for ex in val_union] + val_nonunion_texts
    all_ud_labels = [1.0] * len(val_union) + [0.0] * len(val_nonunion_texts)

    for i in range(0, len(all_ud_texts), 256):
        batch_texts = all_ud_texts[i : i + 256]
        batch_features = [tokenize_for_arcface(text) for text in batch_texts]
        collated = ext._collate(batch_features, ext.union_vocab)

        with torch.no_grad():
            emb = ext.union_encoder(*collated)
            sims = (emb @ ext.union_centroid.unsqueeze(0).T).squeeze(-1)
            ud_sims.extend(sims.cpu().tolist())

    ud_sims_t = torch.tensor(ud_sims)
    ud_labels_t = torch.tensor(all_ud_labels)

    # Fit Platt scaling: sigmoid(a * sim + b)
    log_a = torch.nn.Parameter(torch.zeros(1))
    b = torch.nn.Parameter(torch.zeros(1))
    platt_optimizer = torch.optim.Adam([log_a, b], lr=lr)

    # NLL before (using raw threshold)
    raw_nll = F.binary_cross_entropy_with_logits(ud_sims_t, ud_labels_t).item()

    for step in range(steps):
        platt_optimizer.zero_grad()
        a = log_a.exp()
        logits = a * ud_sims_t + b
        loss = F.binary_cross_entropy_with_logits(logits, ud_labels_t)
        loss.backward()
        platt_optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            print(
                f"  Step {step:3d}: NLL={loss.item():.4f}  a={log_a.exp().item():.4f}  b={b.item():.4f}"
            )

    platt_a = log_a.exp().item()
    platt_b = b.item()
    platt_nll = F.binary_cross_entropy_with_logits(
        platt_a * ud_sims_t + platt_b, ud_labels_t
    ).item()
    print(f"  NLL: {raw_nll:.4f} -> {platt_nll:.4f}")

    # Calibration table
    probs = torch.sigmoid(platt_a * ud_sims_t + platt_b)
    correct = (ud_labels_t == 1).float()
    bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print("\n  Union detector calibration (Platt scaled):")
    print(f"  {'Conf range':>12}  {'Count':>6}  {'Accuracy':>8}  {'Avg conf':>8}")
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean().item()
        avg_conf = probs[mask].mean().item()
        print(
            f"  [{lo:.1f}, {hi:.1f})  {mask.sum().item():>6}  {acc:>8.1%}  {avg_conf:>8.4f}"
        )

    # Save to union detector bundle
    ud_bundle = torch.load(
        WEIGHTS_DIR / "union_detector.pt", map_location="cpu", weights_only=False
    )
    ud_bundle["platt_a"] = platt_a
    ud_bundle["platt_b"] = platt_b
    torch.save(ud_bundle, WEIGHTS_DIR / "union_detector.pt")
    print(f"\nSaved platt_a={platt_a:.4f}, platt_b={platt_b:.4f} to union_detector.pt")


if __name__ == "__main__":
    main()
