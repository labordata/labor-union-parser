"""Compare errors between two ArcFace checkpoints."""

import sys
from collections import Counter

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "training")

from arcface_fnum_spike import (
    ArcFaceFnumModel,
    bucket_label,
    collate_batch,
    compute_fnum_freq,
    encode_examples,
    load_data,
)


def load_model(ckpt_path, device):
    """Load a checkpoint, build vocab/fnum mapping, return model + encoded test data."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    data, _ = load_data(
        "training/data/training_examples.json",
        "training/data/synthetic_mdlm_crf_filtered.json",
    )
    vocab = ckpt["vocab"]
    fnum_to_idx = ckpt["fnum_to_idx"]
    idx_to_fnum = {v: k for k, v in fnum_to_idx.items()}
    n_classes = len(fnum_to_idx)

    fnum_freq = compute_fnum_freq(data)

    # Filter test to known fnums
    test_data = [
        ex for ex in data if ex["split"] == "test" and ex["f_num"] in fnum_to_idx
    ]
    encode_examples(test_data, vocab, fnum_to_idx)

    # Build model
    factored = "factored_info" in ckpt or "field_sizes" in ckpt
    factored_info = None
    if factored:
        factored_info = {
            "field_sizes": ckpt["field_sizes"],
            "field_map": ckpt["field_map"],
            "desig_nums": ckpt["desig_nums"],
        }

    model = ArcFaceFnumModel(
        vocab_size=len(vocab),
        n_classes=n_classes,
        d_model=ckpt.get("d_model", 128),
        n_heads=ckpt.get("n_heads", 4),
        n_layers=ckpt.get("n_layers", 3),
        factored=factored,
        factored_info=factored_info,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, test_data, fnum_to_idx, idx_to_fnum, fnum_freq


def get_predictions(model, test_data, device):
    """Get top-1 predictions for all test examples."""
    preds = []
    with torch.no_grad():
        for start in range(0, len(test_data), 512):
            batch = test_data[start : start + 512]
            token_ids, is_num_t, lengths, targets = collate_batch(batch, device)
            logits, _ = model(token_ids, is_num_t, lengths)
            top1 = logits.argmax(dim=1).cpu().tolist()
            preds.extend(top1)
    return preds


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt_a = "training/data/arcface_synthetic_crf.pt"
    ckpt_b = "training/data/arcface_full_factored_crf.pt"

    print(f"Model A (unfactored CRF): {ckpt_a}")
    print(f"Model B (full factored CRF): {ckpt_b}")

    # Load both models with their own vocabs
    model_a, test_a, fnum_to_idx_a, idx_to_fnum_a, fnum_freq = load_model(
        ckpt_a, device
    )
    model_b, test_b, fnum_to_idx_b, idx_to_fnum_b, _ = load_model(ckpt_b, device)

    # Get predictions
    preds_a = get_predictions(model_a, test_a, device)
    preds_b = get_predictions(model_b, test_b, device)

    # Build lookup by f_num for test_b since vocab may differ
    # Use text as join key
    text_to_pred_b = {}
    for ex, pred_idx in zip(test_b, preds_b):
        pred_fnum = idx_to_fnum_b[pred_idx]
        text_to_pred_b[
            ex["tokens"][0] if len(ex["tokens"]) == 1 else tuple(ex["tokens"])
        ] = (pred_fnum, ex)

    # Compare
    errors_a_only = []  # A wrong, B right (fixed by factoring)
    errors_b_only = []  # A right, B wrong (introduced by factoring)
    errors_both = []  # both wrong

    for ex_a, pred_idx_a in zip(test_a, preds_a):
        gt_fnum = ex_a["f_num"]
        pred_fnum_a = idx_to_fnum_a[pred_idx_a]
        correct_a = pred_fnum_a == gt_fnum

        # Find matching example in B by text
        key = ex_a["tokens"][0] if len(ex_a["tokens"]) == 1 else tuple(ex_a["tokens"])
        if key not in text_to_pred_b:
            continue
        pred_fnum_b, ex_b = text_to_pred_b[key]
        correct_b = pred_fnum_b == gt_fnum

        freq = fnum_freq.get(gt_fnum, 0)
        bucket = bucket_label(freq)
        info = {
            "text": " ".join(ex_a["tokens"][:20]),
            "gt_fnum": gt_fnum,
            "pred_a": pred_fnum_a,
            "pred_b": pred_fnum_b,
            "bucket": bucket,
        }

        if not correct_a and correct_b:
            errors_a_only.append(info)
        elif correct_a and not correct_b:
            errors_b_only.append(info)
        elif not correct_a and not correct_b:
            errors_both.append(info)

    total_a = sum(1 for p, e in zip(preds_a, test_a) if idx_to_fnum_a[p] != e["f_num"])
    total_b = sum(1 for p, e in zip(preds_b, test_b) if idx_to_fnum_b[p] != e["f_num"])

    print(f"\nErrors A (unfactored): {total_a}")
    print(f"Errors B (full factored): {total_b}")
    print(f"Fixed by factoring: {len(errors_a_only)}")
    print(f"Introduced by factoring: {len(errors_b_only)}")
    print(f"Errors in both: {len(errors_both)}")

    # Load gazetteer for union name lookups
    import sqlite3

    conn = sqlite3.connect("training/data/opdr.db")
    cur = conn.cursor()

    def get_union(fnum):
        cur.execute(
            "SELECT DISTINCT union_name FROM gazetteer WHERE f_num = ?", (fnum,)
        )
        row = cur.fetchone()
        return row[0] if row else "???"

    # Bucket breakdown
    print("\n--- Fixed by factoring (by bucket) ---")
    bucket_counts = Counter(e["bucket"] for e in errors_a_only)
    for b in ["1", "2-5", "6-15", "16+"]:
        print(f"  {b:>8s}: {bucket_counts.get(b, 0)}")

    print("\n--- Introduced by factoring (by bucket) ---")
    bucket_counts = Counter(e["bucket"] for e in errors_b_only)
    for b in ["1", "2-5", "6-15", "16+"]:
        print(f"  {b:>8s}: {bucket_counts.get(b, 0)}")

    # Cross-union analysis of introduced errors
    cross_union = 0
    same_union = 0
    for e in errors_b_only:
        gt_union = get_union(e["gt_fnum"])
        pred_union = get_union(e["pred_b"])
        if gt_union != pred_union:
            cross_union += 1
        else:
            same_union += 1

    print(f"\nIntroduced errors: {same_union} same-union, {cross_union} cross-union")

    # Show some introduced errors
    print("\n--- Sample introduced errors (cross-union) ---")
    shown = 0
    for e in errors_b_only:
        gt_union = get_union(e["gt_fnum"])
        pred_union = get_union(e["pred_b"])
        if gt_union != pred_union and shown < 15:
            print(f"  [{e['bucket']}] {e['text']}")
            print(f"    GT: {e['gt_fnum']} ({gt_union})")
            print(f"    Pred B: {e['pred_b']} ({pred_union})")
            shown += 1

    print("\n--- Sample fixed errors (cross-union) ---")
    shown = 0
    for e in errors_a_only:
        gt_union = get_union(e["gt_fnum"])
        pred_union = get_union(e["pred_a"])
        if gt_union != pred_union and shown < 15:
            print(f"  [{e['bucket']}] {e['text']}")
            print(f"    GT: {e['gt_fnum']} ({gt_union})")
            print(f"    Pred A: {e['pred_a']} ({pred_union})")
            shown += 1

    conn.close()


if __name__ == "__main__":
    main()
