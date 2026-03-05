"""Profile the factored ArcFace training to find bottlenecks."""

import sys
import time

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "training")

from arcface_fnum_spike import (
    ArcFaceFnumModel,
    build_vocab,
    collate_batch,
    encode_examples,
    load_data,
)


def profile_step(model, batch, device, n_iters=20):
    """Time individual components of a training step."""
    token_ids, is_num_t, lengths, targets = collate_batch(batch, device)

    # Warmup
    for _ in range(3):
        _, loss = model(token_ids, is_num_t, lengths, targets)
        loss.backward()

    if device.type == "mps":
        torch.mps.synchronize()

    # Profile forward components
    timings = {}

    # 1. Encoder
    t0 = time.perf_counter()
    for _ in range(n_iters):
        embeddings = model.encode(token_ids, is_num_t, lengths)
    if device.type == "mps":
        torch.mps.synchronize()
    timings["encoder"] = (time.perf_counter() - t0) / n_iters

    # 2. Prototype computation (factored only)
    if hasattr(model.arcface, "_prototypes"):
        t0 = time.perf_counter()
        for _ in range(n_iters):
            W = torch.nn.functional.normalize(model.arcface._prototypes(), dim=1)
        if device.type == "mps":
            torch.mps.synchronize()
        timings["prototypes"] = (time.perf_counter() - t0) / n_iters

    # 3. Cosine similarity (the big matmul: B x d_model @ d_model x n_classes)
    W = torch.nn.functional.normalize(
        (
            model.arcface._prototypes()
            if hasattr(model.arcface, "_prototypes")
            else model.arcface.W
        ),
        dim=1,
    )
    t0 = time.perf_counter()
    for _ in range(n_iters):
        cos_theta = torch.nn.functional.linear(embeddings.detach(), W.detach())
    if device.type == "mps":
        torch.mps.synchronize()
    timings["cosine_sim"] = (time.perf_counter() - t0) / n_iters

    # 4. ArcFace loss
    t0 = time.perf_counter()
    for _ in range(n_iters):
        logits = model.arcface.scale * cos_theta
        loss = torch.nn.functional.cross_entropy(logits, targets)
    if device.type == "mps":
        torch.mps.synchronize()
    timings["arcface_loss"] = (time.perf_counter() - t0) / n_iters

    # 5. Full forward + backward
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _, loss = model(token_ids, is_num_t, lengths, targets)
        loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    timings["full_fwd+bwd"] = (time.perf_counter() - t0) / n_iters

    return timings


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data, _ = load_data(
        "training/data/training_examples.json",
        "training/data/synthetic_mdlm_crf_filtered.json",
    )
    vocab = build_vocab(data)

    # Build fnum mapping
    all_fnums = sorted(set(ex["f_num"] for ex in data))
    fnum_to_idx = {f: i for i, f in enumerate(all_fnums)}
    n_classes = len(fnum_to_idx)

    train_data = [ex for ex in data if ex["split"] == "train"]
    encode_examples(train_data, vocab, fnum_to_idx)

    # Build factored info
    field_vocabs = {}
    for field in ["union_name", "desig_name", "prefix", "suffix"]:
        vals = set()
        for ex in data:
            rec = ex.get("record", {})
            v = rec.get(field, "")
            if v:
                vals.add(v)
        field_vocabs[field] = {v: i + 1 for i, v in enumerate(sorted(vals))}

    field_sizes = {k: len(v) for k, v in field_vocabs.items()}

    field_map = torch.zeros(n_classes, 4, dtype=torch.long)
    desig_nums = torch.zeros(n_classes, 8, dtype=torch.long)

    fnum_to_record = {}
    for ex in data:
        rec = ex.get("record", {})
        if rec and ex["f_num"] not in fnum_to_record:
            fnum_to_record[ex["f_num"]] = rec

    for fnum, idx in fnum_to_idx.items():
        rec = fnum_to_record.get(fnum, {})
        for col_i, field in enumerate(["union_name", "desig_name", "prefix", "suffix"]):
            v = rec.get(field, "")
            field_map[idx, col_i] = field_vocabs[field].get(v, 0)

        dnum_str = str(rec.get("desig_num", "") or "")[:8]
        for ci, ch in enumerate(dnum_str):
            if ch.isdigit():
                desig_nums[idx, ci] = int(ch) + 1

    factored_info = {
        "field_sizes": field_sizes,
        "field_map": field_map,
        "desig_nums": desig_nums,
    }

    # Create a batch
    batch = train_data[:256]

    print(f"\nVocab: {len(vocab)}, Classes: {n_classes}")
    print(f"Field sizes: {field_sizes}")
    print(f"Batch size: {len(batch)}")

    # Profile factored model
    print("\n--- Factored Model ---")
    model_f = ArcFaceFnumModel(
        len(vocab),
        n_classes,
        factored=True,
        factored_info=factored_info,
    ).to(device)
    timings_f = profile_step(model_f, batch, device)
    for name, t in timings_f.items():
        print(f"  {name:20s}: {t*1000:8.2f} ms")

    # Profile unfactored model
    print("\n--- Unfactored Model ---")
    model_u = ArcFaceFnumModel(len(vocab), n_classes).to(device)
    timings_u = profile_step(model_u, batch, device)
    for name, t in timings_u.items():
        print(f"  {name:20s}: {t*1000:8.2f} ms")

    # Profile: backward only for factored
    print("\n--- Factored: Forward vs Backward breakdown ---")
    token_ids, is_num_t, lengths, targets = collate_batch(batch, device)
    n_iters = 20

    # Forward only
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _, loss = model_f(token_ids, is_num_t, lengths, targets)
    if device.type == "mps":
        torch.mps.synchronize()
    fwd_time = (time.perf_counter() - t0) / n_iters
    print(f"  forward (no_grad)  : {fwd_time*1000:8.2f} ms")

    # Backward only
    _, loss = model_f(token_ids, is_num_t, lengths, targets)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _, loss = model_f(token_ids, is_num_t, lengths, targets)
        loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    bwd_time = (time.perf_counter() - t0) / n_iters
    print(f"  fwd+bwd            : {bwd_time*1000:8.2f} ms")
    print(f"  backward (est)     : {(bwd_time - fwd_time)*1000:8.2f} ms")

    # Profile: detach prototypes (no gradient through composition)
    print("\n--- Factored with detached prototypes ---")

    class DetachedForward:
        pass

    original_forward = model_f.arcface.forward

    def detached_forward(embeddings, targets=None):
        with torch.no_grad():
            W_detached = torch.nn.functional.normalize(
                model_f.arcface._prototypes(), dim=1
            )
        # Only W_fnum gets gradients
        cos_theta = torch.nn.functional.linear(embeddings, W_detached)
        if targets is None:
            return model_f.arcface.scale * cos_theta, None
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.nn.functional.one_hot(targets, W_detached.shape[0]).float()
        logits = model_f.arcface.scale * torch.cos(
            theta + one_hot * model_f.arcface.margin
        )
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    model_f.arcface.forward = detached_forward
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _, loss = model_f(token_ids, is_num_t, lengths, targets)
        loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    detach_time = (time.perf_counter() - t0) / n_iters
    print(f"  fwd+bwd (detached) : {detach_time*1000:8.2f} ms")
    model_f.arcface.forward = original_forward

    # Profile: torch.compile factored
    print("\n--- Factored + torch.compile ---")
    model_c = ArcFaceFnumModel(
        len(vocab),
        n_classes,
        factored=True,
        factored_info=factored_info,
    ).to(device)
    model_c = torch.compile(model_c)
    timings_c = profile_step(model_c, batch, device)
    for name, t in timings_c.items():
        print(f"  {name:20s}: {t*1000:8.2f} ms")

    # Estimate per-epoch time
    n_train = len(train_data)
    n_batches = n_train // 256
    print(f"\n--- Estimated epoch time ({n_batches} batches) ---")
    print(f"  Factored:   {timings_f['full_fwd+bwd'] * n_batches:.1f}s")
    print(f"  Compiled:   {timings_c['full_fwd+bwd'] * n_batches:.1f}s")
    print(f"  Unfactored: {timings_u['full_fwd+bwd'] * n_batches:.1f}s")
    print(f"  Detached:   {detach_time * n_batches:.1f}s")


if __name__ == "__main__":
    main()
