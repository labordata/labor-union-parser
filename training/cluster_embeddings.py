#!/usr/bin/env python3
"""Cluster union_name embeddings using agglomerative clustering.

Uses scipy's average-linkage agglomerative clustering with cosine distance.
Cuts the dendrogram at a cosine similarity threshold.

Usage:
    python training/cluster_embeddings.py [--checkpoint PATH] [--threshold 0.85]
    python training/cluster_embeddings.py --unlabeled-only --threshold 0.82
"""

import argparse
import csv
import glob
from collections import Counter
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

print = partial(print, flush=True)  # noqa: A001


def load_union_name_embeddings(checkpoint_path, db_path="f7.db"):
    """Load model from checkpoint, compute union_name embeddings for all distinct texts."""
    import sys

    sys.path.insert(0, ".")
    from training.train_record_embeddings import (
        MAX_TOKENS_PER_FIELD,
        NUM_BLOOM_HASHES,
        NUM_FIELDS,
        RecordEmbeddingModel,
        _tokenize,
        bloom_hash_ids,
        build_token_vocab,
        load_f7_records,
        load_fnum_labels,
    )

    # Load checkpoint — handle both lightning (.ckpt) and manual save formats
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load records (needed for both paths)
    records = load_f7_records(db_path)

    if "hyper_parameters" in ckpt:
        # Lightning checkpoint
        hparams = ckpt["hyper_parameters"]
        d_model = hparams["d_model"]
        temperature = hparams["temperature"]
        sd = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        vocab_size = sd["token_embed.weight"].shape[0]
        n_classes = sd["prototypes.weight"].shape[0]
        token_vocab = build_token_vocab(records, min_count=2)
    else:
        # Manual save format
        token_vocab = ckpt["token_vocab"]
        d_model = ckpt["hparams"]["d_model"]
        temperature = ckpt["hparams"]["temperature"]
        n_classes = len(ckpt["fnum_to_idx"])
        sd = ckpt["state_dict"]
        vocab_size = len(token_vocab)

    model = RecordEmbeddingModel(
        d_model=d_model,
        temperature=temperature,
        vocab_size=vocab_size,
        n_classes=n_classes,
    )
    model.load_state_dict(sd)
    model.eval()
    text_to_fnum = load_fnum_labels("training/data/labeled_data.csv")

    union_name_counts = Counter()
    for rec in records:
        un = rec["union_name"].strip()
        if un:
            union_name_counts[un] += 1

    distinct_texts = sorted(union_name_counts.keys())
    counts = np.array([union_name_counts[t] for t in distinct_texts])

    # Check which are labeled
    labeled = np.array([t.lower() in text_to_fnum for t in distinct_texts])
    fnum_labels = np.array([text_to_fnum.get(t.lower(), -1) for t in distinct_texts])

    # Encode each union_name through the model (field index 0)
    unk_id = token_vocab["<UNK>"]
    embeddings = []

    batch_size = 512
    for start in range(0, len(distinct_texts), batch_size):
        batch_texts = distinct_texts[start : start + batch_size]
        B = len(batch_texts)

        token_ids = np.zeros((B, NUM_FIELDS, MAX_TOKENS_PER_FIELD), dtype=np.int64)
        bloom_ids_arr = np.zeros(
            (B, NUM_FIELDS, MAX_TOKENS_PER_FIELD, NUM_BLOOM_HASHES), dtype=np.int64
        )
        is_number = np.zeros((B, NUM_FIELDS, MAX_TOKENS_PER_FIELD), dtype=np.bool_)
        token_lens = np.zeros((B, NUM_FIELDS), dtype=np.int64)

        for i, text in enumerate(batch_texts):
            tokens = _tokenize("union_name", text)
            n_tok = min(len(tokens), MAX_TOKENS_PER_FIELD)
            token_lens[i, 0] = n_tok
            for t in range(n_tok):
                tok = tokens[t]
                if tok["is_num"]:
                    is_number[i, 0, t] = True
                    bloom_ids_arr[i, 0, t] = bloom_hash_ids(tok["token"])
                else:
                    token_ids[i, 0, t] = token_vocab.get(tok["token"], unk_id)

        with torch.no_grad():
            field_embs = model.encode_fields(
                torch.from_numpy(token_ids),
                torch.from_numpy(bloom_ids_arr),
                torch.from_numpy(is_number),
                torch.from_numpy(token_lens),
            )
            # field_embs: [B, NUM_FIELDS, d_model] — take field 0 (union_name)
            union_emb = field_embs[:, 0, :]
            union_emb = F.normalize(union_emb, dim=-1)
            embeddings.append(union_emb.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()

    return distinct_texts, counts, labeled, fnum_labels, embeddings


def cluster_rac(
    embeddings, threshold=0.85, k_neighbors=30, batch_size=1000, n_processors=8
):
    """Reciprocal Agglomerative Clustering via racplusplus.

    Builds a sparse k-NN connectivity graph to avoid materializing the
    full N×N distance matrix, then runs RAC on that graph.

    Returns list of lists (cluster member indices).
    """
    import os

    import racplusplus
    from scipy.sparse import csc_matrix, lil_matrix

    N = len(embeddings)
    n_processors = min(n_processors, os.cpu_count() or 1)
    max_merge_distance = 1.0 - threshold  # cosine distance

    print(
        f"RAC clustering {N} points (threshold={threshold}, "
        f"k={k_neighbors}, batch={batch_size}, threads={n_processors})..."
    )

    # Build symmetric k-NN connectivity graph via chunked matmul
    print("  Building k-NN connectivity graph...")
    emb = embeddings.astype(np.float32)
    conn = lil_matrix((N, N), dtype=np.bool_)
    chunk = 2000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sims = emb[start:end] @ emb.T  # [chunk, N]
        # For each row, find top-k (excluding self)
        for i in range(end - start):
            row = sims[i]
            row[start + i] = -2.0  # exclude self
            top_k = np.argpartition(row, -k_neighbors)[-k_neighbors:]
            for j in top_k:
                conn[start + i, j] = True
                conn[j, start + i] = True  # symmetric
        if (start // chunk) % 10 == 0:
            print(f"    {end}/{N}")

    conn_csc = csc_matrix(conn)
    print(f"  Connectivity: {conn_csc.nnz} edges ({conn_csc.nnz / N:.1f} per node)")

    print("  Running RAC...")
    labels = racplusplus.rac(
        embeddings.astype(np.float64),
        max_merge_distance,
        conn_csc,
        "symmetric",
        batch_size,
        n_processors,
    )

    # Group by cluster label
    from collections import defaultdict

    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    clusters = list(cluster_map.values())
    clusters.sort(key=len, reverse=True)
    print(f"  {len(clusters)} clusters found")
    return clusters


def split_clusters_by_numbers(clusters, texts):
    """Post-process: split clusters where members have non-overlapping numbers.

    Within each cluster, extract number tokens from each text. Two texts are
    connected if they share at least one number token (or if either has no
    numbers). Split the cluster into connected components.

    This is the "blocking" post-processing step — texts with disjoint number
    sets should never be in the same cluster.
    """
    import sys

    sys.path.insert(0, ".")
    from training.train_record_embeddings import _tokenize

    def get_numbers(text):
        """Extract number tokens, treating letter codes as compound.

        If any word token is a single letter (a-z) and there's at least
        one number, treat as compound designation (e.g., 1199C, 751-A,
        GCC-1L) — return 2+ tokens so the caller ejects it as singleton.
        """
        tokens = _tokenize("union_name", text)
        nums = [tok["token"] for tok in tokens if tok["is_num"]]
        if nums:
            has_single_letter = any(
                not tok["is_num"] and tok["token"].isalpha() and len(tok["token"]) == 1
                for tok in tokens
            )
            if has_single_letter:
                nums.append("__suffix__")
        return frozenset(nums)

    new_clusters = []
    for cluster in clusters:
        if len(cluster) <= 1:
            new_clusters.append(cluster)
            continue

        # Extract number sets for each member
        # Compound numbers (2+ tokens) → exclude from clustering
        number_sets = []
        for i in cluster:
            nums = get_numbers(texts[i])
            number_sets.append(nums)

        # Group by number token: same single number → same sub-cluster,
        # no number → own sub-cluster, compound → excluded (singleton)
        from collections import defaultdict

        number_groups = defaultdict(list)
        no_number_members = []
        for idx_in_cluster, i in enumerate(cluster):
            nums = number_sets[idx_in_cluster]
            if len(nums) == 1:
                (num,) = nums
                number_groups[num].append(i)
            elif len(nums) == 0:
                no_number_members.append(i)
            else:
                # Compound number — exclude as singleton
                new_clusters.append([i])

        if len(number_groups) <= 1:
            # All same number or no numbers — keep as-is
            remaining = []
            for members in number_groups.values():
                remaining.extend(members)
            remaining.extend(no_number_members)
            if remaining:
                new_clusters.append(remaining)
        else:
            # Split: each number group becomes its own sub-cluster
            for members in number_groups.values():
                new_clusters.append(members)
            if no_number_members:
                new_clusters.append(no_number_members)

    new_clusters.sort(key=len, reverse=True)
    return new_clusters


def main():
    parser = argparse.ArgumentParser(
        description="RAC clustering of union_name embeddings"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (default: latest lightning checkpoint or record_embeddings.pt)",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--unlabeled-only",
        action="store_true",
        help="Only cluster unlabeled union names",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=2, help="Minimum cluster size to report"
    )
    parser.add_argument("--db", default="f7.db")
    parser.add_argument(
        "--split-numbers",
        action="store_true",
        help="Post-process: split clusters by disjoint number tokens",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint is None:
        import re as _re

        ckpts = glob.glob(
            "training/data/lightning_logs/lightning_logs/*/checkpoints/*.ckpt"
        )

        # Sort by version number (numeric)
        def _version_key(p):
            m = _re.search(r"version_(\d+)", p)
            return int(m.group(1)) if m else 0

        ckpts.sort(key=_version_key)
        if ckpts:
            args.checkpoint = ckpts[-1]
        else:
            import os

            if os.path.exists("training/data/record_embeddings.pt"):
                args.checkpoint = "training/data/record_embeddings.pt"
            else:
                print("No checkpoints found!")
                return
    # If checkpoint path is a glob pattern or was overwritten, find current file
    if not glob.glob(args.checkpoint):
        import os

        ckpt_dir = os.path.dirname(args.checkpoint)
        if os.path.isdir(ckpt_dir):
            files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            if files:
                args.checkpoint = files[-1]
    print(f"Using checkpoint: {args.checkpoint}")

    # Load embeddings
    print("Loading and encoding union names...")
    texts, counts, labeled, fnum_labels, embeddings = load_union_name_embeddings(
        args.checkpoint, args.db
    )
    print(f"  {len(texts)} distinct union names, {labeled.sum()} labeled")

    # Filter to unlabeled only if requested
    if args.unlabeled_only:
        mask = ~labeled
        indices = np.where(mask)[0]
        texts = [texts[i] for i in indices]
        counts = counts[indices]
        labeled = labeled[indices]
        fnum_labels = fnum_labels[indices]
        embeddings = embeddings[indices]
        print(f"  Filtered to {len(texts)} unlabeled texts")

    # Run RAC
    print(f"\nRunning RAC clustering (threshold={args.threshold})...")
    clusters = cluster_rac(embeddings, threshold=args.threshold)

    # Post-process: split by number tokens
    if args.split_numbers:
        n_before = len(clusters)
        clusters = split_clusters_by_numbers(clusters, texts)
        print(f"\nNumber-split: {n_before} → {len(clusters)} clusters")

    # Report
    multi_clusters = [c for c in clusters if len(c) >= args.min_cluster_size]
    singletons = len(clusters) - len(multi_clusters)
    print(
        f"\n{len(multi_clusters)} clusters with {args.min_cluster_size}+ members, "
        f"{singletons} singletons"
    )

    total_in_clusters = sum(len(c) for c in multi_clusters)
    total_records_in_clusters = sum(sum(counts[i] for i in c) for c in multi_clusters)
    print(
        f"  {total_in_clusters} texts in multi-member clusters "
        f"({total_records_in_clusters} total records)"
    )

    # Purity analysis: for clusters with 2+ labeled members, are all labels the same?
    pure = 0
    impure = 0
    impure_details = []
    for cluster in multi_clusters:
        labeled_in_cluster = [(i, fnum_labels[i]) for i in cluster if labeled[i]]
        if len(labeled_in_cluster) < 2:
            continue
        distinct_fnums = set(fn for _, fn in labeled_in_cluster)
        if len(distinct_fnums) == 1:
            pure += 1
        else:
            impure += 1
            members = sorted(cluster, key=lambda i: counts[i], reverse=True)
            impure_details.append((len(distinct_fnums), members, labeled_in_cluster))

    total_evaluated = pure + impure
    if total_evaluated > 0:
        purity = pure / total_evaluated * 100
        print(f"\n=== Threshold {args.threshold:.2f} ===")
        print(f"  Pure: {pure}, Impure: {impure}, Purity: {purity:.1f}%")

        # Show top impure clusters
        impure_details.sort(key=lambda x: x[0], reverse=True)
        print("\n  Impure clusters (top 10):")
        for n_fnums, members, lab_members in impure_details[:10]:
            print(f"    {n_fnums} distinct f_nums:")
            for i in members[:5]:
                fn_str = f", f_num={fnum_labels[i]}" if labeled[i] else ""
                print(f"      {texts[i]} (n={counts[i]}{fn_str})")

    # Show top clusters
    print("\nTop clusters:")
    for rank, cluster in enumerate(multi_clusters[:30]):
        total_count = sum(counts[i] for i in cluster)
        n_labeled_in_cluster = sum(1 for i in cluster if labeled[i])
        members = sorted(cluster, key=lambda i: counts[i], reverse=True)
        label_str = f" ({n_labeled_in_cluster} labeled)" if n_labeled_in_cluster else ""
        print(
            f"\n  Cluster {rank + 1} ({len(cluster)} texts, {total_count} records){label_str}:"
        )
        for i in members[:5]:
            lab = f" [f_num={fnum_labels[i]}]" if labeled[i] else ""
            print(f"    {texts[i]} (n={counts[i]}){lab}")
        if len(members) > 5:
            print(f"    ... and {len(members) - 5} more")

    # Save to CSV if requested
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cluster_id", "text", "count", "is_labeled", "f_num"])
            for cid, cluster in enumerate(multi_clusters):
                for i in sorted(cluster, key=lambda i: counts[i], reverse=True):
                    writer.writerow(
                        [
                            cid,
                            texts[i],
                            counts[i],
                            bool(labeled[i]),
                            int(fnum_labels[i]) if fnum_labels[i] >= 0 else "",
                        ]
                    )
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
