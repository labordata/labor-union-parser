"""
ANCE-style hard negative mining for dual-task training.

Provides:
- encode_all_records: encode the full record corpus
- mine_hard_candidates: find top-k candidates per training query
- ANCEMiningCallback: Lightning callback that re-mines every N epochs
"""

import lightning as L
import torch
from dataset import encode_query_batch, encode_record_batch
from tqdm import tqdm


def encode_all_records(model, fnum_to_records, vocab, batch_size=512):
    """
    Encode ALL records once at start of mining epoch.
    No filtering - include every record variant.
    Returns embeddings (on GPU), records list, and f_num tensor.
    """
    model.eval()
    device = next(model.parameters()).device

    all_records = []
    record_fnums = []

    for fnum, records in fnum_to_records.items():
        for rec in records:
            all_records.append(rec)
            record_fnums.append(fnum)

    all_record_embs = []
    for i in range(0, len(all_records), batch_size):
        batch_recs = all_records[i : i + batch_size]
        rec_batch = encode_record_batch(batch_recs, vocab)
        rec_batch = {k: v.to(device) for k, v in rec_batch.items()}

        with torch.no_grad():
            field_emb, _ = model.record_encoder(
                rec_batch["union_idx"],
                rec_batch["desig_idx"],
                rec_batch["prefix_hash"],
                rec_batch["num_hash"],
                rec_batch["num_val"],
                rec_batch["suffix_idx"],
                rec_batch["unit_id_idx"],
            )
            emb = model.dual_tower.encode_record(field_emb)
        all_record_embs.append(emb)  # Keep on GPU

    all_record_embs = torch.cat(all_record_embs, dim=0)  # [M, 128] on GPU
    record_fnums = torch.tensor(record_fnums)  # CPU is fine for indexing

    return all_record_embs, all_records, record_fnums


def mine_hard_candidates(
    model,
    train_examples,
    all_record_embs,
    all_records,
    record_fnums,
    vocab,
    k=10,
    query_batch_size=256,
):
    """
    Mine top-k most similar records for all training examples.

    Uses batched matrix multiplication for efficiency - encodes all queries
    and computes all similarities in batches on GPU.
    """
    model.eval()
    device = next(model.parameters()).device

    # Filter to train examples only
    train_indices = [
        i for i, ex in enumerate(train_examples) if ex.get("split") == "train"
    ]

    # Move record embeddings to GPU if not already
    all_record_embs = all_record_embs.to(device)  # [M, 128]

    # Process queries in batches
    for batch_start in tqdm(
        range(0, len(train_indices), query_batch_size), desc="Mining"
    ):
        batch_indices = train_indices[batch_start : batch_start + query_batch_size]
        batch_queries = [train_examples[i]["query"] for i in batch_indices]

        # Encode batch of queries
        query_batch = encode_query_batch(batch_queries)
        query_batch = {k: v.to(device) for k, v in query_batch.items()}

        with torch.no_grad():
            token_emb, mask = model.query_encoder(
                query_batch["char_ids"],
                query_batch["is_number"],
                query_batch["numeric_ids"],
            )
            query_embs = model.dual_tower.encode_query(token_emb, mask)  # [B, 128]

            # Compute similarities: [B, M]
            sims = torch.matmul(query_embs, all_record_embs.T)

            # Get top-k for all queries at once: [B, k]
            topk_indices = sims.topk(k, dim=1).indices  # [B, k]

        # Store results
        topk_indices = topk_indices.cpu().tolist()
        for i, ex_idx in enumerate(batch_indices):
            train_examples[ex_idx]["mined_candidates"] = [
                {"f_num": record_fnums[j].item(), "record": all_records[j]}
                for j in topk_indices[i]
            ]

    return train_examples


class ANCEMiningCallback(L.Callback):
    """Re-mine hard candidates every `mine_every` epochs (including epoch 0)."""

    def __init__(self, mine_every, mine_k):
        super().__init__()
        self.mine_every = mine_every
        self.mine_k = mine_k

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.mine_every != 0:
            return

        print(f"\nMining hard candidates at epoch {epoch}...")
        datamodule = trainer.datamodule

        all_record_embs, all_records, record_fnums = encode_all_records(
            pl_module.model, datamodule.fnum_to_records, datamodule.vocab
        )
        print(f"  Encoded {len(all_records)} records")

        mine_hard_candidates(
            pl_module.model,
            datamodule.train_ex,
            all_record_embs,
            all_records,
            record_fnums,
            datamodule.vocab,
            k=self.mine_k,
        )
        pl_module.model.train()
        torch.zeros(1, device=next(pl_module.parameters()).device).item()
        print("  Mining complete — dataloader will rebuild")
