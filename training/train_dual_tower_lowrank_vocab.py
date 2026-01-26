"""
Dual tower model with LOW-RANK f_num embeddings.

Instead of each f_num getting an independent 32-dim offset,
each f_num gets 8 latent factor weights that project through
a shared 8x32 matrix. This forces the model to learn reusable
latent structure (e.g., "Mailers sector", "IUE-CWA merger").

e_i = h_struct + (latent_i @ shared_factors)

Where:
- latent_i is a small (8-dim) per-f_num embedding
- shared_factors is an 8x32 matrix learned globally
"""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.char_cnn import NUMBER_VOCAB, CharacterCNN, tokenize_to_chars

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"

EMBED_DIM = 128  # Shared embedding dimension
LATENT_DIM = 8  # Low-rank latent dimension for f_num
FNUM_EMBED_DIM = 32  # Output dimension for f_num offset (after projection)
SHRINKAGE_LAMBDA = 0.01  # L2 penalty weight


class QueryTower(nn.Module):
    """Text encoder for query strings using CharCNN."""

    def __init__(
        self, char_cnn, frozen_num_embed, embed_dim=EMBED_DIM, frozen_num_dim=32
    ):
        super().__init__()
        self.char_cnn = char_cnn
        self.frozen_num_dim = frozen_num_dim

        # Shared frozen number embedding (passed in from parent)
        self.frozen_num_embed = frozen_num_embed

        # is_number feature embedding
        self.num_embed = nn.Embedding(2, 8)

        input_dim = char_cnn.embed_dim + frozen_num_dim + 8

        # Simple attention pooling
        self.attn_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            batch_first=True,
        )

        # Projection to shared space
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, char_ids, token_type, is_number, numeric_ids):
        batch_size = char_ids.shape[0]

        # CharCNN embeddings (zeroed for numbers)
        char_emb = self.char_cnn(char_ids)
        is_number_mask = is_number.unsqueeze(-1).float()
        char_emb = char_emb * (1 - is_number_mask)

        # Frozen number embeddings
        frozen_emb = self.frozen_num_embed(numeric_ids) * is_number_mask

        # is_number feature
        num_feature = self.num_embed(is_number)

        # Combine
        token_emb = torch.cat([char_emb, frozen_emb, num_feature], dim=-1)

        # Attention pooling
        key_padding_mask = token_type == 4  # padding tokens
        query = self.attn_query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(
            query, token_emb, token_emb, key_padding_mask=key_padding_mask
        )
        pooled = pooled.squeeze(1)

        # Project and normalize
        emb = self.projector(pooled)
        return F.normalize(emb, p=2, dim=-1)


class RecordTowerLowRank(nn.Module):
    """Structured encoder with LOW-RANK f_num offset embeddings."""

    def __init__(
        self,
        num_union_names,
        num_desig_names,
        num_fnums,
        num_prefixes,
        num_suffixes,
        frozen_num_embed,
        embed_dim=EMBED_DIM,
        union_name_embed_dim=64,
        num_embed_dim=32,
        latent_dim=LATENT_DIM,
        fnum_embed_dim=FNUM_EMBED_DIM,
        prefix_suffix_embed_dim=16,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.fnum_embed_dim = fnum_embed_dim

        # Union name embedding (learned)
        self.union_name_embed = nn.Embedding(num_union_names, union_name_embed_dim)

        # Designation name embedding (learned)
        self.desig_name_embed = nn.Embedding(num_desig_names + 1, 16)

        # Shared frozen number embedding for desig_num only
        self.frozen_num_embed = frozen_num_embed

        # Learnable prefix/suffix embeddings (small dedicated vocabs)
        self.prefix_embed = nn.Embedding(
            num_prefixes, prefix_suffix_embed_dim, padding_idx=0
        )
        self.suffix_embed = nn.Embedding(
            num_suffixes, prefix_suffix_embed_dim, padding_idx=0
        )

        # Magnitude embedding
        self.num_magnitude_embed = nn.Linear(1, 16)

        # LOW-RANK f_num embeddings:
        # Each f_num gets a small latent vector
        self.fnum_latent = nn.Embedding(num_fnums + 1, latent_dim)
        nn.init.normal_(self.fnum_latent.weight, mean=0, std=0.01)

        # Shared factor matrix projects latent -> full dimension
        # All f_nums share this, forcing them to use common latent factors
        self.fnum_factors = nn.Linear(latent_dim, fnum_embed_dim, bias=False)
        nn.init.orthogonal_(
            self.fnum_factors.weight.T
        )  # Initialize factors to be orthogonal

        # Input: union_name(64) + desig_name(16) + prefix(16) + desig_num(32) + suffix(16) + magnitude(16)
        struct_dim = (
            union_name_embed_dim
            + 16
            + prefix_suffix_embed_dim
            + num_embed_dim
            + prefix_suffix_embed_dim
            + 16
        )

        # Structural projection
        self.struct_projector = nn.Sequential(
            nn.Linear(struct_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # F_num offset projection (projects from fnum_embed_dim to embed_dim)
        self.fnum_projector = nn.Linear(fnum_embed_dim, embed_dim)

    def forward(
        self,
        union_name_ids,
        desig_name_ids,
        prefix_ids,
        desig_num_hashes,
        suffix_ids,
        desig_nums,
        fnum_ids,
    ):
        # Union name embedding
        union_emb = self.union_name_embed(union_name_ids)

        # Designation name embedding
        desig_name_emb = self.desig_name_embed(desig_name_ids)

        # Learnable prefix/suffix embeddings (semantic)
        prefix_emb = self.prefix_embed(prefix_ids)
        suffix_emb = self.suffix_embed(suffix_ids)

        # Frozen embedding for desig_num only (indexical)
        num_emb = self.frozen_num_embed(desig_num_hashes)

        # Magnitude embedding
        magnitude = torch.log1p(desig_nums.float().clamp(min=0)).unsqueeze(-1)
        mag_emb = self.num_magnitude_embed(magnitude)

        # Structural embedding (the "prior" h_struct)
        struct_combined = torch.cat(
            [union_emb, desig_name_emb, prefix_emb, num_emb, suffix_emb, mag_emb],
            dim=-1,
        )
        h_struct = self.struct_projector(struct_combined)

        # LOW-RANK f_num offset:
        # latent_i @ shared_factors -> full dimension offset
        latent_i = self.fnum_latent(fnum_ids)  # [batch, latent_dim]
        z_i = self.fnum_factors(latent_i)  # [batch, fnum_embed_dim]
        z_i_projected = self.fnum_projector(z_i)

        # Final embedding: e_i = h_struct + z_i
        emb = h_struct + z_i_projected

        return F.normalize(emb, p=2, dim=-1)

    def get_fnum_latent_embeddings(self):
        """Return the raw latent embeddings for L2 penalty."""
        return self.fnum_latent.weight

    def get_fnum_full_embeddings(self):
        """Return full f_num embeddings (latent @ factors) for analysis."""
        return self.fnum_factors(self.fnum_latent.weight)


class DualTowerModelLowRank(nn.Module):
    """Combined dual tower model with low-rank f_num embeddings."""

    def __init__(
        self,
        char_cnn,
        num_union_names,
        num_desig_names,
        num_fnums,
        num_prefixes,
        num_suffixes,
        embed_dim=EMBED_DIM,
        frozen_num_dim=32,
        latent_dim=LATENT_DIM,
    ):
        super().__init__()

        # Shared frozen number embedding for both towers (used for desig_num only on record side)
        self.frozen_num_embed = nn.Embedding(len(NUMBER_VOCAB), frozen_num_dim)
        self.frozen_num_embed.weight.requires_grad = False

        self.query_tower = QueryTower(
            char_cnn, self.frozen_num_embed, embed_dim, frozen_num_dim
        )
        self.record_tower = RecordTowerLowRank(
            num_union_names,
            num_desig_names,
            num_fnums,
            num_prefixes,
            num_suffixes,
            self.frozen_num_embed,
            embed_dim,
            num_embed_dim=frozen_num_dim,
            latent_dim=latent_dim,
        )

    def encode_queries(self, char_ids, token_type, is_number, numeric_ids):
        return self.query_tower(char_ids, token_type, is_number, numeric_ids)

    def encode_records(
        self,
        union_name_ids,
        desig_name_ids,
        prefix_ids,
        desig_num_hashes,
        suffix_ids,
        desig_nums,
        fnum_ids,
    ):
        return self.record_tower(
            union_name_ids,
            desig_name_ids,
            prefix_ids,
            desig_num_hashes,
            suffix_ids,
            desig_nums,
            fnum_ids,
        )

    def get_fnum_latent_embeddings(self):
        return self.record_tower.get_fnum_latent_embeddings()

    def get_fnum_full_embeddings(self):
        return self.record_tower.get_fnum_full_embeddings()


def normalize_designation(s: str) -> str:
    """Normalize designation tokens by stripping leading zeros from numbers."""
    if not s:
        return s
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


def load_vocabularies():
    """Load vocabulary lookups from vocabularies.json."""
    print(f"Loading vocabularies from {VOCAB_PATH}...")

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    union_name_to_idx = vocab["union_name_to_idx"]
    desig_name_to_idx = vocab["desig_name_to_idx"]
    fnum_to_idx = {int(k): v for k, v in vocab["fnum_to_idx"].items()}
    prefix_to_idx = vocab["prefix_to_idx"]
    suffix_to_idx = vocab["suffix_to_idx"]

    print(f"  {len(union_name_to_idx)} union names")
    print(f"  {len(desig_name_to_idx)} designation names")
    print(f"  {len(fnum_to_idx)} f_nums")
    print(f"  {len(prefix_to_idx)} prefixes")
    print(f"  {len(suffix_to_idx)} suffixes")

    return (
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
    )


def load_training_examples():
    """Load training examples from training_examples.json."""
    print(f"Loading training examples from {EXAMPLES_PATH}...")

    with open(EXAMPLES_PATH) as f:
        all_examples = json.load(f)

    # Organize examples by split, keeping records with each example
    train_examples = []
    val_examples = []
    test_examples = []

    # Also build fnum_to_records for evaluation (needs all record variants)
    fnum_to_records = {}

    for ex in all_examples:
        split = ex["split"]
        f_num = ex["f_num"]

        # Store records for this f_num (may see same f_num multiple times, that's ok)
        if f_num not in fnum_to_records:
            fnum_to_records[f_num] = ex["records"]

        if split == "train":
            train_examples.append(ex)
        elif split == "val":
            val_examples.append(ex)
        else:  # test
            test_examples.append(ex)

    total_variants = sum(len(v) for v in fnum_to_records.values())
    print(f"  {len(fnum_to_records)} f_nums with {total_variants} record variants")
    print(f"  Train: {len(train_examples)}")
    print(f"  Val: {len(val_examples)}")
    print(f"  Test: {len(test_examples)}")

    return (
        train_examples,
        val_examples,
        test_examples,
        fnum_to_records,
    )


class DualTowerDataset(Dataset):
    """Dataset for dual tower training."""

    def __init__(
        self,
        examples,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        max_tokens=40,
    ):
        self.examples = examples
        self.union_name_to_idx = union_name_to_idx
        self.desig_name_to_idx = desig_name_to_idx
        self.fnum_to_idx = fnum_to_idx
        self.prefix_to_idx = prefix_to_idx
        self.suffix_to_idx = suffix_to_idx
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query_text = ex["query"]
        # Sample a random variant from this example's records (data augmentation)
        record = random.choice(ex["records"])

        char_ids, _, is_number, token_type, numeric_ids = tokenize_to_chars(
            query_text, max_tokens=self.max_tokens
        )

        union_name_idx = self.union_name_to_idx[record["union_name"]]
        desig_name_idx = self.desig_name_to_idx.get(record["desig_name"], 0)
        fnum_idx = self.fnum_to_idx.get(record["f_num"], 0)

        # Normalize prefix/suffix (strip leading zeros) before lookup
        prefix_norm = (
            normalize_designation(record["prefix"]) if record["prefix"] else ""
        )
        suffix_norm = (
            normalize_designation(record["suffix"]) if record["suffix"] else ""
        )
        # Use dedicated small vocabs for prefix/suffix (learnable)
        prefix_idx = self.prefix_to_idx.get(prefix_norm, 0)
        suffix_idx = self.suffix_to_idx.get(suffix_norm, 0)
        # Use NUMBER_VOCAB for desig_num (frozen, shared with query)
        desig_vocab = NUMBER_VOCAB.get(str(record["desig_num"]), NUMBER_VOCAB["<UNK>"])

        record_key = f"{record['union_name']}|||{record['desig_name']}|||{record['prefix']}|||{record['desig_num']}|||{record['suffix']}|||{record['f_num']}"

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "numeric_ids": torch.tensor(numeric_ids, dtype=torch.long),
            "union_name_idx": torch.tensor(union_name_idx, dtype=torch.long),
            "desig_name_idx": torch.tensor(desig_name_idx, dtype=torch.long),
            "prefix_idx": torch.tensor(prefix_idx, dtype=torch.long),
            "desig_hash": torch.tensor(desig_vocab, dtype=torch.long),
            "suffix_idx": torch.tensor(suffix_idx, dtype=torch.long),
            "desig_num": torch.tensor(record["desig_num"], dtype=torch.long),
            "fnum_idx": torch.tensor(fnum_idx, dtype=torch.long),
            "record_key": record_key,
        }


def collate_fn(batch):
    return {
        "char_ids": torch.stack([b["char_ids"] for b in batch]),
        "token_type": torch.stack([b["token_type"] for b in batch]),
        "is_number": torch.stack([b["is_number"] for b in batch]),
        "numeric_ids": torch.stack([b["numeric_ids"] for b in batch]),
        "union_name_idx": torch.stack([b["union_name_idx"] for b in batch]),
        "desig_name_idx": torch.stack([b["desig_name_idx"] for b in batch]),
        "prefix_idx": torch.stack([b["prefix_idx"] for b in batch]),
        "desig_hash": torch.stack([b["desig_hash"] for b in batch]),
        "suffix_idx": torch.stack([b["suffix_idx"] for b in batch]),
        "desig_num": torch.stack([b["desig_num"] for b in batch]),
        "fnum_idx": torch.stack([b["fnum_idx"] for b in batch]),
        "record_keys": [b["record_key"] for b in batch],
    }


def contrastive_loss_with_shrinkage(
    query_embs,
    record_embs,
    record_keys,
    fnum_latent_embeddings,
    fnum_ids,
    temperature=0.07,
    shrinkage_lambda=SHRINKAGE_LAMBDA,
):
    """
    Supervised contrastive loss with shrinkage on LATENT f_num embeddings.

    We penalize the latent embeddings (8-dim), not the full projected embeddings.
    This encourages sparse use of latent factors.
    """
    batch_size = query_embs.shape[0]
    device = query_embs.device

    # Standard contrastive loss
    similarities = torch.matmul(query_embs, record_embs.T) / temperature
    labels = torch.arange(batch_size, device=device)
    matching_loss = F.cross_entropy(similarities, labels)

    # L2 shrinkage penalty on ALL latent embeddings (not just batch)
    shrinkage_loss = (fnum_latent_embeddings**2).mean()

    total_loss = matching_loss + shrinkage_lambda * shrinkage_loss

    return total_loss, matching_loss.item(), shrinkage_loss.item()


def train_dual_tower(
    epochs=20,
    batch_size=128,
    lr=1e-4,
    shrinkage_lambda=SHRINKAGE_LAMBDA,
    latent_dim=LATENT_DIM,
    resume_from=None,
):
    """Train dual tower model with low-rank f_num embeddings."""
    print(f"Using device: {DEVICE}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Shrinkage lambda: {shrinkage_lambda}")
    if resume_from:
        print(f"Resuming from: {resume_from}")

    # Load vocabularies from JSON
    (
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
    ) = load_vocabularies()

    # Load training examples from JSON
    (
        train_examples,
        val_examples,
        test_examples,
        fnum_to_records,
    ) = load_training_examples()

    # Create datasets
    train_dataset = DualTowerDataset(
        train_examples,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
    )
    val_dataset = DualTowerDataset(
        val_examples,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Initialize model
    char_cnn = CharacterCNN(embed_dim=64, char_embed_dim=16)
    model = DualTowerModelLowRank(
        char_cnn,
        num_union_names=len(union_name_to_idx),
        num_desig_names=len(desig_name_to_idx),
        num_fnums=len(fnum_to_idx),
        num_prefixes=len(prefix_to_idx),
        num_suffixes=len(suffix_to_idx),
        latent_dim=latent_dim,
    )
    model.to(DEVICE)

    # Load checkpoint if resuming
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from {resume_from}")

    # Count parameters
    full_rank_params = len(fnum_to_idx) * FNUM_EMBED_DIM
    low_rank_params = len(fnum_to_idx) * latent_dim + latent_dim * FNUM_EMBED_DIM
    print(
        f"F_num embedding parameters: {low_rank_params:,} (low-rank) vs {full_rank_params:,} (full-rank)"
    )
    print(f"Parameter reduction: {(1 - low_rank_params/full_rank_params)*100:.1f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Train
        model.train()
        train_match_loss = 0
        train_shrink_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            numeric_ids = batch["numeric_ids"].to(DEVICE)
            union_name_idx = batch["union_name_idx"].to(DEVICE)
            desig_name_idx = batch["desig_name_idx"].to(DEVICE)
            prefix_idx = batch["prefix_idx"].to(DEVICE)
            desig_hash = batch["desig_hash"].to(DEVICE)
            suffix_idx = batch["suffix_idx"].to(DEVICE)
            desig_num = batch["desig_num"].to(DEVICE)
            fnum_idx = batch["fnum_idx"].to(DEVICE)

            query_embs = model.encode_queries(
                char_ids, token_type, is_number, numeric_ids
            )
            record_embs = model.encode_records(
                union_name_idx,
                desig_name_idx,
                prefix_idx,
                desig_hash,
                suffix_idx,
                desig_num,
                fnum_idx,
            )

            fnum_latent = model.get_fnum_latent_embeddings()
            loss, match_l, shrink_l = contrastive_loss_with_shrinkage(
                query_embs,
                record_embs,
                batch["record_keys"],
                fnum_latent,
                fnum_idx,
                shrinkage_lambda=shrinkage_lambda,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_match_loss += match_l
            train_shrink_loss += shrink_l

        train_match_loss /= len(train_loader)
        train_shrink_loss /= len(train_loader)

        # Validate
        model.eval()
        val_match_loss = 0
        val_shrink_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                char_ids = batch["char_ids"].to(DEVICE)
                token_type = batch["token_type"].to(DEVICE)
                is_number = batch["is_number"].to(DEVICE)
                numeric_ids = batch["numeric_ids"].to(DEVICE)
                union_name_idx = batch["union_name_idx"].to(DEVICE)
                desig_name_idx = batch["desig_name_idx"].to(DEVICE)
                prefix_idx = batch["prefix_idx"].to(DEVICE)
                desig_hash = batch["desig_hash"].to(DEVICE)
                suffix_idx = batch["suffix_idx"].to(DEVICE)
                desig_num = batch["desig_num"].to(DEVICE)
                fnum_idx = batch["fnum_idx"].to(DEVICE)

                query_embs = model.encode_queries(
                    char_ids, token_type, is_number, numeric_ids
                )
                record_embs = model.encode_records(
                    union_name_idx,
                    desig_name_idx,
                    prefix_idx,
                    desig_hash,
                    suffix_idx,
                    desig_num,
                    fnum_idx,
                )

                fnum_latent = model.get_fnum_latent_embeddings()
                _, match_l, shrink_l = contrastive_loss_with_shrinkage(
                    query_embs,
                    record_embs,
                    batch["record_keys"],
                    fnum_latent,
                    fnum_idx,
                    shrinkage_lambda=shrinkage_lambda,
                )
                val_match_loss += match_l
                val_shrink_loss += shrink_l

        val_match_loss /= len(val_loader)
        val_shrink_loss /= len(val_loader)

        # Compute latent embedding stats
        latent_embs = model.get_fnum_latent_embeddings()
        latent_norms = latent_embs.norm(dim=1)
        avg_latent_norm = latent_norms.mean().item()
        max_latent_norm = latent_norms.max().item()

        # Also compute full embedding stats for comparison
        full_embs = model.get_fnum_full_embeddings()
        full_norms = full_embs.norm(dim=1)
        avg_full_norm = full_norms.mean().item()
        max_full_norm = full_norms.max().item()

        print(
            f"Epoch {epoch+1}: train_match={train_match_loss:.4f}, train_shrink={train_shrink_loss:.4f}, "
            f"val_match={val_match_loss:.4f}, val_shrink={val_shrink_loss:.4f}"
        )
        print(
            f"          latent_norm: avg={avg_latent_norm:.4f}, max={max_latent_norm:.4f} | "
            f"full_norm: avg={avg_full_norm:.4f}, max={max_full_norm:.4f}"
        )

        # Evaluate retrieval on validation set
        top1_acc = evaluate_retrieval(
            model,
            union_name_to_idx,
            desig_name_to_idx,
            fnum_to_idx,
            prefix_to_idx,
            suffix_to_idx,
            val_examples,
            fnum_to_records,
            verbose=False,
        )
        print(f"          Val Top-1 Accuracy: {top1_acc:.4f}")

        # Save best
        if val_match_loss < best_val_loss:
            best_val_loss = val_match_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "union_name_to_idx": union_name_to_idx,
                    "desig_name_to_idx": desig_name_to_idx,
                    "fnum_to_idx": fnum_to_idx,
                    "prefix_to_idx": prefix_to_idx,
                    "suffix_to_idx": suffix_to_idx,
                    "latent_dim": latent_dim,
                },
                Path(__file__).parent / "dual_tower_lowrank_vocab.pt",
            )
            print("  Saved new best model")

    return (
        model,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        fnum_to_records,
        test_examples,
    )


def record_to_key(r):
    """Convert record dict to a hashable key."""
    return (
        r["union_name"],
        r["desig_name"],
        r["prefix"],
        r["desig_num"],
        r["suffix"],
        r.get("f_num", 0),
    )


def evaluate_retrieval(
    model,
    union_name_to_idx,
    desig_name_to_idx,
    fnum_to_idx,
    prefix_to_idx,
    suffix_to_idx,
    examples,
    fnum_to_records,
    batch_size=128,
    verbose=True,
):
    """Evaluate retrieval accuracy using f_num-level success metric."""
    model.eval()

    # Extract queries and fnums from examples
    queries = [ex["query"] for ex in examples]
    fnums = [ex["f_num"] for ex in examples]

    # Build index from all record variants in fnum_to_records
    all_records = []
    for fnum, records in fnum_to_records.items():
        for r in records:
            all_records.append(r)

    unique_record_keys = list(set(record_to_key(r) for r in all_records))
    if verbose:
        print(f"Building index of {len(unique_record_keys)} unique records...")

    # Map record key to f_num for evaluation
    key_to_fnum = {}
    for r in all_records:
        key_to_fnum[record_to_key(r)] = r["f_num"]

    record_embs_list = []
    for i in range(0, len(unique_record_keys), batch_size):
        batch_keys = unique_record_keys[i : i + batch_size]

        union_name_idxs = torch.tensor(
            [union_name_to_idx[k[0]] for k in batch_keys], device=DEVICE
        )
        desig_name_idxs = torch.tensor(
            [desig_name_to_idx.get(k[1], 0) for k in batch_keys], device=DEVICE
        )
        # Use dedicated small vocabs for prefix/suffix (learnable)
        prefix_idxs = torch.tensor(
            [
                prefix_to_idx.get(normalize_designation(k[2]), 0) if k[2] else 0
                for k in batch_keys
            ],
            device=DEVICE,
        )
        # Use NUMBER_VOCAB for desig_num (frozen, shared with query)
        desig_hashes = torch.tensor(
            [NUMBER_VOCAB.get(str(k[3]), NUMBER_VOCAB["<UNK>"]) for k in batch_keys],
            device=DEVICE,
        )
        suffix_idxs = torch.tensor(
            [
                suffix_to_idx.get(normalize_designation(k[4]), 0) if k[4] else 0
                for k in batch_keys
            ],
            device=DEVICE,
        )
        desig_nums = torch.tensor([k[3] for k in batch_keys], device=DEVICE)
        fnum_idxs = torch.tensor(
            [fnum_to_idx.get(k[5], 0) for k in batch_keys], device=DEVICE
        )

        with torch.no_grad():
            embs = model.encode_records(
                union_name_idxs,
                desig_name_idxs,
                prefix_idxs,
                desig_hashes,
                suffix_idxs,
                desig_nums,
                fnum_idxs,
            )
            record_embs_list.append(embs)

    record_embs = torch.cat(record_embs_list, dim=0)
    # Map index to f_num for retrieved records
    idx_to_fnum = [key_to_fnum[k] for k in unique_record_keys]

    # Evaluate queries
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    dataset = DualTowerDataset(
        examples,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    for batch_idx, batch in enumerate(
        tqdm(loader, desc="Evaluating", disable=not verbose)
    ):
        char_ids = batch["char_ids"].to(DEVICE)
        token_type = batch["token_type"].to(DEVICE)
        is_number = batch["is_number"].to(DEVICE)
        numeric_ids = batch["numeric_ids"].to(DEVICE)

        with torch.no_grad():
            query_embs = model.encode_queries(
                char_ids, token_type, is_number, numeric_ids
            )

        sims = torch.matmul(query_embs, record_embs.T)
        _, top_indices = torch.topk(sims, k=min(10, len(unique_record_keys)), dim=1)

        # Get the true f_nums for this batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(batch["char_ids"])
        batch_fnums = fnums[start_idx:end_idx]

        for i, true_fnum in enumerate(batch_fnums):
            preds = top_indices[i].cpu().tolist()
            retrieved_fnums = [idx_to_fnum[p] for p in preds]

            if retrieved_fnums[0] == true_fnum:
                correct_top1 += 1
            if true_fnum in retrieved_fnums[:5]:
                correct_top5 += 1
            if true_fnum in retrieved_fnums[:10]:
                correct_top10 += 1

    total = len(queries)
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    top10_acc = correct_top10 / total
    if verbose:
        print("\nRetrieval Results (f_num-level):")
        print(f"  Top-1:  {correct_top1}/{total} = {top1_acc:.4f}")
        print(f"  Top-5:  {correct_top5}/{total} = {top5_acc:.4f}")
        print(f"  Top-10: {correct_top10}/{total} = {top10_acc:.4f}")
    return top1_acc


if __name__ == "__main__":
    resume_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        resume_path = Path(__file__).parent / "dual_tower_lowrank_vocab.pt"

    (
        model,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        fnum_to_records,
        test_examples,
    ) = train_dual_tower(
        epochs=20,
        batch_size=128,
        lr=1e-4,
        shrinkage_lambda=SHRINKAGE_LAMBDA,
        latent_dim=LATENT_DIM,
        resume_from=resume_path,
    )

    # Evaluate on test set
    print("\nFinal evaluation on test set:")
    evaluate_retrieval(
        model,
        union_name_to_idx,
        desig_name_to_idx,
        fnum_to_idx,
        prefix_to_idx,
        suffix_to_idx,
        test_examples,
        fnum_to_records,
    )
