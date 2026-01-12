"""Stage 2: Affiliation Classification training."""

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from labor_union_parser.char_cnn import tokenize_to_chars
from labor_union_parser.conf import MAX_TOKENS, RANDOM_SEED
from training.train_common import (
    DATA_DIR,
    DEVICE,
    WEIGHTS_DIR,
    CrossAttentionEncoder,
    load_trained_char_cnn,
)


class AffiliationDataset(Dataset):
    """Dataset for affiliation classification."""

    def __init__(self, texts, affiliations):
        self.texts = texts
        self.affiliations = affiliations
        self.all_affs = list(set(affiliations))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        char_ids, _, is_number, token_type = tokenize_to_chars(
            self.texts[idx], max_tokens=MAX_TOKENS
        )
        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "aff_idx": self.all_affs.index(self.affiliations[idx]),
        }


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss.

    Fixed to avoid in-place operations for autograd compatibility.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask for positive pairs (same label)
    labels_col = labels.unsqueeze(1)
    labels_row = labels.unsqueeze(0)
    pos_mask = (labels_col == labels_row).float()

    # Remove diagonal using (1 - eye) instead of fill_diagonal_
    eye = torch.eye(batch_size, device=device)
    pos_mask = pos_mask * (1 - eye)

    # Numerical stability
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max

    # Compute exp(sim) excluding self
    exp_sim = torch.exp(sim) * (1 - eye)
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Log probability
    log_prob = sim - torch.log(denom + 1e-8)

    # Average over positive pairs
    num_pos = pos_mask.sum(dim=1)
    loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)

    # Only include samples with at least one positive pair
    valid = num_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_per_sample[valid].mean()


def train_affiliation_classifier(train_indices=None):
    """Train Stage 2: Affiliation classifier."""
    print("\n" + "=" * 60)
    print("STAGE 2: Training Affiliation Classifier")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    # Exclude UNAFF, UNK, and MULTI (no centroid for these)
    known_df = df[~df["aff_abbr"].isin(["UNAFF", "UNK", "MULTI"])]

    # Filter to training indices if provided
    if train_indices is not None:
        known_df = known_df[known_df.index.isin(train_indices)]

    # Filter rare affiliations
    aff_counts = known_df["aff_abbr"].value_counts()
    valid_affs = aff_counts[aff_counts >= 5].index
    known_df = known_df[known_df["aff_abbr"].isin(valid_affs)]

    print(f"  Training examples: {len(known_df)}")
    print(f"  Affiliations: {known_df['aff_abbr'].nunique()}")

    # Split
    train_df, test_df = train_test_split(
        known_df, test_size=0.2, random_state=RANDOM_SEED, stratify=known_df["aff_abbr"]
    )

    train_dataset = AffiliationDataset(
        train_df["text"].tolist(), train_df["aff_abbr"].tolist()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=True
    )

    # Build model (cross-attention encoder)
    char_cnn = load_trained_char_cnn().to(DEVICE)
    model = CrossAttentionEncoder(
        char_cnn, embed_dim=64, num_embed_dim=8, num_heads=4
    ).to(DEVICE)

    # Train - freeze char_cnn, train attention and projector
    for param in model.char_cnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.query, "lr": 1e-3},
            {"params": model.cross_attn.parameters(), "lr": 1e-3},
            {"params": model.projector.parameters(), "lr": 1e-3},
            {"params": model.num_embed.parameters(), "lr": 1e-4},
        ]
    )

    print("\n  Training...")
    model.train()
    for epoch in tqdm(range(50), desc="  Epochs"):
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            aff_idx = batch["aff_idx"].to(DEVICE)

            optimizer.zero_grad()
            embeddings = model(char_ids, token_type, is_number)
            loss = supervised_contrastive_loss(embeddings, aff_idx)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

    # Compute centroids
    model.eval()
    aff_list = train_dataset.all_affs
    aff_embeddings = {i: [] for i in range(len(aff_list))}

    with torch.no_grad():
        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            aff_idx = batch["aff_idx"]
            embeddings = model(char_ids, token_type, is_number)
            for i, aff in enumerate(aff_idx.tolist()):
                aff_embeddings[aff].append(embeddings[i].cpu())

    centroids = []
    for i in range(len(aff_list)):
        if aff_embeddings[i]:
            centroid = F.normalize(
                torch.stack(aff_embeddings[i]).mean(dim=0), p=2, dim=0
            )
            centroids.append(centroid)
        else:
            centroids.append(torch.zeros(64))
    centroids = torch.stack(centroids).to(DEVICE)

    # Evaluate
    correct = 0
    total = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Evaluating"):
        char_ids, _, is_number, token_type = tokenize_to_chars(
            row["text"], max_tokens=MAX_TOKENS
        )
        char_ids_t = torch.tensor([char_ids], dtype=torch.long, device=DEVICE)
        token_type_t = torch.tensor([token_type], dtype=torch.long, device=DEVICE)
        is_number_t = torch.tensor([is_number], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            emb = model(char_ids_t, token_type_t, is_number_t)
        similarities = torch.matmul(emb, centroids.T)
        pred_idx = similarities.argmax(dim=1).item()
        pred_aff = aff_list[pred_idx]

        if pred_aff == row["aff_abbr"]:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"\n  Classification accuracy: {accuracy:.4f}")

    # Save
    model_path = WEIGHTS_DIR / "contrastive_aff_model.pt"
    centroids_path = WEIGHTS_DIR / "contrastive_aff_centroids.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aff_list": aff_list,
        },
        model_path,
    )
    torch.save(centroids.cpu(), centroids_path)

    print(f"\n  Saved model to {model_path}")
    print(f"  Saved centroids to {centroids_path}")

    return model, aff_list, centroids
