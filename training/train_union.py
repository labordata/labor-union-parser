"""Stage 1: Union Detection training."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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


class UnionDataset(Dataset):
    """Dataset for union vs non-union classification."""

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

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
            "label": self.labels[idx],
        }


def one_class_contrastive_loss(embeddings, labels, temperature=0.1):
    """One-class contrastive loss: only union examples form positive pairs.

    Fixed to avoid in-place operations for autograd compatibility.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Similarity matrix
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask for positive pairs (both are unions)
    labels_row = labels.unsqueeze(0)
    labels_col = labels.unsqueeze(1)
    pos_mask = ((labels_row == 1) & (labels_col == 1)).float()

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
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)

    # Only include union samples with positive pairs
    union_mask = (labels == 1) & (num_pos > 0)
    if union_mask.sum() > 0:
        loss = -mean_log_prob_pos[union_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def train_union_detector(train_indices=None):
    """Train Stage 1: Union vs Non-Union detector."""
    print("\n" + "=" * 60)
    print("STAGE 1: Training Union Detector")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")
    union_df = df[~df["aff_abbr"].isin(["UNK"])]

    # Filter to training indices if provided
    if train_indices is not None:
        union_df = union_df[union_df.index.isin(train_indices)]

    # Load non-union examples
    nonunion_path = DATA_DIR / "nonunion_examples.csv"
    nonunion_df = pd.read_csv(nonunion_path)
    nonunion_texts = nonunion_df["text"].tolist()

    # Load additional union examples if available
    additional_path = DATA_DIR / "unions_model_missed.csv"
    additional_texts = []
    if additional_path.exists():
        additional_texts = pd.read_csv(additional_path)["text"].tolist()

    # Sample union examples
    union_sample = union_df.sample(
        n=min(10000, len(union_df)), random_state=RANDOM_SEED
    )
    union_texts = union_sample["text"].tolist() + additional_texts

    print(f"  Union examples: {len(union_texts)}")
    print(f"  Non-union examples: {len(nonunion_texts)}")

    # Create dataset
    all_texts = union_texts + nonunion_texts
    all_labels = [1] * len(union_texts) + [0] * len(nonunion_texts)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts,
        all_labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=all_labels,
    )

    train_dataset = UnionDataset(train_texts, train_labels)
    test_dataset = UnionDataset(test_texts, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

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
    for epoch in tqdm(range(30), desc="  Epochs"):
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            embeddings = model(char_ids, token_type, is_number)
            loss = one_class_contrastive_loss(embeddings, labels)

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

    # Compute union centroid
    model.eval()
    union_embs = []
    with torch.no_grad():
        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            labels = batch["label"]
            embeddings = model(char_ids, token_type, is_number)
            for i, label in enumerate(labels.tolist()):
                if label == 1:
                    union_embs.append(embeddings[i].cpu())

    union_centroid = F.normalize(torch.stack(union_embs).mean(dim=0), p=2, dim=0).to(
        DEVICE
    )

    # Find optimal threshold
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            labels = batch["label"]
            embeddings = model(char_ids, token_type, is_number)
            sims = torch.matmul(embeddings, union_centroid.unsqueeze(0).T).squeeze(-1)
            y_true.extend(labels.tolist())
            y_scores.extend(sims.cpu().tolist())

    y_true, y_scores = np.array(y_true), np.array(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    accuracy = accuracy_score(y_true, (y_scores > optimal_threshold).astype(int))
    roc_auc = roc_auc_score(y_true, y_scores)

    print("\n  Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ROC-AUC: {roc_auc:.4f}")
    print(f"    Optimal threshold: {optimal_threshold:.4f}")

    # Save
    save_path = WEIGHTS_DIR / "union_detector.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "union_centroid": union_centroid.cpu(),
            "optimal_threshold": optimal_threshold,
        },
        save_path,
    )
    print(f"\n  Saved to {save_path}")

    return model, union_centroid, optimal_threshold
