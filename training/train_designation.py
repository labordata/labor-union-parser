"""Stage 3: Designation Extraction training."""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from labor_union_parser.char_cnn import get_special_token_id, tokenize_to_chars
from labor_union_parser.conf import MAX_TOKENS, RANDOM_SEED
from labor_union_parser.extractor import (
    DesignationExtractor,
    create_desig_label,
    extract_desig_from_pred,
)
from training.train_common import DATA_DIR, DEVICE, WEIGHTS_DIR


class DesignationDataset(Dataset):
    """Dataset for designation extraction."""

    def __init__(self, texts, desig_nums, affiliations, aff_list):
        self.texts = texts
        self.desig_nums = desig_nums
        self.affiliations = affiliations
        self.aff_list = aff_list

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        desig_num = self.desig_nums[idx]
        aff = self.affiliations[idx]

        char_ids, tokens, is_number, token_type = tokenize_to_chars(
            text, max_tokens=MAX_TOKENS
        )

        # Get special IDs for non-word tokens
        special_ids = []
        for i, tt in enumerate(token_type):
            if tt != 0:  # Not a word
                special_ids.append(
                    get_special_token_id(tokens[i] if i < len(tokens) else "")
                )
            else:
                special_ids.append(0)

        # Create designation label
        desig_label = create_desig_label(
            text, str(desig_num) if pd.notna(desig_num) else ""
        )

        # Get affiliation index
        aff_idx = self.aff_list.index(aff) if aff in self.aff_list else 0

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "token_type": torch.tensor(token_type, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "special_ids": torch.tensor(special_ids, dtype=torch.long),
            "token_mask": torch.tensor(
                [1 if tt != 4 else 0 for tt in token_type], dtype=torch.long
            ),
            "aff_idx": aff_idx,
            "desig_label": desig_label,
            "text": text,
            "desig_num": str(desig_num) if pd.notna(desig_num) else "",
        }


def train_designation_extractor(aff_list, train_indices=None):
    """Train Stage 3: Designation extractor."""
    print("\n" + "=" * 60)
    print("STAGE 3: Training Designation Extractor")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_DIR / "labeled_data.csv")

    # Filter to known affiliations (not UNAFF/UNK) that are in our aff_list
    known_df = df[df["aff_abbr"].isin(aff_list)]

    # Filter to training indices if provided
    if train_indices is not None:
        known_df = known_df[known_df.index.isin(train_indices)]

    # Filter to examples with designation numbers
    has_desig = known_df["desig_num"].notna()
    print(f"  Examples with designation: {has_desig.sum()}")
    print(f"  Examples without designation: {(~has_desig).sum()}")

    # Use all examples (both with and without designation)
    train_df, test_df = train_test_split(
        known_df, test_size=0.2, random_state=RANDOM_SEED, stratify=known_df["aff_abbr"]
    )

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    train_dataset = DesignationDataset(
        train_df["text"].tolist(),
        train_df["desig_num"].tolist(),
        train_df["aff_abbr"].tolist(),
        aff_list,
    )

    test_dataset = DesignationDataset(
        test_df["text"].tolist(),
        test_df["desig_num"].tolist(),
        test_df["aff_abbr"].tolist(),
        aff_list,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Build model
    model = DesignationExtractor(
        num_affs=len(aff_list),
        token_embed_dim=64,
        hidden_dim=512,
        aff_embed_dim=64,
        num_attn_heads=4,
        num_feature_dim=16,
        char_embed_dim=16,
        num_attn_layers=3,
    ).to(DEVICE)

    # Load CharCNN weights
    weights_path = WEIGHTS_DIR / "char_cnn.pt"
    if weights_path.exists():
        state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        char_cnn_state = {}
        for k, v in state["model_state_dict"].items():
            if k.startswith("char_cnn."):
                char_cnn_state[k[len("char_cnn.") :]] = v
        model.char_cnn.load_state_dict(char_cnn_state)
        print("  Loaded CharCNN weights")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("\n  Training...")
    model.train()
    for epoch in tqdm(range(10), desc="  Epochs"):
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            char_ids = batch["char_ids"].to(DEVICE)
            token_mask = batch["token_mask"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            special_ids = batch["special_ids"].to(DEVICE)
            aff_idx = batch["aff_idx"].to(DEVICE)
            desig_labels = batch["desig_label"].to(DEVICE)

            optimizer.zero_grad()
            results = model(
                char_ids,
                token_mask,
                is_number,
                token_type,
                special_ids,
                aff_idx,
                desig_labels,
            )

            loss = results["desig_loss"]
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            tqdm.write(f"    Epoch {epoch+1}: loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Evaluating"):
            char_ids = batch["char_ids"].to(DEVICE)
            token_mask = batch["token_mask"].to(DEVICE)
            is_number = batch["is_number"].to(DEVICE)
            token_type = batch["token_type"].to(DEVICE)
            special_ids = batch["special_ids"].to(DEVICE)
            aff_idx = batch["aff_idx"].to(DEVICE)

            results = model(
                char_ids, token_mask, is_number, token_type, special_ids, aff_idx
            )
            preds = results["desig_pred"].cpu().tolist()

            for i, pred in enumerate(preds):
                text = batch["text"][i]
                true_desig = batch["desig_num"][i]
                pred_desig = extract_desig_from_pred(text, pred)

                # Normalize for comparison
                true_norm = str(true_desig).lstrip("0") if true_desig else ""
                if "." in true_norm:
                    true_norm = true_norm.split(".")[0]
                pred_norm = pred_desig.lstrip("0") if pred_desig else ""

                if true_norm == pred_norm:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"\n  Designation accuracy: {accuracy:.4f}")

    # Save
    save_path = WEIGHTS_DIR / "designation_extractor.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "aff_list": aff_list,
        },
        save_path,
    )
    print(f"\n  Saved to {save_path}")

    return model
