"""Shared code for training stages."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from labor_union_parser.char_cnn import CharacterCNN

DEVICE = torch.accelerator.current_accelerator()
WEIGHTS_DIR = Path(__file__).parent.parent / "src" / "labor_union_parser" / "weights"
DATA_DIR = Path(__file__).parent / "data"


def load_trained_char_cnn():
    """Load the trained CharCNN from model weights."""
    char_cnn = CharacterCNN(embed_dim=64, char_embed_dim=16)
    weights_path = WEIGHTS_DIR / "char_cnn.pt"

    if weights_path.exists():
        state = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        char_cnn_state = {}
        for k, v in state["model_state_dict"].items():
            if k.startswith("char_cnn."):
                char_cnn_state[k[len("char_cnn.") :]] = v
        char_cnn.load_state_dict(char_cnn_state)
        print("  Loaded trained CharCNN weights")
    else:
        print("  WARNING: No trained weights found, using random initialization")

    return char_cnn


class CrossAttentionEncoder(nn.Module):
    """Encoder with cross-attention pooling instead of mean pooling.

    Uses a learned query to attend over token embeddings, allowing the model
    to learn which tokens are most relevant for classification.
    """

    def __init__(
        self,
        char_cnn: CharacterCNN,
        embed_dim: int = 64,
        num_embed_dim: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        self.char_cnn = char_cnn
        self.char_embed_dim = char_cnn.embed_dim
        self.num_embed_dim = num_embed_dim
        self.input_dim = self.char_embed_dim + num_embed_dim

        # is_number embedding (0 = not number, 1 = number)
        self.num_embed = nn.Embedding(2, num_embed_dim)

        # Learned query for "what class is this?"
        self.query = nn.Parameter(torch.randn(1, 1, self.input_dim) * 0.02)

        # Cross-attention: query attends to token sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, char_ids, token_type, is_number, return_attention=False):
        batch_size = char_ids.shape[0]

        # Get token embeddings from CharCNN
        token_emb = self.char_cnn(char_ids)

        # Add is_number embedding
        num_emb = self.num_embed(is_number)
        token_emb = torch.cat([token_emb, num_emb], dim=-1)

        # Create padding mask (True = ignore)
        key_padding_mask = token_type == 4

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Cross-attention: query attends to tokens
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=token_emb,
            value=token_emb,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        # Remove sequence dimension and project
        pooled = attn_out.squeeze(1)
        proj = self.projector(pooled)
        normalized = F.normalize(proj, p=2, dim=-1)

        if return_attention:
            return normalized, attn_weights.squeeze(1)
        return normalized
