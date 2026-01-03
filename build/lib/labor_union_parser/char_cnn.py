"""Character-level CNN for computing token embeddings from characters.

Inspired by CharacterBERT and ELMo's character CNN. Instead of looking up
tokens in a vocabulary, we compute each token's embedding from its characters
using parallel CNNs with different filter sizes.

This makes the model robust to typos: "afscme" and "afcsme" produce similar
embeddings because they share most character n-grams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Character vocabulary: lowercase letters, digits, common punctuation
CHAR_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
}
# Add lowercase letters
for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
    CHAR_VOCAB[c] = len(CHAR_VOCAB)
# Add digits
for i, c in enumerate("0123456789"):
    CHAR_VOCAB[c] = len(CHAR_VOCAB)
# Add common punctuation and space
for c in " -/&,.()'\"#:":
    CHAR_VOCAB[c] = len(CHAR_VOCAB)

MAX_CHARS_PER_TOKEN = 20


def chars_to_ids(token: str, max_chars: int = MAX_CHARS_PER_TOKEN) -> list[int]:
    """Convert a token to character IDs."""
    token = token.lower()
    ids = []
    for c in token[:max_chars]:
        ids.append(CHAR_VOCAB.get(c, CHAR_VOCAB["<UNK>"]))
    # Pad to max_chars
    ids = ids + [0] * (max_chars - len(ids))
    return ids


class Highway(nn.Module):
    """Highway network layer.

    Applies a gated transformation: y = g * transform(x) + (1-g) * x
    where g is a learned gate.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        t = F.relu(self.transform(x))
        g = torch.sigmoid(self.gate(x))
        return g * t + (1 - g) * x


class CharacterCNN(nn.Module):
    """Character-level CNN that computes token embeddings from characters.

    Architecture:
    1. Character embedding lookup (small vocab ~50 chars)
    2. Parallel CNNs with filter sizes 1-7 to capture character n-grams
    3. Max-pool over character dimension
    4. Highway network for non-linear transformation
    5. Projection to desired embedding dimension

    Args:
        embed_dim: Output embedding dimension (default 64)
        char_embed_dim: Character embedding dimension (default 16)
        max_chars: Maximum characters per token (default 20)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        char_embed_dim: int = 16,
        max_chars: int = MAX_CHARS_PER_TOKEN,
    ):
        super().__init__()

        self.char_embed_dim = char_embed_dim
        self.max_chars = max_chars
        self.embed_dim = embed_dim

        # Character embeddings
        self.char_embed = nn.Embedding(len(CHAR_VOCAB), char_embed_dim, padding_idx=0)

        # Parallel CNNs with different filter sizes
        # Just 1-3 grams - keeps short words like "seiu" from overfitting
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(char_embed_dim, 32, kernel_size=1),  # unigrams
                nn.Conv1d(char_embed_dim, 64, kernel_size=2, padding=1),  # bigrams
                nn.Conv1d(char_embed_dim, 128, kernel_size=3, padding=1),  # trigrams
            ]
        )

        # Total CNN output dimension
        cnn_out_dim = 32 + 64 + 128  # = 224

        # Single highway layer (simpler)
        self.highway1 = Highway(cnn_out_dim)

        # Final projection to embedding dimension
        self.projection = nn.Linear(cnn_out_dim, embed_dim)

    def forward(self, char_ids):
        """
        Args:
            char_ids: [batch, seq_len, max_chars] character IDs for each token

        Returns:
            [batch, seq_len, embed_dim] token embeddings
        """
        batch_size, seq_len, max_chars = char_ids.shape

        # Reshape to process all tokens at once
        # [batch * seq_len, max_chars]
        char_ids_flat = char_ids.view(-1, max_chars)

        # Character embeddings: [batch * seq_len, max_chars, char_embed_dim]
        char_emb = self.char_embed(char_ids_flat)

        # Conv1d expects [batch, channels, length]
        # Transpose: [batch * seq_len, char_embed_dim, max_chars]
        char_emb = char_emb.transpose(1, 2)

        # Apply each CNN and max-pool
        conv_outputs = []
        for conv in self.convs:
            # [batch * seq_len, num_filters, ~max_chars]
            conv_out = F.relu(conv(char_emb))
            # Max-pool over character dimension: [batch * seq_len, num_filters]
            pooled = conv_out.max(dim=2)[0]
            conv_outputs.append(pooled)

        # Concatenate all CNN outputs: [batch * seq_len, 368]
        cnn_out = torch.cat(conv_outputs, dim=1)

        # Highway layer
        highway_out = self.highway1(cnn_out)

        # Project to embedding dimension: [batch * seq_len, embed_dim]
        token_emb = self.projection(highway_out)

        # Reshape back: [batch, seq_len, embed_dim]
        token_emb = token_emb.view(batch_size, seq_len, self.embed_dim)

        return token_emb


def tokenize_to_chars(
    text: str, max_tokens: int = 80, max_chars: int = MAX_CHARS_PER_TOKEN
) -> tuple[list[list[int]], list[str], list[int], list[int]]:
    """Tokenize text and convert to character IDs.

    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        max_chars: Maximum characters per token

    Returns:
        char_ids: [max_tokens, max_chars] character IDs
        tokens: List of token strings
        is_number: [max_tokens] 1 if token is a number, 0 otherwise
        token_type: [max_tokens] 0=word, 1=number, 2=space, 3=punct, 4=pad
    """
    import re

    # Regex pattern:
    # 1. Acronyms: single letters separated by periods (I.B.E.W.)
    # 2. Words: consecutive letters
    # 3. Numbers: digits
    # 4. Spaces: whitespace (normalized to single space)
    # 5. Period followed by non-space (keeps No.123, drops No. 123)
    # 6. Other punctuation (not period)
    pattern = (
        r"([A-Za-z](?:\.[A-Za-z])+\.?)|([a-zA-Z]+)|(\d+)|(\s+)|(\.(?=\S))|([^\s\w.])"
    )

    tokens = []
    is_number = []
    token_type = []  # 0=word, 1=number, 2=space, 3=punct

    for match in re.finditer(pattern, text.lower()):
        if match.group(1):  # acronym - strip periods
            tokens.append(match.group(1).replace(".", ""))
            is_number.append(0)
            token_type.append(0)
        elif match.group(2):  # word
            tokens.append(match.group(2))
            is_number.append(0)
            token_type.append(0)
        elif match.group(3):  # number
            num = match.group(3).lstrip("0") or "0"
            tokens.append(num)
            is_number.append(1)
            token_type.append(1)
        elif match.group(4):  # space - normalize to single space
            tokens.append(" ")
            is_number.append(0)
            token_type.append(2)
        elif match.group(5):  # period (followed by non-space)
            tokens.append(".")
            is_number.append(0)
            token_type.append(3)
        elif match.group(6):  # other punct
            tokens.append(match.group(6))
            is_number.append(0)
            token_type.append(3)

    # Truncate to max_tokens
    tokens = tokens[:max_tokens]
    is_number = is_number[:max_tokens]
    token_type = token_type[:max_tokens]

    # Convert to character IDs
    char_ids = []
    for token in tokens:
        char_ids.append(chars_to_ids(token, max_chars))

    # Pad to max_tokens
    while len(char_ids) < max_tokens:
        char_ids.append([0] * max_chars)
        tokens.append("")  # empty token for padding
        is_number.append(0)
        token_type.append(4)  # pad

    return char_ids, tokens, is_number, token_type


# Build a small vocab for non-word tokens (numbers, space, punct)
# Numbers: common local numbers get their own embedding
SPECIAL_TOKEN_VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    " ": 2,  # space
}
# Common punctuation
for p in "-/&,.()'\"#:":
    SPECIAL_TOKEN_VOCAB[p] = len(SPECIAL_TOKEN_VOCAB)
# Common numbers (locals 1-1000 and some common ones)
for i in range(1001):
    SPECIAL_TOKEN_VOCAB[str(i)] = len(SPECIAL_TOKEN_VOCAB)


def get_special_token_id(token: str) -> int:
    """Get ID for non-word token (number, space, punct)."""
    return SPECIAL_TOKEN_VOCAB.get(token, SPECIAL_TOKEN_VOCAB["<UNK>"])
