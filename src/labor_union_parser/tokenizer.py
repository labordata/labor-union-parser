"""Tokenization utilities for labor union parser."""

import re


MAX_TOKEN_LEN = 80


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into words, individual digits, and spaces.

    Examples:
        >>> tokenize("SEIU Local 1199")
        ['seiu', ' ', 'local', ' ', '1', '1', '9', '9']
    """
    return re.findall(r"[a-zA-Z]+|\d|\s+|[^\s\w]", text.lower())


def build_token_vocab(examples: list[dict]) -> dict[str, int]:
    """Build token vocabulary from training examples."""
    tokens = {t for ex in examples for t in tokenize(ex["text"])}
    token_list = ["<PAD>", "<UNK>"] + sorted(tokens)
    return {t: i for i, t in enumerate(token_list)}


def build_aff_vocab(examples: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    """Build affiliation vocabulary from training examples."""
    affs = {ex["aff_abbr"] for ex in examples if ex["aff_abbr"]}
    aff_list = ["<UNK>"] + sorted(affs)
    aff_to_idx = {a: i for i, a in enumerate(aff_list)}
    idx_to_aff = {i: a for a, i in aff_to_idx.items()}
    return aff_to_idx, idx_to_aff


def text_to_token_ids(
    text: str, token_to_idx: dict[str, int], max_len: int = MAX_TOKEN_LEN
) -> list[int]:
    """Convert text to token IDs."""
    tokens = tokenize(text)
    ids = [token_to_idx.get(t, token_to_idx["<UNK>"]) for t in tokens[:max_len]]
    return ids + [token_to_idx["<PAD>"]] * (max_len - len(ids))
