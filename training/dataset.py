"""
Dataset classes, collate functions, and data encoding helpers for dual-task training.
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labor_union_parser.char_cnn import NUMBER_VOCAB, tokenize_to_chars
from labor_union_parser.model import MAX_QUERY_LEN

CATEGORY_TO_IDX = {"unique": 0, "ambiguous": 1}
NUM_CATEGORIES = len(CATEGORY_TO_IDX)


def smart_truncate_nonspace(text, max_nonspace_tokens):
    """
    Tokenize text, keep first N non-space tokens, and recover lost numbers.

    This combines two strategies:
    1. Remove space tokens to reduce sequence length
    2. Replace trailing non-numeric tokens with numbers that were truncated

    Args:
        text: Input text to tokenize
        max_nonspace_tokens: Maximum number of non-space tokens to keep

    Returns:
        Same format as tokenize_to_chars: (chars, tokens, is_num, token_types, num_ids)
    """
    # Tokenize full text
    full_chars, full_tokens, full_is_num, full_token_types, full_num_ids = (
        tokenize_to_chars(text, 999)
    )

    # Extract all non-space tokens
    nonspace_data = []
    for i, tt in enumerate(full_token_types):
        if full_tokens[i] and tt != 2:  # token_type 2 = space
            nonspace_data.append(
                {
                    "chars": full_chars[i],
                    "token": full_tokens[i],
                    "is_num": full_is_num[i],
                    "token_type": tt,
                    "num_id": full_num_ids[i],
                }
            )

    # Take first N non-space tokens
    trunc_data = nonspace_data[:max_nonspace_tokens]

    # Find lost numbers and recover them
    trunc_numbers = {d["token"] for d in trunc_data if d["is_num"]}
    all_numbers = [d for d in nonspace_data if d["is_num"]]
    lost_numbers = [d for d in all_numbers if d["token"] not in trunc_numbers]

    if lost_numbers:
        # Replace trailing non-numeric tokens with lost numbers
        replace_positions = []
        for i in range(len(trunc_data) - 1, -1, -1):
            if not trunc_data[i]["is_num"] and trunc_data[i]["token"]:
                replace_positions.append(i)
                if len(replace_positions) >= len(lost_numbers):
                    break

        replace_positions.reverse()

        for pos, lost_num_data in zip(replace_positions, lost_numbers):
            trunc_data[pos] = lost_num_data

    # Pad to max_nonspace_tokens
    while len(trunc_data) < max_nonspace_tokens:
        trunc_data.append(
            {
                "chars": [0] * 20,
                "token": "",
                "is_num": 0,
                "token_type": 4,  # padding
                "num_id": 0,
            }
        )

    # Build output arrays (same format as tokenize_to_chars)
    trunc_chars = [d["chars"] for d in trunc_data]
    trunc_tokens = [d["token"] for d in trunc_data]
    trunc_is_num = [d["is_num"] for d in trunc_data]
    trunc_token_types = [d["token_type"] for d in trunc_data]
    trunc_num_ids = [d["num_id"] for d in trunc_data]

    return trunc_chars, trunc_tokens, trunc_is_num, trunc_token_types, trunc_num_ids


def normalize_designation(s: str) -> str:
    if not s:
        return s
    if s.isdigit():
        return s.lstrip("0") or "0"
    return s


# =============================================================================
# Batch Encoding Helpers (for mining efficiency)
# =============================================================================


def encode_query_batch(queries, max_len=MAX_QUERY_LEN):
    """Encode a batch of query strings into tensors."""
    char_ids_list = []
    is_number_list = []
    numeric_ids_list = []

    for query in queries:
        char_ids, _, is_number, _, numeric_ids = smart_truncate_nonspace(query, max_len)
        char_ids_list.append(char_ids)
        is_number_list.append(is_number)
        numeric_ids_list.append(numeric_ids)

    return {
        "char_ids": torch.tensor(char_ids_list, dtype=torch.long),
        "is_number": torch.tensor(is_number_list, dtype=torch.long),
        "numeric_ids": torch.tensor(numeric_ids_list, dtype=torch.long),
    }


def encode_record_batch(records, vocab):
    """Encode a batch of record dicts into tensors."""
    u_map = vocab["union_name_to_idx"]
    d_map = vocab["desig_name_to_idx"]
    s_map = vocab["suffix_to_idx"]
    uid_map = vocab["unit_id_to_idx"]

    union_idx = []
    desig_idx = []
    prefix_hash = []
    num_hash = []
    suffix_idx = []
    unit_id_idx = []

    for rec in records:
        union_idx.append(u_map.get(rec["union_name"], 0))
        desig_idx.append(d_map.get(rec.get("desig_name", ""), 0))
        prefix_val = rec.get("prefix", 0) or 0
        prefix_hash.append(
            NUMBER_VOCAB.get(str(prefix_val), NUMBER_VOCAB["<UNK>"])
            if prefix_val
            else 0
        )
        num_hash.append(
            NUMBER_VOCAB.get(str(rec.get("desig_num", 0)), NUMBER_VOCAB["<UNK>"])
        )
        suffix_norm = normalize_designation(rec.get("suffix", "") or "")
        suffix_idx.append(s_map.get(suffix_norm, 0))
        unit_id_idx.append(uid_map.get(rec.get("unit_id", ""), 0))

    return {
        "union_idx": torch.tensor(union_idx, dtype=torch.long),
        "desig_idx": torch.tensor(desig_idx, dtype=torch.long),
        "prefix_hash": torch.tensor(prefix_hash, dtype=torch.long),
        "num_hash": torch.tensor(num_hash, dtype=torch.long),
        "suffix_idx": torch.tensor(suffix_idx, dtype=torch.long),
        "unit_id_idx": torch.tensor(unit_id_idx, dtype=torch.long),
    }


# =============================================================================
# Dataset
# =============================================================================


class QueryDataset(Dataset):
    """
    Query-only dataset for realistic validation against the full record corpus.

    Each example provides only query tokens and the target f_num.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def collate(batch):
        queries = [ex["query"] for ex in batch]
        f_nums = torch.tensor([ex["f_num"] for ex in batch])
        encoded = encode_query_batch(queries)
        encoded["f_num"] = f_nums
        encoded["category"] = torch.tensor(
            [CATEGORY_TO_IDX.get(ex.get("category", "unique"), 0) for ex in batch],
            dtype=torch.long,
        )
        return encoded


class QueryRecordDataset(Dataset):
    """
    Dataset for dual-task training with in-batch negatives.

    Each example provides:
    - query tokens
    - positive record fields (including unit_id)

    Both tasks use in-batch negatives, so no explicit negatives needed.
    """

    def __init__(
        self,
        examples,
        union_name_to_idx,
        desig_name_to_idx,
        suffix_to_idx,
        unit_id_to_idx,
    ):
        self.examples = examples
        self.u_map = union_name_to_idx
        self.d_map = desig_name_to_idx
        self.s_map = suffix_to_idx
        self.uid_map = unit_id_to_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"]
        records = ex["records"]

        # Sample a positive record variant
        pos_record = random.choice(records)

        # Tokenize query (smart: no spaces, recover lost numbers)
        char_ids, _, is_number, token_type, numeric_ids = smart_truncate_nonspace(
            query, MAX_QUERY_LEN
        )

        # Encode record fields
        union_idx = self.u_map.get(pos_record["union_name"], 0)
        desig_idx = self.d_map.get(pos_record["desig_name"], 0)
        prefix_val = pos_record.get("prefix", 0) or 0
        prefix_hash = (
            NUMBER_VOCAB.get(str(prefix_val), NUMBER_VOCAB["<UNK>"])
            if prefix_val
            else 0
        )
        suffix_norm = (
            normalize_designation(pos_record["suffix"]) if pos_record["suffix"] else ""
        )
        suffix_idx = self.s_map.get(suffix_norm, 0)
        num_hash = NUMBER_VOCAB.get(str(pos_record["desig_num"]), NUMBER_VOCAB["<UNK>"])
        unit_id_idx = self.uid_map.get(pos_record.get("unit_id", ""), 0)

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "numeric_ids": torch.tensor(numeric_ids, dtype=torch.long),
            "union_idx": torch.tensor(union_idx, dtype=torch.long),
            "desig_idx": torch.tensor(desig_idx, dtype=torch.long),
            "prefix_hash": torch.tensor(prefix_hash, dtype=torch.long),
            "num_hash": torch.tensor(num_hash, dtype=torch.long),
            "suffix_idx": torch.tensor(suffix_idx, dtype=torch.long),
            "unit_id_idx": torch.tensor(unit_id_idx, dtype=torch.long),
            "f_num": torch.tensor(ex["f_num"], dtype=torch.long),
            "category": torch.tensor(
                CATEGORY_TO_IDX.get(ex.get("category", "unique"), 0), dtype=torch.long
            ),
        }

    @staticmethod
    def collate(batch):
        """Collate batch items into tensors."""
        return {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}


class QueryRecordDatasetWithCandidates(Dataset):
    """
    Training dataset that returns K+1 candidates per query (for ANCE training).

    Each example provides:
    - query tokens
    - K+1 candidate records (1 positive + K mined candidates)
    - f_num for each candidate (to identify positives in the loss)
    """

    def __init__(self, examples, vocab):
        self.examples = examples
        self.u_map = vocab["union_name_to_idx"]
        self.d_map = vocab["desig_name_to_idx"]
        self.s_map = vocab["suffix_to_idx"]
        self.uid_map = vocab["unit_id_to_idx"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        query = ex["query"]
        query_fnum = ex["f_num"]

        # Positive record (always first in candidate list)
        pos_record = random.choice(ex["records"])

        # Build candidate list with f_nums: [(fnum, record), ...]
        candidates = [(query_fnum, pos_record)]  # First is always a positive

        if "mined_candidates" in ex and ex["mined_candidates"]:
            for cand_data in ex["mined_candidates"]:
                candidates.append((cand_data["f_num"], cand_data["record"]))

        if "structural_negatives" in ex and ex["structural_negatives"]:
            for cand_data in ex["structural_negatives"]:
                candidates.append((cand_data["f_num"], cand_data["record"]))

        # Encode query
        char_ids, _, is_number, _, numeric_ids = smart_truncate_nonspace(
            query, MAX_QUERY_LEN
        )

        # Encode all candidates [K+1, ...]
        cand_fnums = []  # Track f_num for supervised contrastive loss
        cand_union_idx = []
        cand_desig_idx = []
        cand_prefix_hash = []
        cand_num_hash = []
        cand_suffix_idx = []
        cand_unit_id_idx = []

        for fnum, rec in candidates:
            cand_fnums.append(fnum)
            cand_union_idx.append(self.u_map.get(rec["union_name"], 0))
            cand_desig_idx.append(self.d_map.get(rec.get("desig_name", ""), 0))
            prefix_val = rec.get("prefix", 0) or 0
            cand_prefix_hash.append(
                NUMBER_VOCAB.get(str(prefix_val), NUMBER_VOCAB["<UNK>"])
                if prefix_val
                else 0
            )
            cand_num_hash.append(
                NUMBER_VOCAB.get(str(rec.get("desig_num", 0)), NUMBER_VOCAB["<UNK>"])
            )
            suffix_norm = normalize_designation(rec.get("suffix", "") or "")
            cand_suffix_idx.append(self.s_map.get(suffix_norm, 0))
            cand_unit_id_idx.append(self.uid_map.get(rec.get("unit_id", ""), 0))

        return {
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "is_number": torch.tensor(is_number, dtype=torch.long),
            "numeric_ids": torch.tensor(numeric_ids, dtype=torch.long),
            "f_num": torch.tensor(query_fnum, dtype=torch.long),  # Query's f_num
            # Candidate records [K+1, ...]
            "cand_fnums": torch.tensor(cand_fnums, dtype=torch.long),  # For loss
            "cand_union_idx": torch.tensor(cand_union_idx, dtype=torch.long),
            "cand_desig_idx": torch.tensor(cand_desig_idx, dtype=torch.long),
            "cand_prefix_hash": torch.tensor(cand_prefix_hash, dtype=torch.long),
            "cand_num_hash": torch.tensor(cand_num_hash, dtype=torch.long),
            "cand_suffix_idx": torch.tensor(cand_suffix_idx, dtype=torch.long),
            "cand_unit_id_idx": torch.tensor(cand_unit_id_idx, dtype=torch.long),
            "num_candidates": torch.tensor(len(candidates), dtype=torch.long),
            "category": torch.tensor(
                CATEGORY_TO_IDX.get(ex.get("category", "unique"), 0), dtype=torch.long
            ),
        }

    @staticmethod
    def collate(batch):
        """
        Collate batch items with variable-length candidates.

        Pads candidate tensors to max number of candidates in the batch.
        Uses -1 for cand_fnums padding (will never match a real f_num).
        """
        max_cands = max(item["num_candidates"].item() for item in batch)

        cand_keys = [
            "cand_fnums",
            "cand_union_idx",
            "cand_desig_idx",
            "cand_prefix_hash",
            "cand_num_hash",
            "cand_suffix_idx",
            "cand_unit_id_idx",
        ]

        padded_batch = []
        for item in batch:
            num_cands = item["num_candidates"].item()
            new_item = {}

            for key, val in item.items():
                if key in cand_keys:
                    if num_cands < max_cands:
                        pad_size = max_cands - num_cands
                        pad_value = -1 if key == "cand_fnums" else 0
                        new_item[key] = F.pad(val, (0, pad_size), value=pad_value)
                    else:
                        new_item[key] = val
                else:
                    new_item[key] = val

            padded_batch.append(new_item)

        return {
            k: torch.stack([item[k] for item in padded_batch])
            for k in padded_batch[0].keys()
        }
