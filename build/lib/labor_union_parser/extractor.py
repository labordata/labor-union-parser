"""Main extraction interface for labor union parser."""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .model import CharCNNExtractor as CharCNNModel, extract_desig_from_pred
from .char_cnn import tokenize_to_chars, get_special_token_id
from .tokenizer import MAX_TOKEN_LEN

from tqdm import tqdm

# Lazy-loaded fnum lookup cache
_fnum_lookup: Optional[dict] = None


def _load_fnum_lookup() -> dict:
    """Load the fnum lookup table from package data."""
    global _fnum_lookup
    if _fnum_lookup is None:
        lookup_path = Path(__file__).parent / "weights" / "fnum_lookup.json"
        with open(lookup_path, "r") as f:
            _fnum_lookup = json.load(f)
    return _fnum_lookup


def lookup_fnum(affiliation: str, designation: str) -> list[int]:
    """
    Look up filing numbers (fnum) for an affiliation and designation.

    Args:
        affiliation: Affiliation abbreviation (e.g., "SEIU", "IBT")
        designation: Local designation number as string (e.g., "1199")

    Returns:
        List of filing numbers. Empty list if no match found.
    """
    lookup = _load_fnum_lookup()
    key = f"{affiliation}|{designation}"
    return lookup.get(key, [])


class Extractor:
    """
    Extract affiliation and designation from labor union names.

    Uses CharacterCNN for typo-robust word embeddings.

    Example:
        >>> extractor = Extractor()
        >>> extractor.extract("SEIU Local 1199")
        {'affiliation': 'SEIU', 'designation': '1199'}
    """

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            model_path: Path to model checkpoint. Uses bundled weights if None.
            device: Device to use (default: mps if available, else cpu)
        """
        if model_path is None:
            model_path = Path(__file__).parent / "weights" / "char_cnn.pt"

        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
        self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        self.aff_to_idx = checkpoint["aff_to_idx"]
        self.idx_to_aff = checkpoint["idx_to_aff"]

        self.model = CharCNNModel(
            num_affs=len(self.aff_to_idx),
            token_embed_dim=64,
            hidden_dim=512,
            aff_embed_dim=64,
            char_embed_dim=16,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _prepare_batch(self, texts: list[str]):
        """Prepare a batch of texts for inference."""
        char_ids_list, token_masks, is_number_list = [], [], []
        token_type_list, special_ids_list = [], []

        for text in texts:
            char_ids, tokens, is_number, token_type = tokenize_to_chars(
                text, max_tokens=MAX_TOKEN_LEN
            )
            seq_len = sum(1 for t in tokens if t)

            special_ids = [
                get_special_token_id(t) if tt != 0 else 0
                for t, tt in zip(tokens, token_type)
            ]

            char_ids_list.append(char_ids)
            token_masks.append([1.0] * seq_len + [0.0] * (MAX_TOKEN_LEN - seq_len))
            is_number_list.append(is_number)
            token_type_list.append(token_type)
            special_ids_list.append(special_ids)

        return {
            "char_ids": torch.tensor(
                char_ids_list, dtype=torch.long, device=self.device
            ),
            "token_mask": torch.tensor(
                token_masks, dtype=torch.float, device=self.device
            ),
            "is_number": torch.tensor(
                is_number_list, dtype=torch.long, device=self.device
            ),
            "token_type": torch.tensor(
                token_type_list, dtype=torch.long, device=self.device
            ),
            "special_ids": torch.tensor(
                special_ids_list, dtype=torch.long, device=self.device
            ),
        }

    def extract(self, text: str) -> dict:
        """
        Extract affiliation and designation from a union name.

        Args:
            text: Union name string (e.g., "SEIU Local 1199")

        Returns:
            Dictionary with 'affiliation', 'designation', and 'confidence' keys.
            Confidence is the softmax probability for the predicted affiliation (0-1).
            Example: {'affiliation': 'SEIU', 'designation': '1199', 'confidence': 0.998}
        """
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str]) -> list[dict]:
        """
        Extract affiliation and designation from multiple union names.

        Args:
            texts: List of union name strings

        Returns:
            List of dictionaries with 'affiliation', 'designation', and 'confidence' keys.
            Confidence is the softmax probability for the predicted affiliation (0-1).
        """
        if not texts:
            return []

        batch = self._prepare_batch(texts)

        with torch.no_grad():
            results = self.model(
                batch["char_ids"],
                batch["token_mask"],
                is_number=batch["is_number"],
                token_type=batch["token_type"],
                special_ids=batch["special_ids"],
            )

        aff_indices = results["aff_idx"].cpu().numpy()
        desig_preds = results["desig_pred"].cpu().numpy()

        # Compute confidence as max softmax probability
        probs = F.softmax(results["aff_logits"], dim=-1)
        confidences = probs.max(dim=-1).values.cpu().numpy()

        outputs = []
        for i, text in enumerate(texts):
            affiliation = self.idx_to_aff.get(int(aff_indices[i]), "")
            designation = extract_desig_from_pred(text, int(desig_preds[i]))
            outputs.append(
                {
                    "affiliation": affiliation,
                    "designation": designation,
                    "confidence": float(confidences[i]),
                }
            )

        return outputs

    def extract_all(self, texts, batch_size: int = 256, show_progress: bool = False):
        """
        Extract affiliation and designation from a large list of union names.

        Generator that yields results as they are processed, enabling
        memory-efficient processing of large datasets.

        Args:
            texts: List of union name strings
            batch_size: Number of texts to process at once (default: 256)
            show_progress: If True, show tqdm progress bar

        Yields:
            Dictionaries with 'affiliation', 'designation', and 'confidence' keys.

        Example:
            >>> extractor = Extractor()
            >>> for result in extractor.extract_all(texts, show_progress=True):
            ...     print(result)
        """
        import itertools

        pbar = tqdm(desc="Extracting") if show_progress else None

        for batch in itertools.batched(texts, batch_size):
            results = self.extract_batch(batch)
            if pbar is not None:
                pbar.update(len(batch))
            yield from results

        if pbar is not None:
            pbar.close()


def extract(text: str) -> dict:
    """
    Extract affiliation and designation from a union name.

    Convenience function that uses a cached default extractor instance.

    Args:
        text: Union name string (e.g., "SEIU Local 1199")

    Returns:
        Dictionary with 'affiliation', 'designation', and 'confidence' keys.
        Confidence is the softmax probability for the predicted affiliation (0-1).
        Example: {'affiliation': 'SEIU', 'designation': '1199', 'confidence': 0.998}

    Example:
        >>> from labor_union_parser import extract
        >>> extract("Teamsters Local 705")
        {'affiliation': 'IBT', 'designation': '705', 'confidence': 0.999}
    """
    return Extractor().extract(text)
