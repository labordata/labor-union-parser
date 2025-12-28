"""Main extraction interface for labor union parser."""

from pathlib import Path
from typing import Optional

import torch

from .model import BIOCRFBiLSTMExtractor, extract_desig_from_bio
from .tokenizer import tokenize, text_to_token_ids, MAX_TOKEN_LEN

from tqdm import tqdm


class Extractor:
    """
    Extract affiliation and designation from labor union names.

    Example:
        >>> extractor = Extractor()
        >>> extractor.extract("SEIU Local 1199")
        {'affiliation': 'SEIU', 'designation': '1199'}
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the extractor.

        Args:
            model_path: Path to model checkpoint. Uses bundled weights if None.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "weights" / "bilstm_bio_crf.pt"

        self.device = "cpu"  # CPU for inference, can be overridden
        self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        self.token_to_idx = checkpoint["token_to_idx"]
        self.aff_to_idx = checkpoint["aff_to_idx"]
        self.idx_to_aff = checkpoint["idx_to_aff"]

        self.model = BIOCRFBiLSTMExtractor(
            token_vocab_size=len(self.token_to_idx),
            num_affs=len(self.aff_to_idx),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def extract(self, text: str) -> dict:
        """
        Extract affiliation and designation from a union name.

        Args:
            text: Union name string (e.g., "SEIU Local 1199")

        Returns:
            Dictionary with 'affiliation' and 'designation' keys.
            Example: {'affiliation': 'SEIU', 'designation': '1199'}
        """
        # Prepare input
        tokens = tokenize(text)
        seq_len = min(len(tokens), MAX_TOKEN_LEN)

        token_ids = torch.tensor(
            [text_to_token_ids(text, self.token_to_idx, MAX_TOKEN_LEN)],
            dtype=torch.long,
            device=self.device,
        )
        token_mask = torch.tensor(
            [[1.0] * seq_len + [0.0] * (MAX_TOKEN_LEN - seq_len)],
            dtype=torch.float,
            device=self.device,
        )

        # Run inference
        with torch.no_grad():
            results = self.model(token_ids, token_mask)

        # Extract predictions
        aff_idx = int(results["aff_idx"][0].item())
        affiliation = self.idx_to_aff.get(aff_idx, "")

        bio_preds = results["bio_preds"][0].cpu().numpy()
        mask = token_mask[0].cpu().numpy()
        designation = extract_desig_from_bio(text, bio_preds, mask)

        return {
            "affiliation": affiliation,
            "designation": designation,
        }

    def extract_batch(self, texts: list[str]) -> list[dict]:
        """
        Extract affiliation and designation from multiple union names.

        Args:
            texts: List of union name strings

        Returns:
            List of dictionaries with 'affiliation' and 'designation' keys.
        """
        if not texts:
            return []

        # Prepare batch
        token_ids_list = []
        token_masks_list = []

        for text in texts:
            tokens = tokenize(text)
            seq_len = min(len(tokens), MAX_TOKEN_LEN)
            token_ids_list.append(
                text_to_token_ids(text, self.token_to_idx, MAX_TOKEN_LEN)
            )
            token_masks_list.append([1.0] * seq_len + [0.0] * (MAX_TOKEN_LEN - seq_len))

        token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=self.device)
        token_mask = torch.tensor(
            token_masks_list, dtype=torch.float, device=self.device
        )

        # Run inference
        with torch.no_grad():
            results = self.model(token_ids, token_mask)

        # Extract predictions
        aff_indices = results["aff_idx"].cpu().numpy()
        bio_preds = results["bio_preds"].cpu().numpy()
        masks = token_mask.cpu().numpy()

        outputs = []
        for i, text in enumerate(texts):
            affiliation = self.idx_to_aff.get(int(aff_indices[i]), "")
            designation = extract_desig_from_bio(text, bio_preds[i], masks[i])
            outputs.append(
                {
                    "affiliation": affiliation,
                    "designation": designation,
                }
            )

        return outputs

    def extract_all(
        self, texts: list[str], batch_size: int = 256, show_progress: bool = False
    ):
        """
        Extract affiliation and designation from a large list of union names.

        Generator that yields results as they are processed, enabling
        memory-efficient processing of large datasets.

        Args:
            texts: List of union name strings
            batch_size: Number of texts to process at once (default: 256)
            show_progress: If True, show tqdm progress bar

        Yields:
            Dictionaries with 'affiliation' and 'designation' keys.

        Example:
            >>> extractor = Extractor()
            >>> for result in extractor.extract_all(texts, show_progress=True):
            ...     print(result)
        """
        total = len(texts)
        pbar = tqdm(total=total, desc="Extracting") if show_progress else None

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            results = self.extract_batch(batch)
            if pbar:
                pbar.update(len(batch))
            yield from results

        if pbar:
            pbar.close()


def extract(text: str) -> dict:
    """
    Extract affiliation and designation from a union name.

    Convenience function that uses a cached default extractor instance.

    Args:
        text: Union name string (e.g., "SEIU Local 1199")

    Returns:
        Dictionary with 'affiliation' and 'designation' keys.
        Example: {'affiliation': 'SEIU', 'designation': '1199'}

    Example:
        >>> from labor_union_parser import extract
        >>> extract("Teamsters Local 705")
        {'affiliation': 'IBT', 'designation': '705'}
    """
    return Extractor().extract(text)
