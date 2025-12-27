"""BiLSTM-CRF model for union name extraction."""

import torch
import torch.nn as nn

from .tokenizer import tokenize, MAX_TOKEN_LEN

# BIO tags
O_TAG = 0
B_TAG = 1
I_TAG = 2
NUM_TAGS = 3


class CRFLayer(nn.Module):
    """Conditional Random Field layer for sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        self._init_transitions()

    def _init_transitions(self):
        with torch.no_grad():
            self.transitions[O_TAG, I_TAG] = -10.0
            self.transitions[B_TAG, B_TAG] = -2.0
            self.start_transitions[I_TAG] = -10.0

    def forward(self, emissions, tags, mask):
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood.mean()

    def _compute_log_likelihood(self, emissions, tags, mask):
        gold_score = self._score_sequence(emissions, tags, mask)
        forward_score = self._compute_partition(emissions, mask)
        return gold_score - forward_score

    def _score_sequence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for i in range(1, seq_len):
            valid = mask[:, i]
            trans_score = self.transitions[tags[:, i - 1], tags[:, i]]
            emit_score = emissions[:, i].gather(1, tags[:, i : i + 1]).squeeze(1)
            score = score + (trans_score + emit_score) * valid

        seq_lens = mask.sum(dim=1).long()
        last_tags = tags.gather(1, (seq_lens - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_partition(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_len):
            valid = mask[:, i].unsqueeze(1)
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = (
                broadcast_score + self.transitions.unsqueeze(0) + broadcast_emissions
            )
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(valid.bool(), next_score, score)

        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions, mask):
        """Viterbi decoding to find best tag sequence."""
        batch_size, seq_len, num_tags = emissions.shape
        score = self.start_transitions + emissions[:, 0]
        backpointers = []

        for i in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[:, i].unsqueeze(1)
            next_score = (
                broadcast_score + self.transitions.unsqueeze(0) + broadcast_emissions
            )
            best_prev_score, best_prev_tag = next_score.max(dim=1)
            backpointers.append(best_prev_tag)
            valid = mask[:, i].unsqueeze(1).bool()
            score = torch.where(valid, best_prev_score, score)

        score = score + self.end_transitions
        _, best_last_tag = score.max(dim=1)

        best_tags = [best_last_tag]
        for bp in reversed(backpointers):
            best_last_tag = bp.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_tags.append(best_last_tag)

        best_tags.reverse()
        return torch.stack(best_tags, dim=1)


class BIOCRFBiLSTMExtractor(nn.Module):
    """
    BiLSTM extractor with:
    - Token conv + max pool for affiliation classification
    - Token BiLSTM + CRF for BIO tagging (designation extraction)
    - Token embeddings shared between conv and BiLSTM
    """

    def __init__(
        self,
        token_vocab_size: int,
        num_affs: int,
        token_embed_dim: int = 64,
        hidden_dim: int = 512,
        aff_embed_dim: int = 64,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(
            token_vocab_size, token_embed_dim, padding_idx=0
        )
        self.aff_embed = nn.Embedding(num_affs, aff_embed_dim)

        # Token conv for affiliation classification
        self.token_conv = nn.Sequential(
            nn.Conv1d(token_embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.aff_classifier = nn.Linear(hidden_dim, num_affs)

        # BiLSTM for BIO tagging
        self.lstm = nn.LSTM(
            token_embed_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Emission layer (feeds into CRF)
        self.emission = nn.Linear(hidden_dim * 2 + aff_embed_dim, NUM_TAGS)

        # CRF layer
        self.crf = CRFLayer(NUM_TAGS)

        self.num_affs = num_affs

    def forward(self, token_ids, token_mask, aff_labels=None, bio_labels=None):
        batch_size, seq_len = token_ids.shape

        # Token embeddings (shared between conv and BiLSTM)
        token_emb = self.token_embed(token_ids)

        # Affiliation classification via token conv + max pool
        conv_in = token_emb.transpose(1, 2)
        conv_out = self.token_conv(conv_in)
        aff_features = conv_out.max(dim=2)[0]
        aff_logits = self.aff_classifier(aff_features)
        aff_idx = aff_logits.argmax(dim=1)

        # Get affiliation embedding for BIO gating
        if aff_labels is not None:
            aff_emb = self.aff_embed(aff_labels)
        else:
            aff_emb = self.aff_embed(aff_idx)

        aff_emb_broadcast = aff_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # BiLSTM for BIO
        lstm_out, _ = self.lstm(token_emb)
        lstm_with_aff = torch.cat([lstm_out, aff_emb_broadcast], dim=-1)

        # Emission scores for CRF
        emissions = self.emission(lstm_with_aff)

        # Decode best sequence
        bio_preds = self.crf.decode(emissions, token_mask)

        results = {
            "aff_logits": aff_logits,
            "aff_idx": aff_idx,
            "emissions": emissions,
            "bio_preds": bio_preds,
        }

        if bio_labels is not None and aff_labels is not None:
            crf_loss = self.crf(emissions, bio_labels, token_mask)
            aff_loss = nn.functional.cross_entropy(aff_logits, aff_labels)
            results["crf_loss"] = crf_loss
            results["aff_loss"] = aff_loss
            results["total_loss"] = aff_loss + crf_loss

        return results


def create_bio_labels(
    text: str, desig_num: str, max_len: int = MAX_TOKEN_LEN
) -> list[int]:
    """Create BIO labels at token level for training."""
    tokens = tokenize(text)
    labels = [O_TAG] * min(len(tokens), max_len)

    if not desig_num or desig_num == "N/A":
        return (labels + [O_TAG] * (max_len - len(labels)))[:max_len]

    desig_str = str(desig_num).lstrip("0") or "0"
    desig_digits = list(desig_str)

    # Find the last occurrence of digit sequence
    best_start = None
    for i in range(len(tokens)):
        if i + len(desig_digits) <= len(tokens):
            match = True
            for j, digit in enumerate(desig_digits):
                if tokens[i + j] != digit:
                    match = False
                    break
            if match:
                best_start = i

    if best_start is not None and best_start < max_len:
        labels[best_start] = B_TAG
        for i in range(best_start + 1, min(best_start + len(desig_digits), max_len)):
            labels[i] = I_TAG

    return (labels + [O_TAG] * (max_len - len(labels)))[:max_len]


def extract_desig_from_bio(text: str, bio_preds, mask) -> str:
    """Extract designation number from BIO predictions at token level."""
    tokens = tokenize(text)
    digits = []
    in_span = False

    for i, (pred, m) in enumerate(zip(bio_preds, mask)):
        if i >= len(tokens):
            break
        if m == 0:
            break

        pred = int(pred)
        token = tokens[i]

        if pred == B_TAG:
            digits = [token] if token.isdigit() else []
            in_span = True
        elif pred == I_TAG and in_span:
            if token.isdigit():
                digits.append(token)
        else:
            if in_span:
                break

    result = "".join(digits)
    if result:
        result = result.lstrip("0") or "0"
    return result
