"""
Spike: Structured-field-conditional masked discrete diffusion language model (MDLM)
for generating synthetic union name surface forms.

Operates over token sequences from the existing tokenizer.
Conditioned on structured fields: union_name, desig_name, desig_num, prefix, suffix.

Stage 1: Train MDLM on labeled (token_sequence, fields) pairs
Stage 2: Generate synthetic samples for rare f_nums, inspect quality
"""

import argparse
import json
import math
import random
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from labor_union_parser.tokenizer import smart_truncate_nonspace

SEQ_LEN = 20  # MAX_TOKENS

# Special token IDs
PAD_ID = 0
MASK_ID = 1
DNUM_ID = 2  # placeholder for desig_num
PFX_ID = 3  # placeholder for prefix
SFX_ID = 4  # placeholder for suffix
VOCAB_OFFSET = 5  # real tokens start at index 5

# Structured fields we condition on
COND_FIELDS = ["union_name", "desig_name", "prefix", "suffix"]


# ---------------------------------------------------------------------------
# Token vocabulary
# ---------------------------------------------------------------------------
def build_token_vocab(examples):
    """Build token string -> ID mapping from training examples."""
    counts = Counter()
    for ex in examples:
        tokens = smart_truncate_nonspace(ex["query"])
        for t in tokens:
            if t["token_type"] != 4:
                counts[t["token"]] += 1
    vocab = {}
    for i, (tok, _) in enumerate(counts.most_common()):
        vocab[tok] = i + VOCAB_OFFSET
    return vocab


def tokenize_to_ids(text, token_vocab, record=None):
    """Tokenize text and convert to padded ID sequence.

    If record is provided and has desig_num/prefix/suffix, replace matching
    number tokens with placeholder IDs (DNUM_ID, PFX_ID, SFX_ID).
    Replacement strategy: scan left-to-right, replace first match of prefix,
    then first match of desig_num (after prefix), then first match of suffix
    (after desig_num).
    """
    tokens = smart_truncate_nonspace(text)
    ids = []
    for t in tokens:
        if t["token_type"] == 4:
            ids.append(PAD_ID)
        elif t["token"] in token_vocab:
            ids.append(token_vocab[t["token"]])
        else:
            ids.append(MASK_ID)

    if record is not None:
        dnum = record.get("desig_num", -100)
        prefix = record.get("prefix", -100)
        suffix = record.get("suffix", -100)

        dnum_str = str(int(dnum)) if dnum not in (-100, 0) else None
        pfx_str = str(int(prefix)) if prefix not in (-100, 0) else None
        # Suffix can be string (letter codes like "S", "C") or int
        if suffix in (-100, "", 0):
            sfx_str = None
        elif isinstance(suffix, int):
            sfx_str = str(suffix)
        else:
            sfx_str = str(suffix).lower()  # tokens are lowercased

        # Build list of (position, token_string, token_type) for non-pad tokens
        tok_info = []
        for i, t in enumerate(tokens):
            if t["token_type"] != 4:
                tok_info.append((i, t["token"], t["token_type"]))

        # Replace prefix first (earliest number match)
        pfx_pos = None
        if pfx_str:
            for pos, tok, ttype in tok_info:
                if ttype == 1 and tok == pfx_str:
                    ids[pos] = PFX_ID
                    pfx_pos = pos
                    break

        # Replace desig_num (first number match after prefix position, or anywhere)
        dnum_pos = None
        if dnum_str:
            start_after = pfx_pos if pfx_pos is not None else -1
            for pos, tok, ttype in tok_info:
                if pos <= start_after:
                    continue
                if ttype == 1 and tok == dnum_str:
                    ids[pos] = DNUM_ID
                    dnum_pos = pos
                    break

        # Replace suffix (first match after desig_num position, any token type)
        if sfx_str:
            start_after = dnum_pos if dnum_pos is not None else -1
            for pos, tok, ttype in tok_info:
                if pos <= start_after:
                    continue
                if tok == sfx_str:
                    ids[pos] = SFX_ID
                    break

    return ids


def ids_to_text(ids, inv_vocab, record=None):
    """Convert token IDs back to text string.

    If record is provided, replace placeholder IDs with actual values.
    """
    tokens = []
    dnum_str = ""
    pfx_str = ""
    sfx_str = ""
    if record is not None:
        dnum = record.get("desig_num", -100)
        prefix = record.get("prefix", -100)
        suffix = record.get("suffix", -100)
        dnum_str = str(int(dnum)) if dnum not in (-100, 0) else ""
        pfx_str = str(int(prefix)) if prefix not in (-100, 0) else ""
        if suffix in (-100, "", 0):
            sfx_str = ""
        elif isinstance(suffix, int):
            sfx_str = str(suffix)
        else:
            sfx_str = str(suffix).lower()

    for tid in ids:
        if tid == PAD_ID:
            break
        elif tid == MASK_ID:
            tokens.append("[MASK]")
        elif tid == DNUM_ID:
            tokens.append(dnum_str if dnum_str else "<DNUM>")
        elif tid == PFX_ID:
            tokens.append(pfx_str if pfx_str else "<PFX>")
        elif tid == SFX_ID:
            tokens.append(sfx_str if sfx_str else "<SFX>")
        elif tid in inv_vocab:
            tokens.append(inv_vocab[tid])
        else:
            tokens.append(f"[UNK:{tid}]")
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Field vocabularies
# ---------------------------------------------------------------------------
def build_field_vocabs(examples):
    """Build value->idx mappings for each conditioning field."""
    vocabs = {}
    for field in COND_FIELDS:
        values = set()
        for ex in examples:
            if not ex["records"]:
                continue
            val = ex["records"][0][field]
            if val != -100:
                values.add(val)
        # Sort for determinism; 0 index reserved for unknown/missing
        sorted_vals = sorted(values, key=str)
        vocabs[field] = {v: i + 1 for i, v in enumerate(sorted_vals)}
    return vocabs


def encode_desig_num(desig_num):
    """Encode desig_num as a list of digit tokens (max 8 digits).
    Returns list of ints 0-10 (0=pad, 1-10=digits 0-9)."""
    if desig_num in (-100, 0):
        return [0] * 8
    s = str(int(desig_num))[:8]
    encoded = [int(c) + 1 for c in s]
    encoded = encoded + [0] * (8 - len(encoded))
    return encoded


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MDLMDataset(Dataset):
    def __init__(
        self, examples, token_vocab, field_vocabs, fnum_vocab, use_templates=False
    ):
        self.token_ids = []
        self.lengths = []
        self.field_indices = []  # dict of field -> idx per example
        self.desig_nums = []
        self.fnum_indices = []

        for ex in examples:
            if not ex["records"]:
                continue
            rec = ex["records"][0]
            # Skip if union_name is not a string
            uname = rec["union_name"]
            if not isinstance(uname, str):
                continue

            ids = tokenize_to_ids(
                ex["query"], token_vocab, record=rec if use_templates else None
            )
            length = sum(1 for x in ids if x != PAD_ID)

            field_idx = {}
            for field in COND_FIELDS:
                val = rec[field]
                if val == -100:
                    field_idx[field] = 0  # unknown
                else:
                    field_idx[field] = field_vocabs[field].get(val, 0)

            fnum = rec["f_num"]
            fnum_idx = fnum_vocab.get(fnum, 0) if isinstance(fnum, int) else 0

            self.token_ids.append(ids)
            self.lengths.append(length)
            self.field_indices.append(field_idx)
            self.desig_nums.append(encode_desig_num(rec["desig_num"]))
            self.fnum_indices.append(fnum_idx)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.token_ids[idx], dtype=torch.long),
            self.lengths[idx],
            {f: self.field_indices[idx][f] for f in COND_FIELDS},
            torch.tensor(self.desig_nums[idx], dtype=torch.long),
            self.fnum_indices[idx],
        )


def collate_mdlm(batch):
    token_ids, lengths, field_dicts, desig_nums, fnum_indices = zip(*batch)
    fields = {}
    for f in COND_FIELDS:
        fields[f] = torch.tensor([d[f] for d in field_dicts], dtype=torch.long)
    return (
        torch.stack(token_ids),
        torch.tensor(lengths, dtype=torch.long),
        fields,
        torch.stack(desig_nums),
        torch.tensor(fnum_indices, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# MDLM Transformer with structured field conditioning
# ---------------------------------------------------------------------------
class MDLMTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        field_vocab_sizes,  # dict of field -> vocab_size (including 0=unknown)
        n_fnums,  # number of unique f_nums (0=unknown)
        d_model=256,
        n_heads=4,
        n_layers=3,
        ff_dim=512,
        dropout=0.1,
        compositional_fnum=False,
        fnum_field_map=None,  # (n_fnums+1, len(COND_FIELDS)) field indices per fnum
        fnum_desig_nums=None,  # (n_fnums+1, 8) digit-encoded desig_num per fnum
    ):
        super().__init__()
        self.d_model = d_model
        self.compositional_fnum = compositional_fnum

        # Token embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        # Position embedding
        self.pos_embed = nn.Embedding(SEQ_LEN, d_model)

        # Field conditioning embeddings
        self.field_embeds = nn.ModuleDict(
            {
                f: nn.Embedding(size + 1, d_model)  # +1 for unknown/0
                for f, size in field_vocab_sizes.items()
            }
        )

        # Desig_num embedding: small MLP over digit sequence
        self.digit_embed = nn.Embedding(11, 16)  # 0=pad, 1-10=digits
        self.digit_proj = nn.Sequential(
            nn.Linear(16 * 8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        if compositional_fnum:
            # Compositional: fnum_emb = sum(field_embeds) + desig_num_emb + residual
            # Store the mapping from fnum_idx -> field indices as a buffer
            self.register_buffer("fnum_field_map", fnum_field_map)
            self.register_buffer("fnum_desig_nums", fnum_desig_nums)
            # Small residual per f_num, initialized near zero
            self.fnum_residual = nn.Embedding(n_fnums + 1, d_model)
            nn.init.normal_(self.fnum_residual.weight, std=0.01)
        else:
            # Independent f_num embedding (original approach)
            self.fnum_embed = nn.Embedding(n_fnums + 1, d_model)

        # Noise level embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _compose_fnum_emb(self, fnum_indices):
        """Build f_num embedding compositionally from field embeddings + residual."""
        # Look up what fields each f_num maps to
        field_map = self.fnum_field_map[fnum_indices]  # (B, n_fields)
        desig_nums = self.fnum_desig_nums[fnum_indices]  # (B, 8)

        emb = torch.zeros(
            fnum_indices.size(0), self.d_model, device=fnum_indices.device
        )

        # Add each field's embedding
        for i, field in enumerate(COND_FIELDS):
            field_idx = field_map[:, i]  # (B,)
            emb = emb + self.field_embeds[field](field_idx)

        # Add desig_num embedding
        digit_emb = self.digit_embed(desig_nums)  # (B, 8, 16)
        digit_flat = digit_emb.view(digit_emb.size(0), -1)  # (B, 128)
        emb = emb + self.digit_proj(digit_flat)

        # Add small residual
        emb = emb + self.fnum_residual(fnum_indices)

        return emb

    def forward(
        self, token_ids, field_indices, desig_nums, fnum_indices, t, pad_mask=None
    ):
        """
        Args:
            token_ids: (B, SEQ_LEN)
            field_indices: dict of field -> (B,) long tensors
            desig_nums: (B, 8) digit-encoded designation numbers
            fnum_indices: (B,) f_num vocab indices (0=unknown)
            t: (B,) noise level in [0, 1]
            pad_mask: (B, SEQ_LEN) True for PAD positions
        Returns:
            logits: (B, SEQ_LEN, vocab_size)
        """
        x = self.tok_embed(token_ids)
        positions = torch.arange(SEQ_LEN, device=token_ids.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # Add conditioning from each field
        for field, embed in self.field_embeds.items():
            field_emb = embed(field_indices[field])  # (B, d_model)
            x = x + field_emb.unsqueeze(1)

        # Add f_num conditioning
        if self.compositional_fnum:
            fnum_emb = self._compose_fnum_emb(fnum_indices)
        else:
            fnum_emb = self.fnum_embed(fnum_indices)
        x = x + fnum_emb.unsqueeze(1)

        # Add desig_num conditioning
        digit_emb = self.digit_embed(desig_nums)  # (B, 8, 16)
        digit_flat = digit_emb.view(digit_emb.size(0), -1)  # (B, 128)
        num_emb = self.digit_proj(digit_flat)  # (B, d_model)
        x = x + num_emb.unsqueeze(1)

        # Add time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1))
        x = x + t_emb.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        x = self.ln_f(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# MDLM noise and loss
# ---------------------------------------------------------------------------
def mask_tokens(token_ids, t, lengths):
    B, L = token_ids.shape
    rand = torch.rand(B, L, device=token_ids.device)
    t_expanded = t.unsqueeze(1).expand(B, L)
    should_mask = rand < t_expanded
    is_pad = token_ids == PAD_ID
    should_mask = should_mask & ~is_pad
    masked_ids = token_ids.clone()
    masked_ids[should_mask] = MASK_ID
    return masked_ids, should_mask


def mdlm_loss(logits, original_ids, mask_flags):
    if not mask_flags.any():
        return torch.tensor(0.0, device=logits.device)
    logits_flat = logits[mask_flags]
    targets_flat = original_ids[mask_flags]
    return F.cross_entropy(logits_flat, targets_flat)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(
    model,
    field_indices,
    desig_nums,
    fnum_indices,
    lengths,
    n_steps=50,
    temperature=1.0,
    device="cpu",
):
    B = lengths.size(0)
    seq = torch.full((B, SEQ_LEN), PAD_ID, dtype=torch.long, device=device)
    for i in range(B):
        seq[i, : lengths[i]] = MASK_ID

    pad_mask = seq == PAD_ID

    for step in range(n_steps):
        t_val = 1.0 - (step + 1) / n_steps
        t = torch.full((B,), t_val, device=device)

        logits = model(
            seq, field_indices, desig_nums, fnum_indices, t, pad_mask=pad_mask
        )

        is_masked = seq == MASK_ID
        if not is_masked.any():
            break

        for i in range(B):
            masked_positions = is_masked[i].nonzero(as_tuple=True)[0]
            if len(masked_positions) == 0:
                continue

            n_remaining = len(masked_positions)
            n_unmask = max(1, int(math.ceil(n_remaining / max(n_steps - step, 1))))

            pos_logits = logits[i, masked_positions]

            if temperature > 0:
                scaled = pos_logits / temperature
                probs = F.softmax(scaled, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                confidence = probs.gather(1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
            else:
                confidence, sampled_tokens = pos_logits.max(dim=-1)

            _, top_indices = confidence.topk(min(n_unmask, n_remaining))
            positions_to_unmask = masked_positions[top_indices]
            seq[i, positions_to_unmask] = sampled_tokens[top_indices]

    still_masked = seq == MASK_ID
    if still_masked.any():
        t = torch.zeros(B, device=device)
        logits = model(
            seq, field_indices, desig_nums, fnum_indices, t, pad_mask=pad_mask
        )
        final_preds = logits.argmax(dim=-1)
        seq[still_masked] = final_preds[still_masked]

    return seq


# ---------------------------------------------------------------------------
# Classification via likelihood scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_query_against_fnums(
    model,
    token_ids,
    pad_mask,
    candidate_fields,
    candidate_desig_nums,
    candidate_fnum_ids,
    device,
):
    """Score a single query against all candidate f_nums.

    Fully masks the query (t=1.0), then computes log P(real tokens | candidate)
    for each candidate. Returns log-prob per candidate.

    Args:
        token_ids: (SEQ_LEN,) the real token IDs
        pad_mask: (SEQ_LEN,) True for PAD positions
        candidate_fields: dict of field -> (N_candidates,) long tensors
        candidate_desig_nums: (N_candidates, 8)
        candidate_fnum_ids: (N_candidates,)
    Returns:
        log_probs: (N_candidates,) total log-prob for each candidate
    """
    N = candidate_fnum_ids.size(0)

    # Replicate query for all candidates
    masked_ids = token_ids.unsqueeze(0).expand(N, -1).clone()
    real_ids = token_ids.unsqueeze(0).expand(N, -1)
    pad = pad_mask.unsqueeze(0).expand(N, -1)

    # Mask all non-pad tokens
    masked_ids[~pad] = MASK_ID

    # t=1.0 for all (fully masked)
    t = torch.ones(N, device=device)

    # Forward pass — score all candidates at once
    # Batch in chunks if N is large to avoid OOM
    chunk_size = 512
    all_log_probs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_fields = {f: v[start:end] for f, v in candidate_fields.items()}
        chunk_logits = model(
            masked_ids[start:end],
            chunk_fields,
            candidate_desig_nums[start:end],
            candidate_fnum_ids[start:end],
            t[start:end],
            pad_mask=pad[start:end],
        )
        # Log-softmax over vocab, gather real token log-probs
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        # Gather log-prob of real tokens at each position
        real_chunk = real_ids[start:end]
        gathered = log_probs.gather(2, real_chunk.unsqueeze(-1)).squeeze(
            -1
        )  # (chunk, SEQ_LEN)
        # Zero out pad positions
        gathered[pad[start:end]] = 0.0
        # Sum over positions
        all_log_probs.append(gathered.sum(dim=1))

    return torch.cat(all_log_probs)


def classify(
    model,
    all_examples,
    token_vocab,
    field_vocabs,
    fnum_vocab,
    fnum_records,
    args,
    device,
):
    """Run MDLM as classifier on test split."""
    model.eval()

    # Build candidate tensors (one row per f_num)
    fnum_list = sorted(fnum_vocab.keys())
    N = len(fnum_list)

    cand_fields = {
        f: torch.zeros(N, dtype=torch.long, device=device) for f in COND_FIELDS
    }
    cand_desig_nums = torch.zeros(N, 8, dtype=torch.long, device=device)
    cand_fnum_ids = torch.zeros(N, dtype=torch.long, device=device)

    for i, fnum in enumerate(fnum_list):
        rec = fnum_records[fnum]
        for field in COND_FIELDS:
            val = rec[field]
            if val == -100:
                cand_fields[field][i] = 0
            else:
                cand_fields[field][i] = field_vocabs[field].get(val, 0)
        cand_desig_nums[i] = torch.tensor(
            encode_desig_num(rec["desig_num"]), dtype=torch.long
        )
        cand_fnum_ids[i] = fnum_vocab[fnum]

    # Get test examples
    test_examples = [
        ex
        for ex in all_examples
        if ex["split"] == "test"
        and ex["records"]
        and isinstance(ex["records"][0]["union_name"], str)
        and isinstance(ex["records"][0]["f_num"], int)
        and ex["records"][0]["f_num"] in fnum_vocab
    ]
    if args.classify_max_test:
        test_examples = test_examples[: args.classify_max_test]

    print(f"\n=== Classification ({len(test_examples)} test, {N} candidates) ===")

    correct = 0
    total = 0
    t0 = time.time()

    for i, ex in enumerate(test_examples):
        rec = ex["records"][0]
        true_fnum = rec["f_num"]

        ids = tokenize_to_ids(
            ex["query"], token_vocab, record=rec if args.use_templates else None
        )
        token_ids = torch.tensor(ids, dtype=torch.long, device=device)
        pad_mask = token_ids == PAD_ID

        scores = score_query_against_fnums(
            model,
            token_ids,
            pad_mask,
            cand_fields,
            cand_desig_nums,
            cand_fnum_ids,
            device,
        )

        pred_idx = scores.argmax().item()
        pred_fnum = fnum_list[pred_idx]

        is_correct = pred_fnum == true_fnum
        correct += int(is_correct)
        total += 1

        if not is_correct:
            true_rec = fnum_records[true_fnum]
            pred_rec = fnum_records[pred_fnum]
            true_label = f"{true_rec['union_name']} #{true_rec.get('desig_num', '')}"
            pred_label = f"{pred_rec['union_name']} #{pred_rec.get('desig_num', '')}"
            true_rank = (scores >= scores[fnum_list.index(true_fnum)]).sum().item()
            print(f"  WRONG: \"{ex['query']}\"")
            print(f"    true: f_num={true_fnum} ({true_label})")
            print(f"    pred: f_num={pred_fnum} ({pred_label})")
            print(f"    true_rank: {true_rank}/{N}", flush=True)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  {i+1}/{len(test_examples)} | acc={100*correct/total:.1f}% | "
                f"{elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\nFinal: {correct}/{total} = {100*correct/total:.1f}% "
        f"({elapsed:.1f}s, {elapsed/total:.2f}s/query)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=20,
        help="Number of synthetic examples per f_num",
    )
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--mdlm-layers", type=int, default=3)
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=5,
        help="Generate for f_nums with fewer than this many examples",
    )
    parser.add_argument(
        "--show-common", action="store_true", help="Also show samples for common f_nums"
    )
    parser.add_argument(
        "--save-checkpoint", type=str, default=None, help="Save model after training"
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Load model and skip training"
    )
    parser.add_argument(
        "--compositional-fnum",
        action="store_true",
        help="Compositional f_num embedding: union_name_emb + desig_name_emb + small residual",
    )
    parser.add_argument(
        "--fnum-dropout",
        type=float,
        default=0.0,
        help="Probability of zeroing fnum_idx during training (forces field-only generation)",
    )
    parser.add_argument(
        "--drop-rare-fnum",
        action="store_true",
        help="At generation time, use fnum_idx=0 for rare f_nums",
    )
    parser.add_argument(
        "--use-templates",
        action="store_true",
        help="Replace desig_num/prefix/suffix with placeholder tokens",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run as classifier: score test queries against all f_nums",
    )
    parser.add_argument(
        "--classify-max-test",
        type=int,
        default=None,
        help="Max test examples to classify",
    )
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # --- Load data ---
    with open("training/data/training_examples.json") as f:
        all_examples = json.load(f)

    train_examples = [
        ex for ex in all_examples if ex["split"] == "train" and ex["records"]
    ]

    if args.max_examples:
        train_examples = train_examples[: args.max_examples]

    # Build vocabs
    token_vocab = build_token_vocab(train_examples)
    vocab_size = len(token_vocab) + VOCAB_OFFSET
    inv_token_vocab = {v: k for k, v in token_vocab.items()}

    field_vocabs = build_field_vocabs(train_examples)
    field_vocab_sizes = {f: len(v) for f, v in field_vocabs.items()}

    # Count examples per f_num and build f_num vocab
    fnum_counts = Counter()
    fnum_records = {}  # f_num -> record (for generation)
    for ex in train_examples:
        rec = ex["records"][0]
        fnum = rec["f_num"]
        if isinstance(fnum, int) and fnum > 0:
            fnum_counts[fnum] += 1
            fnum_records[fnum] = rec

    # f_num vocab: 0=unknown, 1..N=known f_nums
    fnum_vocab = {fn: i + 1 for i, fn in enumerate(sorted(fnum_counts.keys()))}
    n_fnums = len(fnum_vocab)

    print(f"Token vocab: {vocab_size}")
    for f, v in field_vocabs.items():
        print(f"  {f}: {len(v)} values")
    print(f"f_nums: {n_fnums}")
    print(f"Train examples: {len(train_examples)}")
    rare_fnums = {fn for fn, c in fnum_counts.items() if c < args.rare_threshold}
    print(f"Rare f_nums (<{args.rare_threshold} examples): {len(rare_fnums)}/{n_fnums}")

    # --- Build fnum -> field index mapping (for compositional embedding) ---
    fnum_field_map = None
    fnum_desig_nums_map = None
    if args.compositional_fnum:
        # For each f_num, store its field indices and desig_num encoding
        # Shape: (n_fnums+1, len(COND_FIELDS)) and (n_fnums+1, 8)
        fnum_field_map = torch.zeros(n_fnums + 1, len(COND_FIELDS), dtype=torch.long)
        fnum_desig_nums_map = torch.zeros(n_fnums + 1, 8, dtype=torch.long)
        for fnum, rec in fnum_records.items():
            fidx = fnum_vocab.get(fnum, 0)
            if fidx == 0:
                continue
            for j, field in enumerate(COND_FIELDS):
                val = rec[field]
                if val == -100:
                    fnum_field_map[fidx, j] = 0
                else:
                    fnum_field_map[fidx, j] = field_vocabs[field].get(val, 0)
            fnum_desig_nums_map[fidx] = torch.tensor(
                encode_desig_num(rec["desig_num"]), dtype=torch.long
            )
        print("Compositional f_num embedding enabled")

    # --- Build dataset ---
    mdlm_ds = MDLMDataset(
        train_examples,
        token_vocab,
        field_vocabs,
        fnum_vocab,
        use_templates=args.use_templates,
    )
    mdlm_loader = DataLoader(
        mdlm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mdlm,
        drop_last=len(mdlm_ds) > args.batch_size,
    )

    # --- Model ---
    model = MDLMTransformer(
        vocab_size=vocab_size,
        field_vocab_sizes=field_vocab_sizes,
        n_fnums=n_fnums,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.mdlm_layers,
        ff_dim=args.d_model * 2,
        dropout=0.1,
        compositional_fnum=args.compositional_fnum,
        fnum_field_map=fnum_field_map,
        fnum_desig_nums=fnum_desig_nums_map,
    ).to(device)

    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}")
        model.load_state_dict(
            torch.load(args.load_checkpoint, map_location=device, weights_only=True)
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        # --- Train ---
        print(f"\n=== Training MDLM ({args.mdlm_epochs} epochs) ===")
        for epoch in range(args.mdlm_epochs):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            n_batches = 0

            for token_ids, lengths, fields, desig_nums, fnum_ids in mdlm_loader:
                token_ids = token_ids.to(device)
                lengths = lengths.to(device)
                desig_nums = desig_nums.to(device)
                fnum_ids = fnum_ids.to(device)
                fields_dev = {f: v.to(device) for f, v in fields.items()}

                B = token_ids.size(0)
                t = torch.rand(B, device=device) * 0.9 + 0.1

                # f_num dropout: randomly zero out fnum conditioning
                if args.fnum_dropout > 0:
                    drop_mask = torch.rand(B, device=device) < args.fnum_dropout
                    fnum_ids = fnum_ids.clone()
                    fnum_ids[drop_mask] = 0

                masked_ids, mask_flags = mask_tokens(token_ids, t, lengths)
                pad_mask = token_ids == PAD_ID

                logits = model(
                    masked_ids, fields_dev, desig_nums, fnum_ids, t, pad_mask=pad_mask
                )
                loss = mdlm_loss(logits, token_ids, mask_flags)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:2d}/{args.mdlm_epochs} | "
                f"loss={avg_loss:.3f} | {elapsed:.1f}s",
                flush=True,
            )

        if args.save_checkpoint:
            torch.save(model.state_dict(), args.save_checkpoint)
            print(f"Saved checkpoint to {args.save_checkpoint}")

    # --- Classify mode ---
    if args.classify:
        classify(
            model,
            all_examples,
            token_vocab,
            field_vocabs,
            fnum_vocab,
            fnum_records,
            args,
            device,
        )
        return

    # --- Generate samples ---
    print(f"\n=== Generating samples (temperature={args.temperature}) ===")
    model.eval()

    # Compute lengths per f_num and per field-structure key
    fnum_lengths = {}
    structure_lengths = (
        {}
    )  # (union_name, desig_name, has_prefix, has_dnum, has_suffix) -> [lengths]
    for ex in train_examples:
        rec = ex["records"][0]
        if not isinstance(rec["union_name"], str):
            continue
        ids = tokenize_to_ids(
            ex["query"], token_vocab, record=rec if args.use_templates else None
        )
        length = sum(1 for x in ids if x != PAD_ID)
        fnum = rec["f_num"]
        if isinstance(fnum, int) and fnum > 0:
            fnum_lengths.setdefault(fnum, []).append(length)
        struct_key = (
            rec["union_name"],
            rec["desig_name"] if rec["desig_name"] != -100 else None,
            rec["prefix"] not in (-100, 0),
            rec["desig_num"] not in (-100, 0),
            rec["suffix"] not in (-100, "", 0),
        )
        structure_lengths.setdefault(struct_key, []).append(length)

    # Also collect real texts per f_num for display
    fnum_texts = {}
    for ex in train_examples:
        rec = ex["records"][0]
        fnum = rec["f_num"]
        if isinstance(fnum, int) and fnum > 0:
            fnum_texts.setdefault(fnum, []).append(ex["query"])

    # Select which f_nums to show
    if args.show_common:
        # Show a mix: some rare, some common
        show_fnums = sorted(rare_fnums)[:20]
        common = [fn for fn, c in fnum_counts.most_common(10)]
        show_fnums.extend(common)
    else:
        show_fnums = sorted(rare_fnums)[:40]

    for fnum in show_fnums:
        if fnum not in fnum_records:
            continue
        rec = fnum_records[fnum]
        n_real = fnum_counts[fnum]

        # Prepare conditioning
        n_gen = args.n_synthetic
        field_idx = {}
        for field in COND_FIELDS:
            val = rec[field]
            if val == -100:
                idx = 0
            else:
                idx = field_vocabs[field].get(val, 0)
            field_idx[field] = torch.full(
                (n_gen,), idx, dtype=torch.long, device=device
            )

        desig_num_enc = encode_desig_num(rec["desig_num"])
        desig_nums = torch.tensor(
            [desig_num_enc] * n_gen, dtype=torch.long, device=device
        )

        # f_num conditioning: use 0 (unknown) for rare f_nums if --drop-rare-fnum
        if args.drop_rare_fnum and fnum in rare_fnums:
            fnum_idx = 0
        else:
            fnum_idx = fnum_vocab.get(fnum, 0)
        fnum_ids = torch.full((n_gen,), fnum_idx, dtype=torch.long, device=device)

        # For rare f_nums, sample lengths from examples with matching field structure
        # For common f_nums, use the f_num's own median length
        if fnum in rare_fnums:
            struct_key = (
                rec["union_name"],
                rec["desig_name"] if rec["desig_name"] != -100 else None,
                rec["prefix"] not in (-100, 0),
                rec["desig_num"] not in (-100, 0),
                rec["suffix"] not in (-100, "", 0),
            )
            all_lengths = structure_lengths.get(struct_key, fnum_lengths.get(fnum, [5]))
            sampled_lengths = random.choices(all_lengths, k=n_gen)
            lengths = torch.tensor(sampled_lengths, dtype=torch.long, device=device)
        else:
            median_len = sorted(fnum_lengths[fnum])[len(fnum_lengths[fnum]) // 2]
            lengths = torch.full((n_gen,), median_len, dtype=torch.long, device=device)

        generated = sample(
            model,
            field_idx,
            desig_nums,
            fnum_ids,
            lengths,
            n_steps=args.sample_steps,
            temperature=args.temperature,
            device=device,
        )

        synth_texts = [
            ids_to_text(
                generated[i].cpu().tolist(),
                inv_token_vocab,
                record=rec if args.use_templates else None,
            )
            for i in range(n_gen)
        ]
        n_unique = len(set(synth_texts))

        # Display
        uname = rec["union_name"]
        dname = rec["desig_name"] if rec["desig_name"] != -100 else ""
        dnum = rec["desig_num"] if rec["desig_num"] not in (-100, 0) else ""
        prefix = rec["prefix"] if rec["prefix"] not in (-100, 0) else ""
        suffix = rec["suffix"] if rec["suffix"] not in (-100, "") else ""

        label = f"f_num={fnum} | {uname}"
        if dname:
            label += f" {dname}"
        if prefix:
            label += f" pfx={prefix}"
        if dnum:
            label += f" #{dnum}"
        if suffix:
            label += f" sfx={suffix}"

        print(f"\n{label} ({n_real} real):")
        for rt in fnum_texts.get(fnum, [])[:3]:
            print(f"  real:  {rt}")
        for st in synth_texts[:10]:
            print(f"  synth: {st}")
        print(f"  [{n_unique}/{n_gen} unique]")


if __name__ == "__main__":
    main()
