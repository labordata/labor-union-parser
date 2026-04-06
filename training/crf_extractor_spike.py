"""
CRF Field Extractor with Latent Alignment.

Extracts desig_num, prefix, and suffix spans from token sequences using a
linear-chain CRF trained with constrained marginalization. We know the field
*values* but not which tokens they correspond to — the loss marginalizes over
all valid alignments (à la CTC).

Tags: O=0, DNUM=1, PFX=2, SFX=3
"""

import argparse
import json
import sys
import time
from collections import Counter
from functools import partial

import torch
import torch.nn as nn

print = partial(print, flush=True)  # noqa: A001

sys.path.insert(0, "src")
from labor_union_parser.tokenizer import smart_truncate_nonspace  # noqa: E402

# Tag IDs
O, DNUM, PFX, SFX = 0, 1, 2, 3  # noqa: E741
TAG_NAMES = ["O", "DNUM", "PFX", "SFX"]
NUM_TAGS = 4


def load_data(path, exclude_sources=None, union_filter=None):
    """Load training_examples.json and build CRF training data.

    Returns list of dicts with keys:
        tokens: list of token strings (no padding)
        is_num: list of bools
        length: int
        desig_num: str or None
        prefix: str or None
        suffix: str or None
        split: str
    """
    with open(path) as f:
        raw = json.load(f)

    if exclude_sources:
        raw = [ex for ex in raw if ex.get("source") not in exclude_sources]
    if union_filter:
        raw = [
            ex
            for ex in raw
            if ex.get("union_name") and union_filter in ex["union_name"].lower()
        ]

    data = []
    skipped = 0
    total = len(raw)
    for idx, ex in enumerate(raw):
        if idx % 10000 == 0:
            print(f"  processing {idx}/{total}...")
        if not ex["records"]:
            skipped += 1
            continue

        rec = ex["records"][0]

        # Parse field values; -100 and 0 are sentinels for absent
        dnum = rec.get("desig_num", 0)
        if dnum in (0, -100, None):
            dnum_str = None
        else:
            dnum_str = str(int(dnum))

        pfx = rec.get("prefix", 0)
        if pfx in (0, -100, None):
            pfx_str = None
        else:
            pfx_str = str(int(pfx))

        sfx = rec.get("suffix", "")
        if sfx in ("", -100, None):
            sfx_str = None
        else:
            sfx_str = sfx.lower()

        # Tokenize
        tok_dicts = smart_truncate_nonspace(ex["query"])
        tokens = []
        is_num = []
        for td in tok_dicts:
            if td["token_type"] == 4:  # pad
                break
            tokens.append(td["token"])
            is_num.append(bool(td["is_num"]))

        length = len(tokens)
        if length == 0:
            skipped += 1
            continue

        # Find valid positions for each field
        valid_dnum = []
        valid_pfx = []
        valid_sfx = []

        for i, tok in enumerate(tokens):
            if dnum_str is not None and tok == dnum_str:
                valid_dnum.append(i)
            if pfx_str is not None and tok == pfx_str:
                valid_pfx.append(i)
            if sfx_str is not None and tok == sfx_str:
                valid_sfx.append(i)

        # Skip if any required field has no valid position in tokens
        if dnum_str is not None and not valid_dnum:
            skipped += 1
            continue
        if pfx_str is not None and not valid_pfx:
            skipped += 1
            continue
        if sfx_str is not None and not valid_sfx:
            skipped += 1
            continue

        # Skip if two present fields share all valid positions (unsolvable)
        present_fields = [
            (s, v)
            for s, v in [
                (dnum_str, valid_dnum),
                (pfx_str, valid_pfx),
                (sfx_str, valid_sfx),
            ]
            if s is not None
        ]
        conflict = False
        for a in range(len(present_fields)):
            for b in range(a + 1, len(present_fields)):
                sa, va = present_fields[a]
                sb, vb = present_fields[b]
                if set(va) == set(vb) and len(va) == 1:
                    conflict = True
        if conflict:
            skipped += 1
            continue

        data.append(
            {
                "tokens": tokens,
                "is_num": is_num,
                "length": length,
                "desig_num": dnum_str,
                "prefix": pfx_str,
                "suffix": sfx_str,
                "valid_dnum": valid_dnum,
                "valid_pfx": valid_pfx,
                "valid_sfx": valid_sfx,
                "has_fnum": bool(ex.get("f_num")) and ex.get("f_num") != -100,
                "split": ex["split"],
            }
        )

    return data, skipped


def build_vocab(data):
    """Build token → id mapping from training data."""
    counter = Counter()
    for ex in data:
        if ex["split"] == "train":
            for tok in ex["tokens"]:
                counter[tok] += 1

    # Keep tokens appearing 2+ times, plus special tokens
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, count in counter.most_common():
        if count >= 2:
            vocab[tok] = len(vocab)
    return vocab


class CRFExtractor(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_attn_layers=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.num_flag = nn.Linear(1, d_model)

        # Self-attention with RoPE for relative position encoding
        self.attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.attn_layers.append(
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(d_model, d_model),
                        "k_proj": nn.Linear(d_model, d_model),
                        "v_proj": nn.Linear(d_model, d_model),
                        "out_proj": nn.Linear(d_model, d_model),
                        "norm1": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 2),
                            nn.GELU(),
                            nn.Linear(d_model * 2, d_model),
                        ),
                        "norm2": nn.LayerNorm(d_model),
                    }
                )
            )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, NUM_TAGS),
        )

        # CRF transition parameters: transitions[i, j] = score for tag i -> tag j
        # Only the O row (transitions from O) is fully learnable.
        # From any non-O tag, only transition back to O is allowed.
        self._trans_logits = nn.Parameter(torch.zeros(NUM_TAGS, NUM_TAGS))

        # Mask: -inf for same-tag consecutive (forces single-token spans)
        mask = torch.zeros(NUM_TAGS, NUM_TAGS)
        for tag in (DNUM, PFX, SFX):
            mask[tag, tag] = float("-inf")
        self.register_buffer("_trans_mask", mask)

    @property
    def transitions(self):
        return self._trans_logits + self._trans_mask

    @staticmethod
    def _rope(x, seq_len):
        """Apply rotary position embeddings to x: (batch, heads, seq_len, head_dim)."""
        head_dim = x.shape[-1]
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        dim_idx = torch.arange(0, head_dim, 2, device=x.device, dtype=x.dtype)
        freq = 1.0 / (10000.0 ** (dim_idx / head_dim))
        angles = pos * freq  # (seq_len, head_dim//2)
        cos = angles.cos()  # (seq_len, head_dim//2)
        sin = angles.sin()
        # Reshape for broadcast: (1, 1, seq_len, head_dim//2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(
            -2
        )

    def get_emissions(self, token_ids, is_num, lengths):
        """Compute per-token emission scores.

        Args:
            token_ids: (batch, max_len) int tensor
            is_num: (batch, max_len) float tensor
            lengths: (batch,) int tensor

        Returns:
            emissions: (batch, max_len, NUM_TAGS) float tensor
        """
        batch_size, max_len = token_ids.shape
        head_dim = self.d_model // self.n_heads

        x = self.token_embed(token_ids)
        x = x + self.num_flag(is_num.unsqueeze(-1))

        # Padding mask for attention: -inf where padded
        pad_mask = torch.arange(max_len, device=token_ids.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(1)
        attn_mask = (
            pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9
        )  # (batch, 1, 1, max_len)

        for layer in self.attn_layers:
            residual = x
            x = layer["norm1"](x)

            q = (
                layer["q_proj"](x)
                .view(batch_size, max_len, self.n_heads, head_dim)
                .transpose(1, 2)
            )
            k = (
                layer["k_proj"](x)
                .view(batch_size, max_len, self.n_heads, head_dim)
                .transpose(1, 2)
            )
            v = (
                layer["v_proj"](x)
                .view(batch_size, max_len, self.n_heads, head_dim)
                .transpose(1, 2)
            )

            # Apply RoPE to queries and keys
            q = self._rope(q, max_len)
            k = self._rope(k, max_len)

            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
            scores = scores + attn_mask
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

            attn_out = (
                attn_out.transpose(1, 2)
                .contiguous()
                .view(batch_size, max_len, self.d_model)
            )
            x = residual + layer["out_proj"](attn_out)

            # FFN with residual
            residual = x
            x = residual + layer["ff"](layer["norm2"](x))

        return self.mlp(x)

    def forward_algorithm(self, emissions, lengths):
        """Standard CRF forward algorithm for unconstrained log Z.

        Args:
            emissions: (batch, max_len, NUM_TAGS)
            lengths: (batch,) int

        Returns:
            log_Z: (batch,) float
        """
        batch_size, max_len, _ = emissions.shape

        # alpha[b, t] = log sum over all tag sequences ending in tag t at current position
        alpha = emissions[:, 0, :]  # (batch, NUM_TAGS)

        for i in range(1, max_len):
            # alpha_expand: (batch, NUM_TAGS, 1) + transitions: (NUM_TAGS, NUM_TAGS)
            # → (batch, NUM_TAGS, NUM_TAGS), then logsumexp over prev tag dim
            emit_i = emissions[:, i, :]  # (batch, NUM_TAGS)
            trans = self.transitions.unsqueeze(0)  # (1, NUM_TAGS, NUM_TAGS)

            # new_alpha[b, j] = logsumexp_over_i(alpha[b, i] + trans[i, j]) + emit[b, j]
            scores = alpha.unsqueeze(2) + trans  # (batch, NUM_TAGS, NUM_TAGS)
            new_alpha = torch.logsumexp(scores, dim=1) + emit_i  # (batch, NUM_TAGS)

            # Mask: only update for positions within sequence length
            mask = (i < lengths).float().unsqueeze(1)  # (batch, 1)
            alpha = mask * new_alpha + (1 - mask) * alpha

        return torch.logsumexp(alpha, dim=1)  # (batch,)

    def constrained_log_z(self, emissions, lengths, field_tensors):
        """Compute log Z over constrained tag sequences (hard labels).

        For each field, exactly one valid position must be tagged (if the field
        is present). Fields that are absent have no valid positions.

        Vectorized: builds per-field delta tensors, outer-sums them with
        conflict masking, then logsumexp over all valid combos.

        field_tensors: list of 3 elements (one per field: DNUM, PFX, SFX),
            each is None or (pos_tensor, pos_valid, has_none, has_any)
        """
        trans = self.transitions
        batch_size = emissions.shape[0]
        max_len = emissions.shape[1]
        device = emissions.device

        # Baseline: all-O score for every example
        o_emissions = emissions[:, :, O]  # (batch, max_len)
        pos_mask = torch.arange(max_len, device=device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        base_emit = (o_emissions * pos_mask).sum(dim=1)  # (batch,)
        base_trans = trans[O, O] * (lengths - 1).float().clamp(min=0)  # (batch,)
        base_score = base_emit + base_trans  # (batch,)

        # Precompute transition deltas
        trans_in = trans[O, :] - trans[O, O]
        trans_out = trans[:, O] - trans[O, O]
        trans_interior = trans_in + trans_out

        # Emission deltas: (batch, max_len, NUM_TAGS)
        emit_deltas = emissions - emissions[:, :, O : O + 1]

        # Position-dependent transition deltas: (batch, max_len, NUM_TAGS)
        pos_idx = torch.arange(max_len, device=device).unsqueeze(0)
        is_first = pos_idx == 0
        is_last = pos_idx >= (lengths - 1).unsqueeze(1)
        is_interior = ~is_first & ~is_last

        trans_delta = (
            is_interior.unsqueeze(2).float() * trans_interior.unsqueeze(0).unsqueeze(0)
            + (is_first & ~is_last).unsqueeze(2).float()
            * trans_out.unsqueeze(0).unsqueeze(0)
            + (is_last & ~is_first).unsqueeze(2).float()
            * trans_in.unsqueeze(0).unsqueeze(0)
        )

        total_delta = emit_deltas + trans_delta  # (batch, max_len, NUM_TAGS)

        # For each field, build options tensor: (batch, n_options)
        # If field absent: single option with delta=0 (no tag change)
        # If field present: one option per valid position
        field_options = []
        field_positions = []  # for conflict detection

        for ft, tag in zip(field_tensors, [DNUM, PFX, SFX]):
            if ft is None:
                # No examples have this field — single no-tag option
                field_options.append(torch.zeros(batch_size, 1, device=device))
                field_positions.append(
                    torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
                )
                continue

            pos_tensor, pos_valid, has_any = ft

            # Gather deltas: (batch, max_pos)
            deltas = total_delta[:, :, tag]
            gathered = deltas.gather(1, pos_tensor)
            gathered = gathered.masked_fill(~pos_valid, float("-inf"))

            # Prepend no-tag option (delta=0) for examples without this field
            no_tag = torch.where(
                has_any,
                torch.full((batch_size,), float("-inf"), device=device),
                torch.zeros(batch_size, device=device),
            ).unsqueeze(1)

            field_options.append(torch.cat([no_tag, gathered], dim=1))

            # Position indices for conflict detection
            no_tag_pos = torch.full(
                (batch_size, 1), -1, dtype=torch.long, device=device
            )
            real_pos = pos_tensor.clone()
            real_pos[~pos_valid] = -2  # unique negative for invalid
            field_positions.append(torch.cat([no_tag_pos, real_pos], dim=1))

        # Outer sum: combo_scores[b, d, p, s] = d_delta + p_delta + s_delta
        d_opts, p_opts, s_opts = field_options
        combo_scores = (
            d_opts.unsqueeze(2).unsqueeze(3)
            + p_opts.unsqueeze(1).unsqueeze(3)
            + s_opts.unsqueeze(1).unsqueeze(2)
        )  # (batch, D, P, S)

        # Mask combos where two fields claim the same position
        d_pos, p_pos, s_pos = field_positions
        d_exp = d_pos.unsqueeze(2).unsqueeze(3)
        p_exp = p_pos.unsqueeze(1).unsqueeze(3)
        s_exp = s_pos.unsqueeze(1).unsqueeze(2)

        dp_conflict = (d_exp == p_exp) & (d_exp >= 0)
        ds_conflict = (d_exp == s_exp) & (d_exp >= 0)
        ps_conflict = (p_exp == s_exp) & (p_exp >= 0)
        conflict = dp_conflict | ds_conflict | ps_conflict
        combo_scores = combo_scores.masked_fill(conflict, float("-inf"))

        # Adjacency corrections: when two tagged positions are adjacent,
        # the independent delta assumption double-counts a trans[O,O].
        # correction(A→B) = trans[A,B] - trans[A,O] - trans[O,B] + trans[O,O]
        for pos_a, pos_b, tag_a, tag_b, dim_a, dim_b in [
            (d_pos, p_pos, DNUM, PFX, 1, 2),
            (d_pos, s_pos, DNUM, SFX, 1, 3),
            (p_pos, s_pos, PFX, SFX, 2, 3),
        ]:
            # Expand positions to combo shape (batch, D, P, S)
            shape_a = [batch_size, 1, 1, 1]
            shape_a[dim_a] = -1
            shape_b = [batch_size, 1, 1, 1]
            shape_b[dim_b] = -1
            pa = pos_a.view(shape_a)
            pb = pos_b.view(shape_b)

            # a immediately before b
            fwd = (pa + 1 == pb) & (pa >= 0)
            corr_fwd = (
                trans[tag_a, tag_b] - trans[tag_a, O] - trans[O, tag_b] + trans[O, O]
            )
            # b immediately before a
            bwd = (pb + 1 == pa) & (pb >= 0)
            corr_bwd = (
                trans[tag_b, tag_a] - trans[tag_b, O] - trans[O, tag_a] + trans[O, O]
            )

            combo_scores = (
                combo_scores + fwd.float() * corr_fwd + bwd.float() * corr_bwd
            )

        # logsumexp over all combos
        combo_flat = combo_scores.view(batch_size, -1)
        results = base_score + torch.logsumexp(combo_flat, dim=1)

        return results

    def viterbi_decode(self, emissions, lengths):
        """Viterbi decode to find best tag sequence.

        Returns:
            best_tags: list of lists of ints
        """
        batch_size, max_len, _ = emissions.shape
        device = emissions.device

        all_tags = []
        for b in range(batch_size):
            L = lengths[b].item()
            emit = emissions[b, :L, :]  # (L, NUM_TAGS)

            # Viterbi
            viterbi = torch.zeros(L, NUM_TAGS, device=device)
            backpointers = torch.zeros(L, NUM_TAGS, dtype=torch.long, device=device)

            viterbi[0] = emit[0]

            for i in range(1, L):
                scores = (
                    viterbi[i - 1].unsqueeze(1) + self.transitions
                )  # (NUM_TAGS, NUM_TAGS)
                best_scores, best_prev = scores.max(dim=0)  # (NUM_TAGS,)
                viterbi[i] = best_scores + emit[i]
                backpointers[i] = best_prev

            # Traceback
            tags = [0] * L
            tags[-1] = viterbi[-1].argmax().item()
            for i in range(L - 2, -1, -1):
                tags[i] = backpointers[i + 1, tags[i + 1]].item()

            all_tags.append(tags)

        return all_tags


def _build_field_tensors(valid_list, batch_size, device):
    """Build padded position tensor, validity mask, has_any for a field."""
    max_pos = max((len(v) for v in valid_list), default=0)
    if max_pos == 0:
        return None

    pos_tensor = torch.zeros(batch_size, max_pos, dtype=torch.long, device=device)
    pos_valid = torch.zeros(batch_size, max_pos, dtype=torch.bool, device=device)
    has_any = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for b, positions in enumerate(valid_list):
        if not positions:
            continue
        has_any[b] = True
        for j, p in enumerate(positions):
            if j < max_pos:
                pos_tensor[b, j] = p
                pos_valid[b, j] = True

    return pos_tensor, pos_valid, has_any


def collate_batch(batch, vocab, device):
    """Collate a list of examples into batched tensors."""
    max_len = max(ex["length"] for ex in batch)
    batch_size = len(batch)

    token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    is_num_t = torch.zeros(batch_size, max_len, dtype=torch.float, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    valid_dnum = []
    valid_pfx = []
    valid_sfx = []

    for i, ex in enumerate(batch):
        L = ex["length"]
        lengths[i] = L
        ids = [vocab.get(tok, 1) for tok in ex["tokens"]]
        token_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        nums = [float(n) for n in ex["is_num"]]
        is_num_t[i, :L] = torch.tensor(nums, dtype=torch.float)

        valid_dnum.append(ex["valid_dnum"])
        valid_pfx.append(ex["valid_pfx"])
        valid_sfx.append(ex["valid_sfx"])

    # Precompute field tensors for vectorized constrained_log_z
    field_tensors = []
    for vlist in [valid_dnum, valid_pfx, valid_sfx]:
        field_tensors.append(_build_field_tensors(vlist, batch_size, device))

    return (
        token_ids,
        is_num_t,
        lengths,
        field_tensors,
        batch,
    )


def evaluate(model, data, vocab, device, batch_size=512):
    """Evaluate per-field extraction accuracy.

    Returns dict with keys per field, each containing:
        present_correct, present_total: field-present accuracy
        absent_correct, absent_total: field-absent accuracy (correctly not tagged)
        ambig_correct, ambig_total: Counter by n_valid_positions
    """
    model.eval()

    results = {}
    for field in ["desig_num", "prefix", "suffix"]:
        results[field] = {
            "present_correct": 0,
            "present_total": 0,
            "ambig_correct": Counter(),
            "ambig_total": Counter(),
        }

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            token_ids, is_num_t, lengths, field_tensors, _ = collate_batch(
                batch, vocab, device
            )
            emissions = model.get_emissions(token_ids, is_num_t, lengths)
            tag_seqs = model.viterbi_decode(emissions, lengths)

            for i, ex in enumerate(batch):
                # Only evaluate examples with an f_num
                if not ex["has_fnum"]:
                    continue

                tags = tag_seqs[i]
                tokens = ex["tokens"]

                # Extract values from tags
                extracted = {"desig_num": None, "prefix": None, "suffix": None}
                tag_to_field = {DNUM: "desig_num", PFX: "prefix", SFX: "suffix"}
                for pos, tag in enumerate(tags):
                    if tag in tag_to_field:
                        field = tag_to_field[tag]
                        extracted[field] = tokens[pos]

                # Compare to ground truth
                for field, gt_key, valid_key in [
                    ("desig_num", "desig_num", "valid_dnum"),
                    ("prefix", "prefix", "valid_pfx"),
                    ("suffix", "suffix", "valid_sfx"),
                ]:
                    gt = ex[gt_key]
                    r = results[field]
                    n_valid = len(ex[valid_key])
                    if gt is None and not ex[valid_key]:
                        # Field absent in GT — don't penalize predictions
                        # (model may be finding real structure GT missed)
                        continue
                    elif gt is not None:
                        # Field present
                        r["present_total"] += 1
                        r["ambig_total"][n_valid] += 1
                        if extracted[field] == gt:
                            r["present_correct"] += 1
                            r["ambig_correct"][n_valid] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="CRF field extractor spike")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--exclude-source",
        type=str,
        nargs="+",
        default=["synthetic"],
        help="Sources to exclude (default: synthetic)",
    )
    parser.add_argument(
        "--data", type=str, default="training/data/training_examples.json"
    )
    parser.add_argument(
        "--union-filter",
        type=str,
        default=None,
        help="Filter to union_name containing this substring (lowercase match)",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Save model checkpoint (state_dict + vocab + hparams) to this path",
    )
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load all data (no cap — max_examples only limits training)
    print("Loading data...")
    data, skipped = load_data(
        args.data,
        exclude_sources=set(args.exclude_source) if args.exclude_source else None,
        union_filter=args.union_filter,
    )
    print(f"Loaded {len(data)} examples ({skipped} skipped)")

    import random

    random.seed(42)

    train_data = [ex for ex in data if ex["split"] == "train" and ex["has_fnum"]]
    test_data = [ex for ex in data if ex["split"] == "test" and ex["has_fnum"]]

    # Val: all prefix/suffix examples + 500 random from the rest
    val_all = [ex for ex in data if ex["split"] == "val" and ex["has_fnum"]]
    val_pfx_sfx = [
        ex for ex in val_all if ex["prefix"] is not None or ex["suffix"] is not None
    ]
    val_rest = [ex for ex in val_all if ex["prefix"] is None and ex["suffix"] is None]
    val_data = val_pfx_sfx + random.sample(val_rest, min(500, len(val_rest)))

    if args.max_examples and len(train_data) > args.max_examples:
        train_data = random.sample(train_data, args.max_examples)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Report field coverage
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        has_dnum = sum(1 for ex in split_data if ex["desig_num"] is not None)
        has_pfx = sum(1 for ex in split_data if ex["prefix"] is not None)
        has_sfx = sum(1 for ex in split_data if ex["suffix"] is not None)
        print(
            f"  {split_name}: dnum={has_dnum}/{len(split_data)} "
            f"pfx={has_pfx}/{len(split_data)} sfx={has_sfx}/{len(split_data)}"
        )

    # Report ambiguity stats
    for field, valid_key in [
        ("dnum", "valid_dnum"),
        ("pfx", "valid_pfx"),
        ("sfx", "valid_sfx"),
    ]:
        counts = Counter(len(ex[valid_key]) for ex in train_data if ex[valid_key])
        if counts:
            print(
                f"  {field} valid positions: "
                + ", ".join(f"{k}pos={v}" for k, v in sorted(counts.items())[:6])
            )

    # Build vocab
    vocab = build_vocab(data)
    print(f"Vocab size: {len(vocab)}")

    # Model
    model = CRFExtractor(
        len(vocab), d_model=args.d_model, n_attn_layers=args.n_layers
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_train = len(train_data)

    # Build sampling weights: oversample examples with prefix/suffix
    n_has_pfx = sum(1 for ex in train_data if ex["prefix"] is not None)
    n_has_sfx = sum(1 for ex in train_data if ex["suffix"] is not None)
    target_frac = 0.25
    pfx_oversample = min((target_frac * n_train) / max(n_has_pfx, 1), 20.0)
    sfx_oversample = min((target_frac * n_train) / max(n_has_sfx, 1), 20.0)
    sample_weights = torch.ones(n_train)
    for i, ex in enumerate(train_data):
        if ex["prefix"] is not None:
            sample_weights[i] = max(sample_weights[i], pfx_oversample)
        if ex["suffix"] is not None:
            sample_weights[i] = max(sample_weights[i], sfx_oversample)
    print(f"\nOversampling: pfx={pfx_oversample:.1f}x, sfx={sfx_oversample:.1f}x")

    # Training loop
    print(
        f"\n{'Epoch':>7} | {'Loss':>8} | {'dnum+':>8} | {'pfx+':>8} | "
        f"{'sfx+':>8} | {'Time':>6}"
    )
    print("-" * 62)
    print("  (dnum+/pfx+/sfx+ = field-present accuracy only)")

    import copy

    best_val_errors = float("inf")
    best_state = None
    patience = 5
    wait = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        sampled = torch.multinomial(sample_weights, n_train, replacement=True).tolist()
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            batch = [train_data[i] for i in sampled[start : start + args.batch_size]]

            token_ids, is_num_t, lengths, field_tensors, _ = collate_batch(
                batch, vocab, device
            )

            emissions = model.get_emissions(token_ids, is_num_t, lengths)

            # Unconstrained log Z
            log_z = model.forward_algorithm(emissions, lengths)

            # Constrained log Z
            log_z_constrained = model.constrained_log_z(
                emissions,
                lengths,
                field_tensors,
            )

            loss = (log_z - log_z_constrained).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate on val set
        val_results = evaluate(model, val_data, vocab, device)

        def present_acc(r):
            return r["present_correct"] / max(r["present_total"], 1)

        def present_errors(r):
            return r["present_total"] - r["present_correct"]

        dnum_acc = present_acc(val_results["desig_num"])
        pfx_acc = present_acc(val_results["prefix"])
        sfx_acc = present_acc(val_results["suffix"])

        val_errors = sum(
            present_errors(val_results[f]) for f in ["desig_num", "prefix", "suffix"]
        )

        # Early stopping on total val errors
        marker = ""
        if val_errors < best_val_errors:
            best_val_errors = val_errors
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            marker = " *"
        else:
            wait += 1

        elapsed = time.time() - t0
        print(
            f"  {epoch + 1:2d}/{args.epochs:2d} | {avg_loss:8.4f} | "
            f"err={val_errors:3d} | "
            f"{dnum_acc:7.1%} | {pfx_acc:7.1%} | {sfx_acc:7.1%} | "
            f"{elapsed:5.1f}s{marker}"
        )

        if wait >= patience:
            print(f"  Early stopping (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_errors={best_val_errors})")

    # Save checkpoint
    if args.save_checkpoint:
        checkpoint = {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
        }
        torch.save(checkpoint, args.save_checkpoint)
        print(f"  Checkpoint saved to {args.save_checkpoint}")

    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    test_results = evaluate(model, test_data, vocab, device)

    for field in ["desig_num", "prefix", "suffix"]:
        r = test_results[field]
        pc, pt = r["present_correct"], r["present_total"]
        p_acc = pc / max(pt, 1)
        print(f"  {field}: {pc}/{pt} = {p_acc:.1%}")

        # Ambiguity breakdown for present fields
        if r["ambig_total"]:
            parts = []
            for n_valid in sorted(r["ambig_total"].keys()):
                c = r["ambig_correct"][n_valid]
                t = r["ambig_total"][n_valid]
                a = c / max(t, 1)
                parts.append(f"{n_valid}pos: {c}/{t}={a:.0%}")
            print(f"    by ambiguity: {', '.join(parts)}")

    # Show misclassified examples
    print("\n--- Errors ---")
    model.eval()
    tag_to_field = {DNUM: "desig_num", PFX: "prefix", SFX: "suffix"}
    errors = []
    with torch.no_grad():
        for start in range(0, len(test_data), 512):
            batch = test_data[start : start + 512]
            token_ids, is_num_t, lengths, field_tensors, _ = collate_batch(
                batch, vocab, device
            )
            emissions = model.get_emissions(token_ids, is_num_t, lengths)
            tag_seqs = model.viterbi_decode(emissions, lengths)

            for i, ex in enumerate(batch):
                if not ex["has_fnum"]:
                    continue

                tags = tag_seqs[i]
                tokens = ex["tokens"]

                extracted = {"desig_num": None, "prefix": None, "suffix": None}
                for pos, tag in enumerate(tags):
                    if tag in tag_to_field:
                        extracted[tag_to_field[tag]] = tokens[pos]

                wrong_fields = []
                for field, gt_key, valid_key in [
                    ("desig_num", "desig_num", "valid_dnum"),
                    ("prefix", "prefix", "valid_pfx"),
                    ("suffix", "suffix", "valid_sfx"),
                ]:
                    gt = ex[gt_key]
                    pred = extracted[field]
                    if gt is None and not ex[valid_key]:
                        # Don't penalize predictions when GT is absent
                        continue
                    elif gt is not None:
                        if pred != gt:
                            wrong_fields.append(f"{field}: pred={pred} gt={gt}")

                if wrong_fields:
                    tagged = " ".join(
                        (
                            f"[{tokens[j]}:{TAG_NAMES[tags[j]]}]"
                            if tags[j] != O
                            else tokens[j]
                        )
                        for j in range(ex["length"])
                    )
                    errors.append((ex, tagged, wrong_fields))

    n_with_fnum = sum(1 for ex in test_data if ex["has_fnum"])
    print(f"  {len(errors)} errors out of {n_with_fnum} test examples (with f_num)")
    print()
    for ex, tagged, wrong_fields in errors:
        print(f"  query: {' '.join(ex['tokens'])}")
        print(f"  gt: dnum={ex['desig_num']} pfx={ex['prefix']} sfx={ex['suffix']}")
        print(f"  pred: {tagged}")
        print(f"  wrong: {', '.join(wrong_fields)}")
        print()


if __name__ == "__main__":
    main()
