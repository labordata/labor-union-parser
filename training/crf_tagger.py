"""CRF field tagger with latent alignment for number role disambiguation.

Tags numeric tokens as desig_num (DN), prefix (PFX), or suffix (SFX) using
a linear-chain CRF trained with constrained marginalization. We know the
field VALUES from the gazetteer but not which tokens they correspond to —
the loss marginalizes over all valid alignments (à la CTC).

Used as an auxiliary training loss to teach the encoder to represent
number roles. Not used at inference.
"""

import torch
import torch.nn as nn

# Tag IDs
TAG_O = 0
TAG_DN = 1
TAG_PFX = 2
TAG_SFX = 3
N_TAGS = 4


def find_valid_positions(tokens, record):
    """Find valid token positions for each field value.

    Returns (valid_dnum, valid_pfx, valid_sfx) — each is a list of
    token indices where that field value appears.
    """
    desig_num = record.get("desig_num", 0)
    prefix = record.get("prefix", 0)
    suffix = record.get("suffix", "")

    dn_str = (
        str(int(desig_num)) if desig_num and desig_num not in (-100, 0, None) else None
    )
    pfx_str = str(int(prefix)) if prefix and prefix not in (-100, 0, None) else None
    sfx_str = None
    if suffix and suffix not in (-100, 0, "", None):
        try:
            sfx_str = str(int(float(suffix)))
        except (ValueError, TypeError):
            pass

    valid_dnum, valid_pfx, valid_sfx = [], [], []
    for i, tok in enumerate(tokens):
        if dn_str is not None and tok == dn_str:
            valid_dnum.append(i)
        if pfx_str is not None and tok == pfx_str:
            valid_pfx.append(i)
        if sfx_str is not None and tok == sfx_str:
            valid_sfx.append(i)
    return valid_dnum, valid_pfx, valid_sfx


def build_field_tensors(valid_list, batch_size, device):
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


class CRFTaggerMixin:
    """Mixin providing CRF forward/constrained algorithms for a nn.Module.

    The host class must have:
        - self.tag_head: nn.Module mapping hidden states → (B, L, N_TAGS)
        - self._crf_trans: nn.Parameter (N_TAGS, N_TAGS)
        - self._crf_mask: buffer (N_TAGS, N_TAGS)
    """

    @property
    def crf_transitions(self):
        return self._crf_trans + self._crf_mask

    def crf_loss(self, tag_logits, lengths, crf_fields):
        """Compute CRF marginalization loss. Returns loss or None if NaN."""
        clamped = tag_logits.clamp(-20, 20)
        log_z = self._crf_forward(clamped, lengths)
        log_z_constrained = self._crf_constrained_log_z(clamped, lengths, crf_fields)
        loss = (log_z - log_z_constrained).mean()
        if torch.isnan(loss):
            return None
        return loss

    def _crf_forward(self, emissions, lengths):
        """Standard CRF forward algorithm for unconstrained log Z."""
        alpha = emissions[:, 0, :]
        trans = self.crf_transitions.unsqueeze(0)
        for i in range(1, emissions.shape[1]):
            scores = alpha.unsqueeze(2) + trans
            max_s = scores.max(dim=1, keepdim=True).values
            new_alpha = (
                (scores - max_s).exp().sum(dim=1).log()
                + max_s.squeeze(1)
                + emissions[:, i, :]
            )
            mask = (i < lengths).float().unsqueeze(1)
            alpha = mask * new_alpha + (1 - mask) * alpha
        max_a = alpha.max(dim=1, keepdim=True).values
        return (alpha - max_a).exp().sum(dim=1).log() + max_a.squeeze(1)

    def _crf_constrained_log_z(self, emissions, lengths, crf_fields):
        """Log Z over constrained tag sequences (latent alignment).

        Marginalizes over all valid alignments where each field value
        appears at exactly one of its matching positions.
        """
        trans = self.crf_transitions
        B, max_len = emissions.shape[0], emissions.shape[1]
        device = emissions.device

        o_em = emissions[:, :, TAG_O]
        pos_mask = torch.arange(max_len, device=device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        base_score = (o_em * pos_mask).sum(dim=1) + trans[TAG_O, TAG_O] * (
            lengths - 1
        ).float().clamp(min=0)

        trans_in = trans[TAG_O, :] - trans[TAG_O, TAG_O]
        trans_out = trans[:, TAG_O] - trans[TAG_O, TAG_O]
        emit_deltas = emissions - emissions[:, :, TAG_O : TAG_O + 1]

        pos_idx = torch.arange(max_len, device=device).unsqueeze(0)
        is_first = pos_idx == 0
        is_last = pos_idx >= (lengths - 1).unsqueeze(1)
        is_interior = ~is_first & ~is_last
        trans_delta = (
            is_interior.unsqueeze(2).float()
            * (trans_in + trans_out).unsqueeze(0).unsqueeze(0)
            + (is_first & ~is_last).unsqueeze(2).float()
            * trans_out.unsqueeze(0).unsqueeze(0)
            + (is_last & ~is_first).unsqueeze(2).float()
            * trans_in.unsqueeze(0).unsqueeze(0)
        )
        total_delta = emit_deltas + trans_delta

        field_options, field_positions = [], []
        for ft, tag in zip(crf_fields, [TAG_DN, TAG_PFX, TAG_SFX]):
            if ft is None:
                field_options.append(torch.zeros(B, 1, device=device))
                field_positions.append(
                    torch.full((B, 1), -1, dtype=torch.long, device=device)
                )
                continue
            pos_tensor, pos_valid, has_any = ft
            gathered = (
                total_delta[:, :, tag]
                .gather(1, pos_tensor)
                .masked_fill(~pos_valid, float("-inf"))
            )
            no_tag = torch.where(
                has_any,
                torch.full((B,), float("-inf"), device=device),
                torch.zeros(B, device=device),
            ).unsqueeze(1)
            field_options.append(torch.cat([no_tag, gathered], dim=1))
            real_pos = pos_tensor.clone()
            real_pos[~pos_valid] = -2
            field_positions.append(
                torch.cat(
                    [
                        torch.full((B, 1), -1, dtype=torch.long, device=device),
                        real_pos,
                    ],
                    dim=1,
                )
            )

        d_opts, p_opts, s_opts = field_options
        combo = (
            d_opts.unsqueeze(2).unsqueeze(3)
            + p_opts.unsqueeze(1).unsqueeze(3)
            + s_opts.unsqueeze(1).unsqueeze(2)
        )

        d_pos, p_pos, s_pos = field_positions
        de, pe, se = (
            d_pos.unsqueeze(2).unsqueeze(3),
            p_pos.unsqueeze(1).unsqueeze(3),
            s_pos.unsqueeze(1).unsqueeze(2),
        )
        conflict = (
            ((de == pe) & (de >= 0))
            | ((de == se) & (de >= 0))
            | ((pe == se) & (pe >= 0))
        )
        combo = combo.masked_fill(conflict, float("-inf"))

        for pa, pb, ta, tb, da, db in [
            (d_pos, p_pos, TAG_DN, TAG_PFX, 1, 2),
            (d_pos, s_pos, TAG_DN, TAG_SFX, 1, 3),
            (p_pos, s_pos, TAG_PFX, TAG_SFX, 2, 3),
        ]:
            sa, sb = [B, 1, 1, 1], [B, 1, 1, 1]
            sa[da], sb[db] = -1, -1
            pav, pbv = pa.view(sa), pb.view(sb)
            combo = combo + ((pav + 1 == pbv) & (pav >= 0)).float() * (
                trans[ta, tb]
                - trans[ta, TAG_O]
                - trans[TAG_O, tb]
                + trans[TAG_O, TAG_O]
            )
            combo = combo + ((pbv + 1 == pav) & (pbv >= 0)).float() * (
                trans[tb, ta]
                - trans[tb, TAG_O]
                - trans[TAG_O, ta]
                + trans[TAG_O, TAG_O]
            )

        flat = combo.view(B, -1)
        max_c = flat.max(dim=1, keepdim=True).values
        return base_score + (flat - max_c).exp().sum(dim=1).log() + max_c.squeeze(1)

    @staticmethod
    def init_crf_params(model, d_model):
        """Initialize CRF parameters on a model. Call in __init__."""
        model.tag_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, N_TAGS),
        )
        model._crf_trans = nn.Parameter(torch.zeros(N_TAGS, N_TAGS))
        crf_mask = torch.zeros(N_TAGS, N_TAGS)
        for tag in (TAG_DN, TAG_PFX, TAG_SFX):
            crf_mask[tag, tag] = float("-inf")
        model.register_buffer("_crf_mask", crf_mask)
