#!/usr/bin/env python3
"""Train the factored ArcFace f_num classifier.

Uses PyTorch Lightning for structured training with:
- TrainingModel (LightningModule): encoder + ArcFace + CRF + disagree penalty
- ArcFaceDataModule (LightningDataModule): data loading, vocab, prototypes, collation
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from crf_tagger import CRFTaggerMixin, build_field_tensors, find_valid_positions

from labor_union_parser.arcface_model import (
    ArcFaceModel,
    FactoredPrototypeClassifier,
    FastTextRoPEEncoder,
)
from labor_union_parser.tokenizer import (
    DEFAULT_N_BUCKETS,
    NUM_BLOOM_HASHES,
    bloom_hash_ids,
    tokenize_for_arcface,
)

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
N_BUCKETS = DEFAULT_N_BUCKETS
ARCFACE_SCALE = 30.0
FNUM_REG = 100.0


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """All tensors for a training/validation batch."""

    token_ids: torch.Tensor
    ngram_ids: torch.Tensor
    ngram_counts: torch.Tensor
    bloom_ids: torch.Tensor
    is_num: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor
    union_targets: torch.Tensor
    desig_targets: torch.Tensor
    prefix_targets: torch.Tensor
    suffix_targets: torch.Tensor
    crf_dnum: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    crf_pfx: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    crf_sfx: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TrainingModel(CRFTaggerMixin, ArcFaceModel, L.LightningModule):
    """ArcFaceModel + CRF tag head + disagree penalty, wrapped for Lightning."""

    def __init__(
        self, n_classes, n_train_classes, vocab_size, field_sizes, class_to_union
    ):
        L.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["class_to_union"])

        self.scale = ARCFACE_SCALE
        self.encoder = FastTextRoPEEncoder(
            D_MODEL, N_HEADS, N_LAYERS, N_BUCKETS, vocab_size
        )
        self.classifier = FactoredPrototypeClassifier(
            D_MODEL, n_classes, field_sizes, scale=ARCFACE_SCALE
        )
        self.union_scale = nn.Parameter(torch.tensor(10.0))
        self.desig_scale = nn.Parameter(torch.tensor(10.0))
        CRFTaggerMixin.init_crf_params(self, D_MODEL)

        self.n_train_classes = n_train_classes
        self.register_buffer("class_to_union", class_to_union)

    def forward(self, batch: Batch):
        embeddings, hidden = self.encode(
            batch.token_ids,
            batch.ngram_ids,
            batch.ngram_counts,
            batch.bloom_ids,
            batch.is_num,
            batch.lengths,
        )
        logits, arcface_loss = self.classifier(embeddings, batch.targets)

        W_u = self.classifier.W_union.weight[1:]
        union_logits = self.union_scale * F.linear(embeddings, F.normalize(W_u, dim=1))
        W_dn = self.classifier.W_desig_name.weight[1:]
        desig_logits = self.desig_scale * F.linear(embeddings, F.normalize(W_dn, dim=1))
        tag_logits = self.tag_head(hidden)

        # Field classification losses (-100 targets ignored by default)
        field_losses = {
            "union_name": F.cross_entropy(union_logits, batch.union_targets),
            "desig_name": F.cross_entropy(desig_logits, batch.desig_targets),
        }

        # CRF token role tagging loss
        crf_fields = [batch.crf_dnum, batch.crf_pfx, batch.crf_sfx]
        crf_loss_val = self.crf_loss(tag_logits, batch.lengths, crf_fields)
        if crf_loss_val is not None:
            field_losses["crf_tags"] = crf_loss_val

        # Disagree penalty
        disagree_loss = torch.tensor(0.0, device=logits.device)
        fnum_valid = batch.targets >= 0
        if fnum_valid.any():
            fnum_probs = F.softmax(logits[fnum_valid], dim=1)
            union_lp = F.log_softmax(union_logits[fnum_valid], dim=1)
            disagree_loss = -(
                (fnum_probs * union_lp[:, self.class_to_union]).sum(dim=1).mean()
            )

        return logits, arcface_loss, field_losses, disagree_loss

    def training_step(self, batch: Batch):
        _, arcface_loss, field_losses, disagree_loss = self(batch)
        loss = arcface_loss + disagree_loss
        for fl in field_losses.values():
            loss = loss + fl
        loss = loss + FNUM_REG * self.classifier.W_fnum.pow(2).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch):
        embeddings, _ = self.encode(
            batch.token_ids,
            batch.ngram_ids,
            batch.ngram_counts,
            batch.bloom_ids,
            batch.is_num,
            batch.lengths,
        )
        logits, _ = self.classifier(embeddings)
        mask = batch.targets != -100
        if mask.any():
            preds = logits.argmax(dim=1)[mask]
            top1 = (preds == batch.targets[mask]).float().mean()
            self.log("val_top1", top1, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    def on_fit_start(self):
        with torch.no_grad():
            self.classifier.W_fnum.data[self.n_train_classes :].zero_()
        self.classifier.W_fnum.register_hook(
            lambda grad: grad.__setitem__(slice(self.n_train_classes, None), 0) or grad
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _collate_batch(examples) -> Batch:
    """Collate list of example dicts into a Batch."""
    B = len(examples)
    max_len = max(ex["length"] for ex in examples)
    max_ngrams = len(examples[0]["ngram_ids"][0])

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    ngram_ids = torch.zeros(B, max_len, max_ngrams, dtype=torch.long)
    ngram_counts = torch.zeros(B, max_len, dtype=torch.long)
    bloom_ids = torch.zeros(B, max_len, NUM_BLOOM_HASHES, dtype=torch.long)
    is_num = torch.zeros(B, max_len, dtype=torch.float)
    lengths = torch.zeros(B, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)
    union_tgt = torch.full((B,), -100, dtype=torch.long)
    desig_tgt = torch.full((B,), -100, dtype=torch.long)
    prefix_tgt = torch.full((B,), -100, dtype=torch.long)
    suffix_tgt = torch.full((B,), -100, dtype=torch.long)
    crf_dnum_list, crf_pfx_list, crf_sfx_list = [], [], []

    for i, ex in enumerate(examples):
        L = ex["length"]
        lengths[i] = L
        token_ids[i, :L] = torch.tensor(ex["token_ids"][:L], dtype=torch.long)
        ngram_ids[i, :L] = torch.tensor(ex["ngram_ids"][:L], dtype=torch.long)
        ngram_counts[i, :L] = torch.tensor(ex["ngram_counts"][:L], dtype=torch.long)
        bloom_ids[i, :L] = torch.tensor(ex["bloom_ids"][:L], dtype=torch.long)
        is_num[i, :L] = torch.tensor(ex["is_num_f"], dtype=torch.float)
        targets[i] = ex["target"]
        union_tgt[i] = ex["union_target"]
        desig_tgt[i] = ex["desig_name_target"]
        prefix_tgt[i] = ex["prefix_target"]
        suffix_tgt[i] = ex["suffix_target"]
        crf_dnum_list.append(ex["valid_dnum"])
        crf_pfx_list.append(ex["valid_pfx"])
        crf_sfx_list.append(ex["valid_sfx"])

    device = torch.device("cpu")
    return Batch(
        token_ids=token_ids,
        ngram_ids=ngram_ids,
        ngram_counts=ngram_counts,
        bloom_ids=bloom_ids,
        is_num=is_num,
        lengths=lengths,
        targets=targets,
        union_targets=union_tgt,
        desig_targets=desig_tgt,
        prefix_targets=prefix_tgt,
        suffix_targets=suffix_tgt,
        crf_dnum=build_field_tensors(crf_dnum_list, B, device),
        crf_pfx=build_field_tensors(crf_pfx_list, B, device),
        crf_sfx=build_field_tensors(crf_sfx_list, B, device),
    )


class ArcFaceDataModule(L.LightningDataModule):
    """Loads training_examples.json, builds vocab/prototypes, provides dataloaders."""

    def __init__(self, batch_size=256):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size

    def setup(self, stage=None):
        print("Loading data...")
        data, skipped, n_nofnum = self._load_data()
        print(
            f"Loaded {len(data) - n_nofnum} with f_num, "
            f"{n_nofnum} union-only ({skipped} skipped)"
        )

        self.train_data = [ex for ex in data if ex["split"] == "train"]
        self.val_data = [ex for ex in data if ex["split"] == "val"]
        self.test_data = [ex for ex in data if ex["split"] == "test"]

        self._build_fnum_mapping()
        self._build_vocab()
        self._build_field_vocabs()
        self._build_prototypes()
        self._encode_all()
        self._build_class_to_union()

        print(
            f"Classes: {self.n_train_classes}, Vocab: {len(self.vocab)}, "
            f"Unions: {len(self.field_vocabs['union_name'])}"
        )

    def _build_fnum_mapping(self):
        # Train classes must come first (0..n_train-1) so the model knows
        # which get learnable W_fnum embeddings vs frozen OOV prototypes.
        self.fnum_to_idx = {
            f: i
            for i, f in enumerate(
                sorted(
                    set(ex["f_num"] for ex in self.train_data if ex["f_num"] != -100)
                )
            )
        }
        self.n_train_classes = len(self.fnum_to_idx)

        # Add OOV classes from gazetteer
        with open(DATA_DIR / "gazetteer.json") as f:
            self.gazetteer_data = json.load(f)

        n_oov = 0
        for fnum_str in sorted(self.gazetteer_data, key=int):
            fn = int(fnum_str)
            if fn not in self.fnum_to_idx:
                self.fnum_to_idx[fn] = len(self.fnum_to_idx)
                n_oov += 1

        self.n_classes = len(self.fnum_to_idx)
        self.idx_to_fnum_map = {v: k for k, v in self.fnum_to_idx.items()}
        print(
            f"Train classes: {self.n_train_classes}, "
            f"OOV: {n_oov}, Total: {self.n_classes}"
        )

    def _build_vocab(self):
        counter = Counter()
        for ex in self.train_data:
            for tok in ex["tokens"]:
                counter[tok] += 1
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for tok, count in counter.most_common():
            if count >= 2:
                self.vocab[tok] = len(self.vocab)

    def _build_field_vocabs(self):
        self.field_vocabs = {}
        self.fnum_records = defaultdict(list)

        for ex in self.train_data:
            fn = ex["f_num"]
            if fn == -100 or not ex.get("union_name"):
                continue
            self.fnum_records[fn].append(ex["record"])

        for field in ["union_name", "desig_name", "prefix", "suffix"]:
            vals = sorted(
                set(
                    r[field]
                    for recs in self.fnum_records.values()
                    for r in recs
                    if r[field]
                )
            )
            self.field_vocabs[field] = {v: i + 1 for i, v in enumerate(vals)}
        self.field_sizes = {f: len(v) for f, v in self.field_vocabs.items()}

    def _encode_all(self):
        fva = {
            f: {v: idx - 1 for v, idx in self.field_vocabs[f].items()}
            for f in self.field_vocabs
        }
        for dataset in [self.train_data, self.val_data, self.test_data]:
            for ex in dataset:
                ex["token_ids"] = [self.vocab.get(tok, 1) for tok in ex["tokens"]]
                ex["is_num_f"] = [float(n) for n in ex["is_num"]]
                ex["target"] = self.fnum_to_idx.get(ex["f_num"], -100)
                ex["union_target"] = fva.get("union_name", {}).get(
                    ex.get("union_name", ""), -100
                )
                rec = ex.get("record", {})
                for field in ["desig_name", "prefix", "suffix"]:
                    val = rec.get(field, "")
                    ex[f"{field}_target"] = (
                        fva.get(field, {}).get(val, -100) if val else -100
                    )

    def _record_to_fields_and_hashes(self, rec):
        """Convert a record dict to (field_indices, bloom_hashes)."""
        fields = [
            self.field_vocabs[f].get(rec.get(f, ""), 0)
            for f in ["union_name", "desig_name", "prefix", "suffix"]
        ]
        dnum = rec.get("desig_num", 0)
        hashes = bloom_hash_ids(str(dnum)) if dnum else [0] * NUM_BLOOM_HASHES
        return fields, hashes

    def _deduped_proto_rows(self, class_idx, records):
        """Build deduplicated prototype rows for a class from its records."""
        rows = []
        seen = set()
        for rec in records:
            fields, hashes = self._record_to_fields_and_hashes(rec)
            key = (tuple(fields), tuple(hashes))
            if key not in seen:
                seen.add(key)
                rows.append((class_idx, fields, hashes))
        return rows

    def _build_prototypes(self):
        proto_rows = []

        # Training prototypes (from training data records)
        for i in range(self.n_train_classes):
            fn = self.idx_to_fnum_map[i]
            recs = self.fnum_records[fn]
            proto_rows.extend(self._deduped_proto_rows(i, recs))

        # OOV prototypes (from gazetteer)
        for fnum_str, gaz_records in sorted(
            self.gazetteer_data.items(), key=lambda x: int(x[0])
        ):
            fn = int(fnum_str)
            if fn in self.fnum_records:
                continue
            ci = self.fnum_to_idx[fn]
            proto_rows.extend(self._deduped_proto_rows(ci, gaz_records))

        print(f"Prototypes: {len(proto_rows)}")

        # Tensorize
        n_protos = len(proto_rows)
        field_map = torch.zeros(n_protos, 4, dtype=torch.long)
        desig_bloom_t = torch.zeros(n_protos, NUM_BLOOM_HASHES, dtype=torch.long)
        proto_to_class = torch.zeros(n_protos, dtype=torch.long)
        for p, (ci, fields, hashes) in enumerate(proto_rows):
            proto_to_class[p] = ci
            field_map[p] = torch.tensor(fields)
            desig_bloom_t[p] = torch.tensor(hashes)

        self.factored_info = {
            "field_sizes": self.field_sizes,
            "field_map": field_map,
            "desig_bloom": desig_bloom_t,
            "proto_to_class": proto_to_class,
        }

    def _build_class_to_union(self):
        self.class_to_union = torch.zeros(self.n_classes, dtype=torch.long)
        for i in range(self.n_classes):
            fn = self.idx_to_fnum_map[i]
            rec = self.fnum_records.get(fn, self.gazetteer_data.get(str(fn), [{}]))[0]
            un = rec.get("union_name", "")
            self.class_to_union[i] = max(
                self.field_vocabs["union_name"].get(un, 0) - 1, 0
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_batch,
            num_workers=0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=512,
            shuffle=False,
            collate_fn=_collate_batch,
            num_workers=0,
        )

    def _load_data(self):
        with open(DATA_DIR / "training_examples.json") as f:
            raw = json.load(f)
        raw = [ex for ex in raw if ex.get("source") != "synthetic"]

        data, skipped, n_nofnum = [], 0, 0
        for ex in raw:
            f_num = ex.get("f_num")
            has_fnum = f_num and f_num != -100
            if not has_fnum:
                union_name = ex.get("union_name")
                if not union_name or union_name == -100:
                    skipped += 1
                    continue
            elif not ex.get("records"):
                skipped += 1
                continue

            tokens, is_num, ngram_ids, ngram_counts, bloom_ids = tokenize_for_arcface(
                ex["query"]
            )
            if not tokens:
                skipped += 1
                continue

            raw_rec = ex["records"][0] if ex.get("records") else {}
            record = {}
            for f in ["union_name", "desig_name", "prefix", "suffix"]:
                v = raw_rec.get(f)
                record[f] = v if v and v not in (-100, 0) else ""
            dnum = raw_rec.get("desig_num")
            record["desig_num"] = int(dnum) if dnum and dnum not in (-100, 0) else 0
            valid_dnum, valid_pfx, valid_sfx = find_valid_positions(tokens, raw_rec)

            data.append(
                {
                    "tokens": tokens,
                    "is_num": is_num,
                    "length": len(tokens),
                    "f_num": int(f_num) if has_fnum else -100,
                    "split": ex["split"],
                    "source": ex.get("source"),
                    "union_name": ex.get("union_name"),
                    "record": record,
                    "ngram_ids": ngram_ids,
                    "ngram_counts": ngram_counts,
                    "bloom_ids": bloom_ids,
                    "valid_dnum": valid_dnum,
                    "valid_pfx": valid_pfx,
                    "valid_sfx": valid_sfx,
                }
            )
            if not has_fnum:
                n_nofnum += 1
        return data, skipped, n_nofnum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    L.seed_everything(42)

    dm = ArcFaceDataModule(batch_size=256)
    dm.setup()

    model = TrainingModel(
        n_classes=dm.n_classes,
        n_train_classes=dm.n_train_classes,
        vocab_size=len(dm.vocab),
        field_sizes=dm.field_sizes,
        class_to_union=dm.class_to_union,
    )
    model.classifier.field_map = dm.factored_info["field_map"]
    model.classifier.desig_bloom = dm.factored_info["desig_bloom"]
    model.classifier.proto_to_class = dm.factored_info["proto_to_class"]

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = L.Trainer(
        max_epochs=50,
        enable_progress_bar=True,
        enable_checkpointing=False,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_top1", patience=15, mode="max"
            ),
        ],
    )
    trainer.fit(model, dm)

    # Save checkpoint in our format
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state,
            "fnum_to_idx": dm.fnum_to_idx,
            "vocab": dm.vocab,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "n_classes": dm.n_classes,
            "n_train_classes": dm.n_train_classes,
            "n_buckets": N_BUCKETS,
            "arcface_scale": ARCFACE_SCALE,
            "field_vocabs": dm.field_vocabs,
            "field_sizes": dm.field_sizes,
            "field_map": dm.factored_info["field_map"],
            "desig_bloom": dm.factored_info["desig_bloom"],
            "proto_to_class": dm.factored_info["proto_to_class"],
            "idx_to_fnum": dm.idx_to_fnum_map,
            "union_vocab": {
                v: idx - 1 for v, idx in dm.field_vocabs["union_name"].items()
            },
            "n_unions": len(dm.field_vocabs["union_name"]),
        },
        str(DATA_DIR / "arcface_classifier.ckpt"),
    )
    print(f"Checkpoint saved to {DATA_DIR / 'arcface_classifier.ckpt'}")


if __name__ == "__main__":
    main()
