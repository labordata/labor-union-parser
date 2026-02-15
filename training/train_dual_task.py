"""
Dual-Task Training: Retrieval (Dual Tower) + Re-ranking (Cross-Attention Bridge)

Uses PyTorch Lightning. Two modes selectable at launch:
  --mode inbatch : in-batch negatives throughout (QueryRecordDataset)
  --mode ance    : ANCE hard-negative mining throughout (QueryRecordDatasetWithCandidates)
"""

import json
from pathlib import Path

import click
import lightning as L
import torch
from dataset import (
    QueryDataset,
    QueryRecordDataset,
    QueryRecordDatasetWithCandidates,
    encode_record_batch,
)
from mining import ANCEMiningCallback
from torch.utils.data import DataLoader

from labor_union_parser.model import DualTaskModel

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
DATA_DIR = Path(__file__).parent / "data"
VOCAB_PATH = DATA_DIR / "vocabularies.json"
EXAMPLES_PATH = DATA_DIR / "training_examples.json"
FNUM_TO_RECORDS_PATH = DATA_DIR / "fnum_to_records.json"

# =============================================================================
# Loss Functions
# =============================================================================


def supervised_contrastive_loss(similarities, fnum_ids, temperature=0.07):
    """
    Any-positive contrastive loss (NCA/InfoNCE style).

    Sums positive probabilities in the numerator so that ANY one positive
    ranking high satisfies the loss:
      -log(Σ_{p∈P_i} exp(s_ip/τ) / Σ_k exp(s_ik/τ))

    Identical to per-positive SupCon when |P_i|=1.
    """
    positive_mask = fnum_ids.unsqueeze(0) == fnum_ids.unsqueeze(1)
    sim_scaled = similarities / temperature

    neg_inf = torch.finfo(sim_scaled.dtype).min
    pos_sim = sim_scaled.masked_fill(~positive_mask, neg_inf)
    log_numerator = torch.logsumexp(pos_sim, dim=1)
    log_denominator = torch.logsumexp(sim_scaled, dim=1)
    loss = -(log_numerator - log_denominator)

    return loss.mean()


def contrastive_loss_with_mask(
    similarities, positive_mask, temperature=0.07, weights=None
):
    """
    Any-positive contrastive loss for per-query candidate lists.

    Sums positive probabilities in the numerator so that ANY one positive
    ranking high satisfies the loss:
      -log(Σ_{p∈P_i} exp(s_ip/τ) / Σ_k exp(s_ik/τ))

    Identical to per-positive SupCon when |P_i|=1.

    If weights is provided, computes a weighted mean instead of simple mean.
    """
    sim_scaled = similarities / temperature

    neg_inf = torch.finfo(sim_scaled.dtype).min
    pos_sim = sim_scaled.masked_fill(~positive_mask, neg_inf)
    log_numerator = torch.logsumexp(pos_sim, dim=1)
    log_denominator = torch.logsumexp(sim_scaled, dim=1)
    loss = -(log_numerator - log_denominator)

    if weights is not None:
        return (loss * weights).sum() / weights.sum()
    return loss.mean()


# =============================================================================
# Checkpoint Loading
# =============================================================================


def load_checkpoint_with_mismatch(model, path, device=None):
    """
    Load a checkpoint into model, handling embedding-size mismatches gracefully.

    Supports both Lightning checkpoints (.ckpt with "state_dict" key and
    "model." prefix) and plain torch checkpoints ("model_state_dict" key).

    When vocabulary sizes grow between runs, embedding layers may have more rows
    in the new model than the checkpoint. This copies old weights into the first
    N rows and keeps the new rows randomly initialized.
    """
    if device is None:
        device = next(model.parameters()).device
    ckpt = torch.load(path, map_location=device)

    # Lightning checkpoints store weights under "state_dict" with "model." prefix
    if "state_dict" in ckpt:
        ckpt_state = {
            k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
        }
    else:
        ckpt_state = ckpt["model_state_dict"]
    model_state = model.state_dict()

    for name, param in ckpt_state.items():
        if name in model_state:
            if param.shape == model_state[name].shape:
                model_state[name].copy_(param)
            elif (
                param.shape[0] < model_state[name].shape[0]
                and len(param.shape) == 2
                and param.shape[1] == model_state[name].shape[1]
            ):
                # Embedding grew - copy old weights to beginning
                model_state[name][: param.shape[0]].copy_(param)
                print(
                    f"  Expanded {name}: {param.shape[0]} -> {model_state[name].shape[0]}"
                )
            else:
                print(
                    f"  Skipped {name}: shape mismatch {param.shape} vs {model_state[name].shape}"
                )

    model.load_state_dict(model_state)
    print(f"  Loaded model weights from {path}")
    return ckpt


# =============================================================================
# Lightning Modules
# =============================================================================


class DualTaskLitBase(L.LightningModule):
    """Shared logic for validation, optimizer, and NaN handling."""

    def __init__(
        self,
        model,
        lr=1e-4,
        warmup_epochs=1,
        retrieval_weight=1.0,
        reranking_weight=1.0,
        temperature=0.07,
        weight_alpha=0.5,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        # Dynamic category weights (updated after each validation epoch)
        from dataset import NUM_CATEGORIES

        self._category_weights = torch.ones(NUM_CATEGORIES)
        self._train_category_fracs = None  # set by datamodule

    # ------------------------------------------------------------------
    # Full-corpus validation: encode all records, retrieve top-k, rerank
    # ------------------------------------------------------------------
    VAL_K = 50

    def on_validation_epoch_start(self):
        from dataset import NUM_CATEGORIES

        self._val_errors_by_cat = torch.zeros(NUM_CATEGORIES, dtype=torch.long)
        self._val_total_by_cat = torch.zeros(NUM_CATEGORIES, dtype=torch.long)

        dm = self.trainer.datamodule
        if self._train_category_fracs is None and hasattr(dm, "train_category_fracs"):
            self._train_category_fracs = dm.train_category_fracs
        # Encode all records in one pass (dual-tower + field embeddings)
        all_records = []
        record_fnums = []
        for fnum, records in dm.fnum_to_records.items():
            for rec in records:
                all_records.append(rec)
                record_fnums.append(fnum)

        all_record_embs = []
        field_embs, field_masks = [], []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(all_records), 512):
                rec_batch = encode_record_batch(all_records[i : i + 512], dm.vocab)
                rec_batch = {k: v.to(self.device) for k, v in rec_batch.items()}
                fe, fm = self.model.record_encoder(
                    rec_batch["union_idx"],
                    rec_batch["desig_idx"],
                    rec_batch["prefix_hash"],
                    rec_batch["num_hash"],
                    rec_batch["suffix_idx"],
                    rec_batch["unit_id_idx"],
                )
                emb = self.model.dual_tower.encode_record(fe)
                all_record_embs.append(emb)
                field_embs.append(fe)
                field_masks.append(fm)

        self._val_record_embs = torch.cat(all_record_embs).to(self.device)
        self._val_record_fnums = torch.tensor(record_fnums).to(self.device)
        self._val_field_embs = torch.cat(field_embs)
        self._val_field_masks = torch.cat(field_masks)

    def on_validation_epoch_end(self):
        # Compute dynamic category weights from validation errors
        if (
            hasattr(self, "_val_errors_by_cat")
            and self._train_category_fracs is not None
        ):
            total_errors = self._val_errors_by_cat.sum().item()
            if total_errors > 0:
                error_fracs = self._val_errors_by_cat.float() / total_errors
                alpha = self.hparams.get("weight_alpha", 0.5)
                for i in range(len(self._category_weights)):
                    train_frac = self._train_category_fracs[i]
                    if train_frac > 0 and error_fracs[i] > 0:
                        self._category_weights[i] = (
                            error_fracs[i].item() / train_frac
                        ) ** alpha
                    else:
                        self._category_weights[i] = 1.0
                # Log weights
                from dataset import CATEGORY_TO_IDX

                cat_names = {v: k for k, v in CATEGORY_TO_IDX.items()}
                parts = []
                for i in range(len(self._category_weights)):
                    name = cat_names.get(i, f"cat{i}")
                    parts.append(f"{name}={self._category_weights[i]:.2f}")
                err_parts = []
                for i in range(len(self._category_weights)):
                    name = cat_names.get(i, f"cat{i}")
                    err_parts.append(
                        f"{name}={self._val_errors_by_cat[i]}/{self._val_total_by_cat[i]}"
                    )
                print(f"  Errors by cat: {', '.join(err_parts)}")
                print(f"  Category weights: {', '.join(parts)}")

        for attr in (
            "_val_record_embs",
            "_val_record_fnums",
            "_val_field_embs",
            "_val_field_masks",
            "_val_errors_by_cat",
            "_val_total_by_cat",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _eval_step(self, batch, prefix):
        f_nums = batch["f_num"].to(self.device)
        categories = batch["category"].to(self.device)
        n = f_nums.shape[0]

        token_emb, padding_mask = self.model.query_encoder(
            batch["char_ids"], batch["is_number"], batch["numeric_ids"]
        )
        query_embs = self.model.dual_tower.encode_query(token_emb, padding_mask)

        # Retrieval: top-K candidates per query [n, VAL_K]
        sims = query_embs @ self._val_record_embs.T
        topk_vals, topk_idx = sims.topk(self.VAL_K, dim=1)
        top1_fnums = self._val_record_fnums[sims.argmax(dim=1)]

        ret1_correct = (top1_fnums == f_nums).sum().item()

        # Check which queries have their target in top-K
        cand_fnums = self._val_record_fnums[topk_idx]  # [n, VAL_K]
        retrieved = (cand_fnums == f_nums.unsqueeze(1)).any(dim=1).cpu()  # [n]
        retk_correct = retrieved.sum().item()

        # Category tracking: totals for all, errors for retrieval misses
        cats_cpu = categories.cpu()
        self._val_total_by_cat.scatter_add_(
            0, cats_cpu, torch.ones_like(cats_cpu, dtype=torch.long)
        )
        miss_cats = cats_cpu[~retrieved]
        if miss_cats.numel() > 0:
            self._val_errors_by_cat.scatter_add_(
                0, miss_cats, torch.ones_like(miss_cats, dtype=torch.long)
            )

        # Reranking: batch all retrieved queries at once
        acc_correct = 0
        if retk_correct > 0:
            ret_idx = retrieved.nonzero(as_tuple=True)[
                0
            ]  # indices of retrieved queries
            n_ret = ret_idx.shape[0]

            # Gather candidates for retrieved queries [n_ret, VAL_K]
            ret_topk = topk_idx[ret_idx]
            ret_cand_fnums = cand_fnums[ret_idx]
            ret_targets = f_nums[ret_idx]

            # Batch cross-attention: [n_ret * VAL_K, ...]
            cand_fe = self._val_field_embs[ret_topk.reshape(-1)]
            cand_fm = self._val_field_masks[ret_topk.reshape(-1)]
            q_tok = (
                token_emb[ret_idx]
                .unsqueeze(1)
                .expand(n_ret, self.VAL_K, -1, -1)
                .reshape(n_ret * self.VAL_K, token_emb.shape[1], -1)
            )
            q_mask = (
                padding_mask[ret_idx]
                .unsqueeze(1)
                .expand(n_ret, self.VAL_K, -1)
                .reshape(n_ret * self.VAL_K, -1)
            )
            rerank_scores = self.model.cross_attention.score_pair(
                q_tok, q_mask, cand_fe, cand_fm
            ).view(n_ret, self.VAL_K)

            # For each query, check if target fnum has the highest max score
            # Set scores for target-fnum candidates to -inf, find max non-target
            is_target = ret_cand_fnums == ret_targets.unsqueeze(1)  # [n_ret, VAL_K]
            neg_inf = torch.finfo(rerank_scores.dtype).min

            # Max score among target-fnum candidates
            target_scores = rerank_scores.masked_fill(~is_target, neg_inf)
            target_max = target_scores.max(dim=1).values  # [n_ret]

            # Max score among non-target candidates
            other_scores = rerank_scores.masked_fill(is_target, neg_inf)
            other_max = other_scores.max(dim=1).values  # [n_ret]

            rerank_correct = (target_max > other_max).cpu()  # [n_ret]
            acc_correct = rerank_correct.sum().item()

            # Category errors for reranking misses
            rerank_miss_cats = cats_cpu[ret_idx[~rerank_correct]]
            if rerank_miss_cats.numel() > 0:
                self._val_errors_by_cat.scatter_add_(
                    0,
                    rerank_miss_cats,
                    torch.ones_like(rerank_miss_cats, dtype=torch.long),
                )

        rer_acc = acc_correct / max(retk_correct, 1)
        self.log(f"{prefix}_ret1", ret1_correct / n, batch_size=n, prog_bar=True)
        self.log(f"{prefix}_ret", retk_correct / n, batch_size=n, prog_bar=True)
        self.log(f"{prefix}_rer", rer_acc, batch_size=n, prog_bar=True)
        self.log(f"{prefix}_acc", acc_correct / n, batch_size=n, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "v")

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "t")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01
        )
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // self.trainer.max_epochs
        warmup_steps = steps_per_epoch * self.hparams.warmup_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(
                0.1, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _compute_loss(self, retrieval_loss, rerank_loss):
        loss = (
            self.hparams.retrieval_weight * retrieval_loss
            + self.hparams.reranking_weight * rerank_loss
        )
        self.log("ret", retrieval_loss, prog_bar=True)
        self.log("rer", rerank_loss, prog_bar=True)
        return loss


class InBatchLitModule(DualTaskLitBase):
    """In-batch negatives: uses forward_dual_task for N x N similarities."""

    def training_step(self, batch, batch_idx):
        retrieval_sim, rerank_scores = self.model.forward_dual_task(
            batch["char_ids"],
            batch["is_number"],
            batch["numeric_ids"],
            batch["union_idx"],
            batch["desig_idx"],
            batch["prefix_hash"],
            batch["num_hash"],
            batch["suffix_idx"],
            batch["unit_id_idx"],
        )
        retrieval_loss = supervised_contrastive_loss(
            retrieval_sim, batch["f_num"], self.hparams.temperature
        )
        rerank_loss = supervised_contrastive_loss(
            rerank_scores, batch["f_num"], temperature=1.0
        )
        return self._compute_loss(retrieval_loss, rerank_loss)


class ANCELitModule(DualTaskLitBase):
    """ANCE mode: each query vs its own K+1 mined candidates."""

    def training_step(self, batch, batch_idx):
        N = batch["char_ids"].shape[0]
        K_plus_1 = batch["cand_union_idx"].shape[1]

        # Per-example weights from category
        weights = self._category_weights[batch["category"].cpu()].to(self.device)

        # Encode queries
        token_emb, padding_mask = self.model.query_encoder(
            batch["char_ids"], batch["is_number"], batch["numeric_ids"]
        )
        query_emb = self.model.dual_tower.encode_query(token_emb, padding_mask)

        # Encode candidates [N*(K+1)] -> [N, K+1, 128]
        field_emb_flat, field_mask_flat = self.model.record_encoder(
            batch["cand_union_idx"].reshape(-1),
            batch["cand_desig_idx"].reshape(-1),
            batch["cand_prefix_hash"].reshape(-1),
            batch["cand_num_hash"].reshape(-1),
            batch["cand_suffix_idx"].reshape(-1),
            batch["cand_unit_id_idx"].reshape(-1),
        )
        cand_emb = self.model.dual_tower.encode_record(field_emb_flat).view(
            N, K_plus_1, -1
        )

        # Retrieval loss
        retrieval_sim = torch.einsum("nd,nkd->nk", query_emb, cand_emb)
        positive_mask = batch["f_num"].unsqueeze(1) == batch["cand_fnums"]
        retrieval_loss = contrastive_loss_with_mask(
            retrieval_sim, positive_mask, self.hparams.temperature, weights=weights
        )

        # Reranking loss
        seq_len = token_emb.shape[1]
        q_exp = (
            token_emb.unsqueeze(1)
            .expand(N, K_plus_1, seq_len, -1)
            .reshape(N * K_plus_1, seq_len, -1)
        )
        q_mask_exp = (
            padding_mask.unsqueeze(1)
            .expand(N, K_plus_1, seq_len)
            .reshape(N * K_plus_1, seq_len)
        )
        r_flat = field_emb_flat.view(N, K_plus_1, 6, -1).reshape(N * K_plus_1, 6, -1)
        r_mask_flat = field_mask_flat.view(N, K_plus_1, 6).reshape(N * K_plus_1, 6)

        rerank_scores = self.model.cross_attention.score_pair(
            q_exp, q_mask_exp, r_flat, r_mask_flat
        ).view(N, K_plus_1)

        rerank_loss = contrastive_loss_with_mask(
            rerank_scores, positive_mask, temperature=1.0, weights=weights
        )
        return self._compute_loss(retrieval_loss, rerank_loss)


# =============================================================================
# Data Module
# =============================================================================


class DualTaskDataModule(L.LightningDataModule):
    def __init__(
        self, vocab_path, examples_path, fnum_to_records_path, batch_size, mode
    ):
        super().__init__()
        self.vocab_path = vocab_path
        self.examples_path = examples_path
        self.fnum_to_records_path = fnum_to_records_path
        self.batch_size = batch_size
        self.mode = mode  # "inbatch" or "ance"

        # Populated by setup()
        self.vocab = None
        self.train_ex = None
        self.val_ex = None
        self.test_ex = None
        self.fnum_to_records = None

    def setup(self, stage=None):
        if self.vocab is not None:
            return  # already set up

        with open(self.vocab_path) as f:
            self.vocab = json.load(f)

        with open(self.examples_path) as f:
            all_examples = json.load(f)

        if self.mode == "structural":
            self.train_ex = [
                ex
                for ex in all_examples
                if ex["split"] == "train" and ex.get("structural_negatives")
            ]
        else:
            self.train_ex = [ex for ex in all_examples if ex["split"] == "train"]
        self.val_ex = [ex for ex in all_examples if ex["split"] == "val"]
        self.test_ex = [ex for ex in all_examples if ex["split"] == "test"]

        print(
            f"  Data: {len(self.train_ex)} train, {len(self.val_ex)} val, "
            f"{len(self.test_ex)} test"
        )

        # Full gazetteer for ANCE mining
        with open(self.fnum_to_records_path) as f:
            raw = json.load(f)
        self.fnum_to_records = {int(k): v for k, v in raw.items()}

        total_records = sum(len(recs) for recs in self.fnum_to_records.values())
        print(
            f"  {len(self.fnum_to_records)} unique f_nums, "
            f"{total_records} records (full gazetteer)"
        )

        # Compute training category fractions for dynamic weighting
        from dataset import CATEGORY_TO_IDX, NUM_CATEGORIES

        cat_counts = torch.zeros(NUM_CATEGORIES)
        for ex in self.train_ex:
            cat_idx = CATEGORY_TO_IDX.get(ex.get("category", "unique"), 0)
            cat_counts[cat_idx] += 1
        self.train_category_fracs = cat_counts / cat_counts.sum()

    def _make_inbatch_dataset(self, examples):
        return QueryRecordDataset(
            examples,
            self.vocab["union_name_to_idx"],
            self.vocab["desig_name_to_idx"],
            self.vocab["suffix_to_idx"],
            self.vocab["unit_id_to_idx"],
        )

    def train_dataloader(self):
        if self.mode in ("ance", "structural"):
            ds = QueryRecordDatasetWithCandidates(self.train_ex, self.vocab)
        else:
            ds = self._make_inbatch_dataset(self.train_ex)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=ds.collate,
            drop_last=True,
        )

    def val_dataloader(self):
        ds = QueryDataset(self.val_ex)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ds.collate,
            drop_last=False,
        )

    def test_dataloader(self):
        ds = QueryDataset(self.test_ex)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ds.collate,
            drop_last=False,
        )


# =============================================================================
# MPS Workaround
# =============================================================================


class DeviceSyncCallback(L.Callback):
    """Force GPU synchronization after eval-mode forward passes.

    On Apple Silicon (MPS), Lightning's sanity-check validation and ANCE mining
    run eval-mode forward passes whose results may still be in-flight when
    training starts, causing NaN.  A blocking device-to-host transfer flushes
    the compute stream on any accelerator.
    """

    @staticmethod
    def _sync(pl_module):
        torch.zeros(1, device=pl_module.device).item()

    def on_sanity_check_end(self, trainer, pl_module):
        self._sync(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync(pl_module)


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option("--epochs", default=40, help="Number of epochs to train")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option(
    "--checkpoint", type=click.Path(exists=True), help="Resume from checkpoint"
)
@click.option(
    "--mode",
    type=click.Choice(["inbatch", "ance", "structural"]),
    default="inbatch",
    help="Training mode (structural = ance dataset with no mining)",
)
@click.option(
    "--mine-every", default=5, help="Re-mine interval in epochs (ance mode only)"
)
@click.option("--mine-k", default=50, help="Candidates per query (ance mode only)")
@click.option("--warmup-epochs", default=1, help="LR warmup epochs")
@click.option(
    "--weight-alpha", default=0.5, help="Dampening exponent for category weights"
)
@click.option(
    "--pool-mode",
    type=click.Choice(["cls", "mean"]),
    default="cls",
    help="Cross-attention pooling mode",
)
@click.option(
    "--retrieval-weight", default=1.0, help="Weight for retrieval loss (0 to disable)"
)
@click.option("--ca-heads", default=4, help="Number of cross-attention heads")
def train(
    epochs,
    batch_size,
    lr,
    checkpoint,
    mode,
    mine_every,
    mine_k,
    warmup_epochs,
    weight_alpha,
    pool_mode,
    retrieval_weight,
    ca_heads,
):
    print(f"Mode: {mode}, Pool: {pool_mode}")

    # Data
    datamodule = DualTaskDataModule(
        VOCAB_PATH, EXAMPLES_PATH, FNUM_TO_RECORDS_PATH, batch_size, mode
    )
    datamodule.setup()

    vocab = datamodule.vocab

    # Model
    model = DualTaskModel(
        num_union_names=len(vocab["union_name_to_idx"]),
        num_desig_names=len(vocab["desig_name_to_idx"]),
        num_suffixes=len(vocab["suffix_to_idx"]),
        num_unit_ids=len(vocab["unit_id_to_idx"]),
        pool_mode=pool_mode,
        ca_num_heads=ca_heads,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if checkpoint:
        load_checkpoint_with_mismatch(model, checkpoint)

    # Lightning module
    common_kwargs = dict(
        model=model,
        lr=lr,
        warmup_epochs=warmup_epochs,
        retrieval_weight=retrieval_weight,
        reranking_weight=1.0,
        temperature=0.07,
        weight_alpha=weight_alpha,
    )

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath="training",
            filename="dual_task_model",
            monitor="v_acc",
            mode="max",
            save_top_k=1,
        ),
        DeviceSyncCallback(),
    ]

    if mode == "inbatch":
        lit_module = InBatchLitModule(**common_kwargs)
        reload_every = 0
    elif mode == "structural":
        # ANCE dataset (reads structural_negatives) but no mining
        lit_module = ANCELitModule(**common_kwargs)
        reload_every = 0
    else:
        lit_module = ANCELitModule(**common_kwargs)
        callbacks.append(ANCEMiningCallback(mine_every, mine_k))
        reload_every = mine_every

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=reload_every,
        callbacks=callbacks,
        enable_checkpointing=True,
        limit_val_batches=0 if mode == "structural" else None,
        limit_test_batches=0 if mode == "structural" else None,
    )

    trainer.fit(lit_module, datamodule=datamodule)
    if mode == "structural":
        trainer.save_checkpoint("training/dual_task_model.ckpt")
    else:
        trainer.test(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    train()
