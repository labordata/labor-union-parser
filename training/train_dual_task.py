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
    QueryRecordDataset,
    QueryRecordDatasetWithCandidates,
)
from mining import (
    ANCEMiningCallback,
)
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
    Supervised contrastive loss: maximize the combined softmax mass on all
    positives (same f_num).  -log(Σ_p exp(s_p) / Σ_k exp(s_k))

    Only requires *any* correct record to be retrievable, rather than forcing
    every variant to be individually retrievable.
    """
    positive_mask = fnum_ids.unsqueeze(0) == fnum_ids.unsqueeze(1)
    sim_scaled = similarities / temperature

    # logsumexp over positives - logsumexp over all = log(positive_mass / total)
    neg_inf = torch.finfo(sim_scaled.dtype).min
    pos_sim = sim_scaled.masked_fill(~positive_mask, neg_inf)
    loss = -torch.logsumexp(pos_sim, dim=1) + torch.logsumexp(sim_scaled, dim=1)

    return loss.mean()


def contrastive_loss_with_mask(similarities, positive_mask, temperature=0.07):
    """
    Contrastive loss for per-query candidate lists: maximize the combined
    softmax mass on all positives.  -log(Σ_p exp(s_p) / Σ_k exp(s_k))
    """
    sim_scaled = similarities / temperature

    neg_inf = torch.finfo(sim_scaled.dtype).min
    pos_sim = sim_scaled.masked_fill(~positive_mask, neg_inf)
    loss = -torch.logsumexp(pos_sim, dim=1) + torch.logsumexp(sim_scaled, dim=1)

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
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def _eval_step(self, batch, prefix):
        fnum_ids = batch["f_num"]
        retrieval_sim, rerank_scores = self.model.forward_dual_task(
            batch["char_ids"],
            batch["is_number"],
            batch["numeric_ids"],
            batch["union_idx"],
            batch["desig_idx"],
            batch["prefix_idx"],
            batch["num_hash"],
            batch["num_val"],
            batch["suffix_idx"],
            batch["unit_id_idx"],
        )

        n = fnum_ids.shape[0]
        ret_acc = (fnum_ids[retrieval_sim.argmax(1)] == fnum_ids).float().mean()
        rer_acc = (fnum_ids[rerank_scores.argmax(1)] == fnum_ids).float().mean()
        avg_acc = (ret_acc + rer_acc) / 2

        self.log(f"{prefix}_ret", ret_acc, batch_size=n, prog_bar=True)
        self.log(f"{prefix}_rer", rer_acc, batch_size=n, prog_bar=True)
        self.log(f"{prefix}_avg", avg_acc, batch_size=n)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "v")

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
            batch["prefix_idx"],
            batch["num_hash"],
            batch["num_val"],
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

        # Encode queries
        token_emb, padding_mask = self.model.query_encoder(
            batch["char_ids"], batch["is_number"], batch["numeric_ids"]
        )
        query_emb = self.model.dual_tower.encode_query(token_emb, padding_mask)

        # Encode candidates [N*(K+1)] -> [N, K+1, 128]
        field_emb_flat, field_mask_flat = self.model.record_encoder(
            batch["cand_union_idx"].reshape(-1),
            batch["cand_desig_idx"].reshape(-1),
            batch["cand_prefix_idx"].reshape(-1),
            batch["cand_num_hash"].reshape(-1),
            batch["cand_num_val"].reshape(-1),
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
            retrieval_sim, positive_mask, self.hparams.temperature
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
            rerank_scores, positive_mask, temperature=1.0
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

    def _make_inbatch_dataset(self, examples):
        return QueryRecordDataset(
            examples,
            self.vocab["union_name_to_idx"],
            self.vocab["desig_name_to_idx"],
            self.vocab["prefix_to_idx"],
            self.vocab["suffix_to_idx"],
            self.vocab["unit_id_to_idx"],
        )

    def train_dataloader(self):
        if self.mode == "ance":
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
        ds = self._make_inbatch_dataset(self.val_ex)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=ds.collate,
            drop_last=False,
        )

    def test_dataloader(self):
        ds = self._make_inbatch_dataset(self.test_ex)
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
    type=click.Choice(["inbatch", "ance"]),
    default="inbatch",
    help="Training mode",
)
@click.option(
    "--mine-every", default=5, help="Re-mine interval in epochs (ance mode only)"
)
@click.option("--mine-k", default=50, help="Candidates per query (ance mode only)")
@click.option("--warmup-epochs", default=1, help="LR warmup epochs")
def train(epochs, batch_size, lr, checkpoint, mode, mine_every, mine_k, warmup_epochs):
    print(f"Mode: {mode}")

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
        num_prefixes=len(vocab["prefix_to_idx"]),
        num_suffixes=len(vocab["suffix_to_idx"]),
        num_unit_ids=len(vocab["unit_id_to_idx"]),
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if checkpoint:
        load_checkpoint_with_mismatch(model, checkpoint)

    # Lightning module
    common_kwargs = dict(
        model=model,
        lr=lr,
        warmup_epochs=warmup_epochs,
        retrieval_weight=1.0,
        reranking_weight=1.0,
        temperature=0.07,
    )

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath="training",
            filename="dual_task_model",
            monitor="v_avg",
            mode="max",
            save_top_k=1,
        ),
        DeviceSyncCallback(),
    ]

    if mode == "inbatch":
        lit_module = InBatchLitModule(**common_kwargs)
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
    )

    trainer.fit(lit_module, datamodule=datamodule)
    trainer.test(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    train()
