import typing as tp

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from rectools.semantic import LRScheduleType, OptimizerType
from rectools.semantic.metrics import coverage_k, gini_k
from rectools.semantic.tokenizer import SIDTokenizer

from .loss import compute_tiger_loss
from .module import TIGERNet


class TIGERLightning(pl.LightningModule):
    """A ``pytorch_lightninig` wrapper around TIGERNet which used used for
    training, validation and testing of the model.

    Parameters
    ----------
    model : TIGERNet
        Model to train
    tokenizer : SIDTokenizer
        Semantic ID tokenizer
    optimizer_name : OptimizerType, optional
        Which optimizer to use, by default "adamw"
    beam_size : int, optional
        Beam size for model inference, by default 20
    top_k : int, optional
        Used for @k metrics like HR@k or MRR@k, by default 10
    lr : float, optional
        Model learning rate, by default 5e-4
    weight_decay : float, optional
        Optimizer weight decay, by default 1e-4
    warmup_steps : int | float, optional
        Learning rate warm-up steps, by default 300
    lr_schedule : tp.Optional[LRScheduleType], optional
        Learning rate schedule to use. Constant LR if set to None, by default None
    max_iters : int, optional
        Maximum number of iterations for training, by default 1
    num_items : tp.Optional[int], optional
        Number of unique items, by default None
    """

    def __init__(
        self,
        model: TIGERNet,
        tokenizer: SIDTokenizer,
        optimizer_name: OptimizerType = "adamw",
        beam_size: int = 20,
        top_k: int = 10,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int | float = 300,
        lr_schedule: tp.Optional[LRScheduleType] = None,
        max_iters: int = 1,
        num_items: tp.Optional[int] = None,
        decode_rng: tp.Optional[np.random.RandomState] = None,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.top_k = top_k
        self.num_items = num_items

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weight_decay = weight_decay
        self.max_iters = max_iters
        self.decode_rng = decode_rng

        self._all_topk_items: tp.List[tp.List[int]] = []

        if isinstance(warmup_steps, float) and 0 < warmup_steps < 1:
            self.warmup_steps = round(max_iters * warmup_steps)
        else:
            self.warmup_steps = int(warmup_steps)

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        input_ids = batch["input_ids"]
        dec_input = batch["dec_input"]
        labels = batch["labels"]

        enc_padding_mask = input_ids == TIGERNet.PAD_TOKEN_ID

        logits = self.model(input_ids, dec_input, enc_padding_mask)
        loss = compute_tiger_loss(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _valtest_step(  # pylint: disable=too-many-locals
        self, batch: tp.Dict[str, torch.Tensor], batch_idx: int, mode: tp.Literal["val", "test"]
    ) -> STEP_OUTPUT:
        prefix = "" if mode == "test" else "val_"
        input_sids = batch["input_ids"]
        labels = batch["labels"].cpu().tolist()

        metrics = {}
        all_topk_items = []

        enc_padding_mask = input_sids == TIGERNet.PAD_TOKEN_ID

        codes, _scores = self.model.generate(input_sids, enc_padding_mask=enc_padding_mask, beam_size=self.beam_size)
        batch_size, num_beams = codes.size(0), codes.size(1)

        # Batch-decode: flatten (batch, beam, depth) -> list of tuples, decode once
        codes_cpu = codes.cpu().tolist()
        all_sids = [tuple(codes_cpu[i][j]) for i in range(batch_size) for j in range(num_beams)]
        all_decoded: tp.List[tp.Optional[int]] = self.tokenizer.decode(  # type: ignore[assignment]
            all_sids,
            rng=self.decode_rng,
        )

        # Reshape decoded items back to (batch, beam)
        decoded_grid = [all_decoded[i * num_beams : (i + 1) * num_beams] for i in range(batch_size)]

        # Per-row deduplication + rank finding (small Python loop)
        ranks = np.zeros(batch_size, dtype=np.int64)
        valid_sids = 0
        for i in range(batch_size):
            row = decoded_grid[i]
            target = labels[i]
            seen = set()
            unique_items = []
            rank_pos = 0
            found = False
            for item_id in row:
                if item_id is not None and item_id not in seen:
                    valid_sids += 1
                    rank_pos += 1
                    seen.add(item_id)
                    unique_items.append(item_id)
                    if not found and item_id == target:
                        ranks[i] = rank_pos
                        found = True
            all_topk_items.append(unique_items)

        self._all_topk_items.extend(all_topk_items)

        # Vectorized metric computation
        in_top_k = (ranks > 0) & (ranks <= self.top_k)
        top_k_ranks = ranks[in_top_k]

        hit_sum = in_top_k.sum()

        metrics[f"{prefix}Hit@{self.top_k}"] = hit_sum / batch_size
        metrics[f"{prefix}NDCG@{self.top_k}"] = (
            float((1.0 / np.log2(top_k_ranks + 1)).sum()) if hit_sum > 0 else 0.0
        ) / batch_size
        metrics[f"{prefix}MRR@{self.top_k}"] = (float((1.0 / top_k_ranks).sum()) if hit_sum > 0 else 0.0) / batch_size
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return metrics

    def _on_valtest_epoch_end(self, mode: tp.Literal["val", "test"]) -> STEP_OUTPUT:
        prefix = "" if mode == "test" else "val_"
        metrics = {}
        metrics[f"{prefix}Gini@{self.top_k}"] = gini_k(self._all_topk_items, self.top_k)
        if self.num_items is not None:
            metrics[f"{prefix}Coverage@{self.top_k}"] = coverage_k(self._all_topk_items, self.top_k, self.num_items)
        self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=True)
        return metrics

    def on_validation_epoch_start(self) -> None:
        self._all_topk_items = []

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        return self._valtest_step(batch, batch_idx, mode="val")

    def on_validation_epoch_end(self) -> None:
        self._on_valtest_epoch_end(mode="val")

    def on_test_epoch_start(self) -> None:
        self._all_topk_items = []

    def test_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        return self._valtest_step(batch, batch_idx, mode="test")

    def on_test_epoch_end(self) -> None:
        self._on_valtest_epoch_end(mode="test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer: torch.optim.Optimizer
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adagrad":
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.lr_schedule is None:
            return optimizer

        schedulers: tp.List[torch.optim.lr_scheduler.LRScheduler] = []
        milestones: tp.List[int] = []
        if self.warmup_steps > 0:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_steps)
            )
            milestones = [self.warmup_steps]
        if self.lr_schedule == "cosine":
            t_0 = max(1, self.max_iters)
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=t_0,
                )
            )
        elif self.lr_schedule == "linear":
            schedulers.append(torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0))
        else:
            schedulers.append(
                torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=1.0,
                )
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers,
            milestones=milestones,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
