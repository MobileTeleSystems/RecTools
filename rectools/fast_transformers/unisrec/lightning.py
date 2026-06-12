#  Copyright 2026 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Lightning wrapper for UniSRec with configurable loss, optimizer, scheduler."""

import math
import typing as tp

import pytorch_lightning as pl
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim.lr_scheduler import LambdaLR

from .net import UniSRecNet

SUPPORTED_LOSSES = ("softmax", "BCE", "gBCE", "sampled_softmax")
SUPPORTED_OPTIMIZERS = ("adam", "adamw")
SUPPORTED_SCHEDULERS = (None, "cosine_warmup")


class UniSRecLightning(pl.LightningModule):
    """
    Thin Lightning wrapper for joint UniSRec training.

    Wraps a :class:`UniSRecNet` network with configurable loss, optimizer,
    and learning-rate scheduler.
    """

    def __init__(
        self,
        net: UniSRecNet,
        param_groups: tp.List[tp.Dict[str, tp.Any]],
        loss: str = "softmax",
        n_negatives: tp.Optional[int] = None,
        gbce_t: float = 0.2,
        optimizer: str = "adamw",
        scheduler: tp.Optional[str] = None,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        total_steps: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.net = net
        self._param_groups = param_groups
        self.loss_name = loss
        self.n_negatives = n_negatives
        self.gbce_t = gbce_t
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = total_steps

    # ── helpers ──

    def _get_item_embs(self, item_ids: torch.Tensor) -> torch.Tensor:
        # pylint: disable=protected-access
        return self.net._adapt_score(self.net._sample_frozen(item_ids))

    def _get_all_embs(self) -> torch.Tensor:
        return self.net.project_all()

    def _get_pos_neg_logits(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (B, L, 1+N) logits where index 0 = positive."""
        emb_pos = self._get_item_embs(labels)
        logits_pos = (hidden * emb_pos).sum(dim=-1)

        emb_neg = self._get_item_embs(negatives)
        logits_neg = torch.matmul(
            hidden.unsqueeze(2),
            emb_neg.transpose(2, 3),
        ).squeeze(2)

        return torch.cat([logits_pos.unsqueeze(-1), logits_neg], dim=-1)

    # ── losses ──

    def _calc_loss(
        self,
        hidden: torch.Tensor,
        batch: tp.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        labels = batch["y"]
        has_neg = "negatives" in batch

        if self.loss_name == "softmax":
            return self._full_softmax_loss(hidden, labels)

        if not has_neg:
            raise ValueError(f"Loss '{self.loss_name}' requires negatives but batch has none")

        logits = self._get_pos_neg_logits(hidden, labels, batch["negatives"])
        mask = labels != 0

        if self.loss_name == "sampled_softmax":
            return self._sampled_softmax_loss(logits, mask)
        if self.loss_name == "BCE":
            return self._bce_loss(logits, mask)
        if self.loss_name == "gBCE":
            return self._gbce_loss(logits, mask)

        raise ValueError(f"Unknown loss: {self.loss_name}")

    def _full_softmax_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # NOTE: project_all() always uses variant 0 of frozen embeddings
        # (deterministic), whereas BCE/gBCE/sampled_softmax score against
        # _get_item_embs() which randomly samples variants during training.
        # This means softmax never sees variant augmentation.
        all_emb = self._get_all_embs()
        logits = hidden @ all_emb.T
        logits[:, :, 0] = float("-inf")

        targets = labels.clone()
        targets[targets == 0] = -100
        return cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

    def _sampled_softmax_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute sampled softmax by swapping positive to index 1."""
        logits = logits.clone()
        logits[:, :, [0, 1]] = logits[:, :, [1, 0]]
        targets = mask.long()  # 1 where non-padding, 0 where padding
        return cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=0,
        )

    def _bce_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = torch.zeros_like(logits)
        target[:, :, 0] = 1.0
        loss = binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = loss.mean(-1) * mask
        return loss.sum() / mask.sum().clamp(min=1)

    def _gbce_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_items = self.net.n_items
        n_neg = self.n_negatives or logits.size(-1) - 1
        alpha = n_neg / max(n_items - 1, 1)
        beta = alpha * (self.gbce_t * (1 - 1 / alpha) + 1 / alpha)

        dtype = torch.float64
        pos_logits = logits[:, :, 0:1].to(dtype)
        neg_logits = logits[:, :, 1:]

        eps = 1e-10
        pos_probs = torch.clamp(torch.sigmoid(pos_logits), eps, 1 - eps)
        pos_adjusted = torch.clamp(pos_probs.pow(-beta), 1 + eps, torch.finfo(dtype).max)
        pos_adjusted = torch.clamp(1.0 / (pos_adjusted - 1), eps, torch.finfo(dtype).max)
        pos_transformed = torch.log(pos_adjusted).to(logits.dtype)

        adjusted_logits = torch.cat([pos_transformed, neg_logits], dim=-1)
        return self._bce_loss(adjusted_logits, mask)

    # ── training / validation ──

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single training step and log the loss."""
        hidden = self.net(batch["x"])
        loss = self._calc_loss(hidden, batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single validation step on the last sequence position."""
        hidden = self.net(batch["x"])
        # Validation batch has y of shape (B, 1) -- take last hidden position only
        hidden = hidden[:, -1:, :]
        loss = self._calc_loss(hidden, batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ── optimizer / scheduler ──

    def configure_optimizers(self) -> tp.Any:
        """Build optimizer and optional LR scheduler."""
        opt: torch.optim.Optimizer
        if self.optimizer_name == "adamw":
            opt = torch.optim.AdamW(self._param_groups)
        elif self.optimizer_name == "adam":
            opt = torch.optim.Adam(self._param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.scheduler_name is None:
            return opt

        if self.scheduler_name == "cosine_warmup":
            total = self.total_steps or 1
            warmup_steps = int(total * self.warmup_ratio)
            scheduler = _cosine_warmup_scheduler(opt, warmup_steps, total, self.min_lr_ratio)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        raise ValueError(f"Unknown scheduler: {self.scheduler_name}")


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
