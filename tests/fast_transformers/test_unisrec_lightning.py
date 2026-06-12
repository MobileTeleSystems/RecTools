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

"""Tests for UniSRecLightning wrapper and _cosine_warmup_scheduler."""

# pylint: disable=redefined-outer-name,protected-access

import math
import typing as tp

import pytest
import torch

from rectools.fast_transformers.unisrec.lightning import (
    SUPPORTED_LOSSES,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    UniSRecLightning,
    _cosine_warmup_scheduler,
)
from rectools.fast_transformers.unisrec.net import UniSRecNet


@pytest.fixture()
def pretrained_emb() -> torch.Tensor:
    """Fake pretrained embeddings: (11, 32) -- 10 items + 1 padding."""
    torch.manual_seed(0)
    emb = torch.randn(11, 32)
    emb[0] = 0.0  # padding
    return emb


@pytest.fixture()
def net(pretrained_emb: torch.Tensor) -> UniSRecNet:
    return UniSRecNet(
        n_items=10,
        pretrained_embeddings=pretrained_emb,
        n_factors=8,
        projection_hidden=16,
        n_blocks=1,
        n_heads=1,
        session_max_len=5,
        dropout=0.0,
        adaptor_dropout=0.0,
    )


def _make_module(
    net: UniSRecNet,
    loss: str = "softmax",
    n_negatives: tp.Optional[int] = None,
    optimizer: str = "adamw",
    scheduler: tp.Optional[str] = None,
    total_steps: tp.Optional[int] = None,
    lr: float = 1e-3,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.1,
    gbce_t: float = 0.2,
) -> UniSRecLightning:
    """Build a UniSRecLightning with a single param group."""
    param_groups = [{"params": list(net.parameters()), "lr": lr}]
    return UniSRecLightning(
        net=net,
        param_groups=param_groups,
        loss=loss,
        n_negatives=n_negatives,
        gbce_t=gbce_t,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
        total_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_supported_losses(self) -> None:
        assert SUPPORTED_LOSSES == ("softmax", "BCE", "gBCE", "sampled_softmax")

    def test_supported_optimizers(self) -> None:
        assert SUPPORTED_OPTIMIZERS == ("adam", "adamw")

    def test_supported_schedulers(self) -> None:
        assert SUPPORTED_SCHEDULERS == (None, "cosine_warmup")


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


class TestConfigureOptimizers:
    def test_adam_returns_adam(self, net: UniSRecNet) -> None:
        module = _make_module(net, optimizer="adam")
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)

    def test_adamw_returns_adamw(self, net: UniSRecNet) -> None:
        module = _make_module(net, optimizer="adamw")
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_no_scheduler_returns_optimizer_only(self, net: UniSRecNet) -> None:
        module = _make_module(net, scheduler=None)
        result = module.configure_optimizers()
        # When scheduler is None, returns just the optimizer (not a dict)
        assert isinstance(result, torch.optim.Optimizer)

    def test_cosine_warmup_returns_dict(self, net: UniSRecNet) -> None:
        module = _make_module(net, scheduler="cosine_warmup", total_steps=100)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "step"

    def test_unknown_optimizer_raises(self, net: UniSRecNet) -> None:
        module = _make_module(net, optimizer="sgd")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            module.configure_optimizers()

    def test_unknown_scheduler_raises(self, net: UniSRecNet) -> None:
        module = _make_module(net, scheduler="step_lr")
        with pytest.raises(ValueError, match="Unknown scheduler"):
            module.configure_optimizers()

    def test_cosine_warmup_total_steps_default(self, net: UniSRecNet) -> None:
        """When total_steps is None, it defaults to 1."""
        module = _make_module(net, scheduler="cosine_warmup", total_steps=None)
        result = module.configure_optimizers()
        assert isinstance(result, dict)

    def test_optimizer_lr(self, net: UniSRecNet) -> None:
        lr = 5e-4
        module = _make_module(net, optimizer="adam", lr=lr)
        opt = module.configure_optimizers()
        assert opt.param_groups[0]["lr"] == lr


# ---------------------------------------------------------------------------
# _cosine_warmup_scheduler
# ---------------------------------------------------------------------------


class TestCosineWarmupScheduler:
    def test_lr_at_step_zero_is_zero(self) -> None:
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = _cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.0)
        # LambdaLR stores the lambda; get factor for step 0
        lr_factor = scheduler.lr_lambdas[0](0)
        assert lr_factor == 0.0

    def test_lr_during_warmup_is_linear(self) -> None:
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        warmup_steps = 10
        scheduler = _cosine_warmup_scheduler(opt, warmup_steps=warmup_steps, total_steps=100)
        lr_fn = scheduler.lr_lambdas[0]
        for step in range(1, warmup_steps):
            assert lr_fn(step) == pytest.approx(step / warmup_steps)

    def test_lr_at_warmup_end_is_one(self) -> None:
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = _cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)
        lr_fn = scheduler.lr_lambdas[0]
        # At warmup_steps, progress = 0, cos(0) = 1 => factor = 1.0
        assert lr_fn(10) == pytest.approx(1.0)

    def test_lr_at_end_equals_min_lr_ratio(self) -> None:
        min_lr_ratio = 0.1
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = _cosine_warmup_scheduler(
            opt,
            warmup_steps=10,
            total_steps=100,
            min_lr_ratio=min_lr_ratio,
        )
        lr_fn = scheduler.lr_lambdas[0]
        # At total_steps, progress = 1, cos(pi) = -1 => factor = min_lr_ratio
        assert lr_fn(100) == pytest.approx(min_lr_ratio)

    def test_lr_at_cosine_midpoint(self) -> None:
        """At the midpoint of the cosine phase, factor should be (1 + min_lr_ratio) / 2."""
        warmup_steps = 10
        total_steps = 110
        min_lr_ratio = 0.0
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = _cosine_warmup_scheduler(
            opt,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
        lr_fn = scheduler.lr_lambdas[0]
        midpoint = warmup_steps + (total_steps - warmup_steps) // 2  # 60
        # progress = 0.5 => cos(pi/2) = 0 => factor = 0.5
        expected = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        assert lr_fn(midpoint) == pytest.approx(expected, abs=1e-6)

    def test_lr_with_nonzero_min_lr_ratio(self) -> None:
        min_lr_ratio = 0.3
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = _cosine_warmup_scheduler(
            opt,
            warmup_steps=0,
            total_steps=100,
            min_lr_ratio=min_lr_ratio,
        )
        lr_fn = scheduler.lr_lambdas[0]
        # At step 0 (warmup_steps=0, so cosine phase), progress=0, cos(0)=1 => factor=1.0
        assert lr_fn(0) == pytest.approx(1.0)
        # At total_steps => factor = min_lr_ratio
        assert lr_fn(100) == pytest.approx(min_lr_ratio)

    def test_returns_lambda_lr(self) -> None:
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = _cosine_warmup_scheduler(opt, warmup_steps=5, total_steps=50)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


# ---------------------------------------------------------------------------
# training_step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_softmax_returns_scalar(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="softmax")
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_softmax_positive_loss(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="softmax")
        batch = {
            "x": torch.tensor([[1, 2, 3, 4, 5]]),
            "y": torch.tensor([[2, 3, 4, 5, 6]]),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.item() > 0, "Cross-entropy loss should be positive"

    def test_bce_loss_returns_scalar(self, net: UniSRecNet) -> None:
        n_negatives = 3
        module = _make_module(net, loss="BCE", n_negatives=n_negatives)
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
            "negatives": torch.randint(1, 10, (2, 5, n_negatives)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gbce_loss_returns_scalar(self, net: UniSRecNet) -> None:
        n_negatives = 3
        module = _make_module(net, loss="gBCE", n_negatives=n_negatives)
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
            "negatives": torch.randint(1, 10, (2, 5, n_negatives)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sampled_softmax_loss_returns_scalar(self, net: UniSRecNet) -> None:
        n_negatives = 3
        module = _make_module(net, loss="sampled_softmax", n_negatives=n_negatives)
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
            "negatives": torch.randint(1, 10, (2, 5, n_negatives)),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_softmax_ignores_negatives_when_present(self, net: UniSRecNet) -> None:
        """Softmax loss uses full softmax even when negatives are provided."""
        module_no_neg = _make_module(net, loss="softmax")
        module_with_neg = _make_module(net, loss="softmax")
        net.eval()

        batch_no_neg = {
            "x": torch.tensor([[1, 2, 3, 4, 5]]),
            "y": torch.tensor([[2, 3, 4, 5, 6]]),
        }
        batch_with_neg = {
            "x": torch.tensor([[1, 2, 3, 4, 5]]),
            "y": torch.tensor([[2, 3, 4, 5, 6]]),
            "negatives": torch.randint(1, 10, (1, 5, 3)),
        }
        with torch.no_grad():
            loss_no_neg = module_no_neg.training_step(batch_no_neg, batch_idx=0)
            loss_with_neg = module_with_neg.training_step(batch_with_neg, batch_idx=0)
        torch.testing.assert_close(loss_no_neg, loss_with_neg)

    def test_all_padding_softmax(self, net: UniSRecNet) -> None:
        """When all targets are padding, cross_entropy with ignore_index returns NaN."""
        module = _make_module(net, loss="softmax")
        batch = {
            "x": torch.tensor([[0, 0, 0, 0, 0]]),
            "y": torch.tensor([[0, 0, 0, 0, 0]]),
        }
        loss = module.training_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert torch.isnan(loss)


# ---------------------------------------------------------------------------
# validation_step
# ---------------------------------------------------------------------------


class TestValidationStep:
    def test_validation_returns_scalar(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="softmax")
        module.eval()
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[4], [8]]),  # (B, 1)
        }
        with torch.no_grad():
            loss = module.validation_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_validation_uses_last_hidden(self, net: UniSRecNet) -> None:
        """Validation slices hidden to [:, -1:, :], so y shape (B, 1) works."""
        module = _make_module(net, loss="softmax")
        module.eval()
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3]]),
            "y": torch.tensor([[4]]),  # single target per sequence
        }
        with torch.no_grad():
            loss = module.validation_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_validation_with_negatives(self, net: UniSRecNet) -> None:
        n_negatives = 3
        module = _make_module(net, loss="BCE", n_negatives=n_negatives)
        module.eval()
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[4], [8]]),
            "negatives": torch.randint(1, 10, (2, 1, n_negatives)),
        }
        with torch.no_grad():
            loss = module.validation_step(batch, batch_idx=0)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# _calc_loss dispatch
# ---------------------------------------------------------------------------


class TestCalcLossDispatch:
    def test_softmax_without_negatives_uses_full_softmax(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="softmax")
        hidden = torch.randn(2, 5, 8)
        batch = {
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
        }
        loss = module._calc_loss(hidden, batch)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_bce_without_negatives_raises(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="BCE")
        hidden = torch.randn(2, 5, 8)
        batch = {
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
        }
        with pytest.raises(ValueError, match="requires negatives"):
            module._calc_loss(hidden, batch)

    def test_gbce_without_negatives_raises(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="gBCE")
        hidden = torch.randn(2, 5, 8)
        batch = {"y": torch.tensor([[1, 2, 3, 4, 5]])}
        with pytest.raises(ValueError, match="requires negatives"):
            module._calc_loss(hidden, batch)

    def test_sampled_softmax_without_negatives_raises(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="sampled_softmax")
        hidden = torch.randn(1, 5, 8)
        batch = {"y": torch.tensor([[1, 2, 3, 4, 5]])}
        with pytest.raises(ValueError, match="requires negatives"):
            module._calc_loss(hidden, batch)

    def test_unknown_loss_raises(self, net: UniSRecNet) -> None:
        module = _make_module(net, loss="mse")
        hidden = torch.randn(1, 5, 8)
        batch = {
            "y": torch.tensor([[1, 2, 3, 4, 5]]),
            "negatives": torch.randint(1, 10, (1, 5, 3)),
        }
        with pytest.raises(ValueError, match="Unknown loss"):
            module._calc_loss(hidden, batch)


# ---------------------------------------------------------------------------
# _get_item_embs / _get_all_embs
# ---------------------------------------------------------------------------


class TestEmbeddingHelpers:
    def test_get_item_embs(self, net: UniSRecNet) -> None:
        module = _make_module(net)
        item_ids = torch.tensor([[1, 2, 3]])
        embs = module._get_item_embs(item_ids)
        assert embs.shape == (1, 3, 8)  # (B, L, n_factors)

    def test_get_all_embs(self, net: UniSRecNet) -> None:
        module = _make_module(net)
        all_embs = module._get_all_embs()
        assert all_embs.shape == (11, 8)  # n_items + 1

    def test_get_pos_neg_logits_shape(self, net: UniSRecNet) -> None:
        module = _make_module(net)
        hidden = torch.randn(2, 5, 8)
        labels = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        negatives = torch.randint(1, 10, (2, 5, 3))
        logits = module._get_pos_neg_logits(hidden, labels, negatives)
        assert logits.shape == (2, 5, 4)  # 1 positive + 3 negatives


# ---------------------------------------------------------------------------
# Init stores params
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_all_attributes(self, net: UniSRecNet) -> None:
        module = _make_module(
            net,
            loss="BCE",
            n_negatives=5,
            optimizer="adam",
            scheduler="cosine_warmup",
            total_steps=200,
            warmup_ratio=0.1,
            min_lr_ratio=0.05,
            gbce_t=0.3,
        )
        assert module.loss_name == "BCE"
        assert module.n_negatives == 5
        assert module.optimizer_name == "adam"
        assert module.scheduler_name == "cosine_warmup"
        assert module.total_steps == 200
        assert module.warmup_ratio == 0.1
        assert module.min_lr_ratio == 0.05
        assert module.gbce_t == 0.3
        assert module.net is net
