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

"""Tests for UniSRecModel (standalone, tensor-based API)."""

import typing as tp

import pytest
import torch

from rectools.fast_transformers import UniSRecModel


def _make_embeddings(n_items: int = 25, dim: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    emb = torch.randn(n_items, dim)
    emb[0] = 0.0
    return emb


def _make_interactions(
    n_users: int = 20, n_items: int = 25, seed: int = 42
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic (user_ids, item_ids, timestamps) tensors."""
    rng = torch.Generator().manual_seed(seed)
    users, items, timestamps = [], [], []
    for u in range(n_users):
        n_inter = torch.randint(3, 8, (1,), generator=rng).item()
        item_pool = torch.randperm(n_items, generator=rng)[: int(n_inter)] + 1  # 1-based
        for rank, item in enumerate(item_pool):
            users.append(u)
            items.append(item.item())
            timestamps.append(rank)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
        torch.tensor(timestamps, dtype=torch.long),
    )


def _make_model(**kwargs: tp.Any) -> UniSRecModel:
    defaults: tp.Dict[str, tp.Any] = {
        "pretrained_item_embeddings": _make_embeddings(),
        "n_factors": 16,
        "projection_hidden": 32,
        "n_blocks": 1,
        "n_heads": 2,
        "session_max_len": 8,
        "epochs": 1,
        "batch_size": 16,
        "verbose": 0,
    }
    defaults.update(kwargs)
    return UniSRecModel(**defaults)


class TestFit:
    def test_fit_returns_self(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model()
        result = model.fit(user_ids, item_ids, timestamps)
        assert result is model

    def test_is_fitted_after_fit(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model()
        assert not model.is_fitted
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_net_accessible_after_fit(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model()
        model.fit(user_ids, item_ids, timestamps)
        net = model.net
        assert net is not None

    def test_item_id_mapping_has_original_ids(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model()
        model.fit(user_ids, item_ids, timestamps)
        mapping = model.item_id_mapping
        assert mapping is not None
        original_unique = torch.unique(item_ids)
        assert set(mapping.tolist()) == set(original_unique.tolist())

    def test_net_not_accessible_before_fit(self) -> None:
        model = _make_model()
        with pytest.raises(AssertionError):
            _ = model.net


class TestLosses:
    def test_softmax_loss(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="softmax", epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_bce_loss(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="BCE", n_negatives=3, epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_gbce_loss(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="gBCE", n_negatives=3, epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_sampled_softmax_loss(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="sampled_softmax", n_negatives=3, epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_bce_loss_with_patience(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="BCE", n_negatives=3, patience=2, epochs=3)
        with pytest.warns(UserWarning, match="Early stopping"):
            model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_gbce_loss_with_patience(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="gBCE", n_negatives=3, patience=2, epochs=3)
        with pytest.warns(UserWarning, match="Early stopping"):
            model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_sampled_softmax_loss_with_patience(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(loss="sampled_softmax", n_negatives=3, patience=2, epochs=3)
        with pytest.warns(UserWarning, match="Early stopping"):
            model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_invalid_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported loss"):
            _make_model(loss="invalid")

    def test_n_negatives_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _make_model(loss="BCE", n_negatives=0)

    def test_n_negatives_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _make_model(loss="BCE", n_negatives=-1)

    def test_n_negatives_none_for_bce_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _make_model(loss="BCE", n_negatives=None)


class TestOptimizer:
    def test_adam(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(optimizer="adam", epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_adamw(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(optimizer="adamw", epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_invalid_optimizer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            _make_model(optimizer="sgd")


class TestScheduler:
    def test_cosine_warmup(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(scheduler="cosine_warmup", warmup_ratio=0.1, epochs=2)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted

    def test_invalid_scheduler_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            _make_model(scheduler="step")


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path: tp.Any) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1)
        model.fit(user_ids, item_ids, timestamps)

        ckpt_path = tmp_path / "model.pt"
        model.save_checkpoint(ckpt_path)

        model2 = _make_model(epochs=1)
        model2.load_checkpoint(ckpt_path, device="cpu")
        assert model2.is_fitted

        mapping1 = model.item_id_mapping
        mapping2 = model2.item_id_mapping
        assert mapping1 is not None
        assert mapping2 is not None
        assert torch.equal(mapping1, mapping2)

    def test_checkpoint_contains_arch_key(self, tmp_path: tp.Any) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1)
        model.fit(user_ids, item_ids, timestamps)

        ckpt_path = tmp_path / "model.pt"
        model.save_checkpoint(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "arch" in ckpt
        assert "n_factors" in ckpt["arch"]
        assert ckpt["arch"]["n_factors"] == 16

    def test_load_mismatched_arch_raises(self, tmp_path: tp.Any) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1, n_factors=16)
        model.fit(user_ids, item_ids, timestamps)

        ckpt_path = tmp_path / "model.pt"
        model.save_checkpoint(ckpt_path)

        model2 = _make_model(epochs=1, n_factors=32)
        with pytest.raises(ValueError, match="Architecture mismatch"):
            model2.load_checkpoint(ckpt_path, device="cpu")


class TestFFNTypes:
    @pytest.mark.parametrize("ffn_type", ["conv1d", "linear_gelu", "linear_relu"])
    def test_ffn_type(self, ffn_type: str) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(ffn_type=ffn_type, ffn_expansion=2, epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted


class TestEarlyStopping:
    def test_patience(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(patience=2, epochs=5)
        with pytest.warns(UserWarning, match="Early stopping"):
            model.fit(user_ids, item_ids, timestamps)
        assert model.is_fitted


class TestPredictTopk:
    def test_returns_external_ids(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1)
        model.fit(user_ids, item_ids, timestamps)

        mapping = model.item_id_mapping
        assert mapping is not None
        external_set = set(mapping.tolist())

        # Build a dummy input sequence from known external IDs
        known_external = mapping[:3]
        internal = model.map_item_ids(known_external)
        # Left-pad to session_max_len
        padded = torch.zeros(1, 8, dtype=torch.long)
        padded[0, -len(internal) :] = internal

        scores, top_ids = model.predict_topk(padded, k=5)
        assert scores.shape == (1, 5)
        assert top_ids.shape == (1, 5)

        # All returned IDs should be external (present in item_id_mapping)
        for item_id in top_ids[0].tolist():
            assert item_id in external_set, f"predict_topk returned internal ID {item_id}; expected external IDs"

    def test_scores_are_descending(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1)
        model.fit(user_ids, item_ids, timestamps)

        mapping = model.item_id_mapping
        assert mapping is not None
        internal = model.map_item_ids(mapping[:3])
        padded = torch.zeros(1, 8, dtype=torch.long)
        padded[0, -len(internal) :] = internal

        scores, _ = model.predict_topk(padded, k=5)
        for i in range(scores.shape[1] - 1):
            assert scores[0, i] >= scores[0, i + 1]


class TestMapItemIds:
    def test_dense_known_items(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        unique = model.item_id_mapping
        assert unique is not None
        result = model.map_item_ids(unique)
        expected = torch.arange(1, len(unique) + 1, dtype=torch.long)
        assert result.tolist() == expected.tolist()

    def test_dense_unknown_items(self) -> None:
        user_ids, item_ids, timestamps = _make_interactions()
        model = _make_model(epochs=1)
        model.fit(user_ids, item_ids, timestamps)
        unknown = torch.tensor([9999, 8888], dtype=torch.long)
        result = model.map_item_ids(unknown)
        assert result.tolist() == [0, 0]

    def test_unfitted_raises(self) -> None:
        model = _make_model()
        with pytest.raises(AssertionError):
            model.map_item_ids(torch.tensor([1, 2]))
