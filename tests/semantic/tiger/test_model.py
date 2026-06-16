#  Copyright 2025 MTS (Mobile Telesystems)
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

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.semantic.tiger.model import TIGERModel
from rectools.semantic.tokenizer.emb_dataset import EmbDataset
from rectools.semantic.tokenizer.model import SIDTokenizer


@pytest.fixture
def trained_tokenizer() -> SIDTokenizer:
    seed_everything(42, workers=True)
    rng = np.random.RandomState(42)
    item_ids = list(range(1, 51))
    embeddings = rng.randn(50, 16).astype(np.float32)
    dataset = EmbDataset(item_ids, embeddings)
    tok = SIDTokenizer(
        codebook_dim=16,
        hidden_dims=None,
        codebook_sizes=[4, 4],
        input_dim=16,
        device="cpu",
        quantizer="rkmeans",
    )
    tok.init_codebooks(dataset)
    batch = dataset[:]
    tok(batch)
    return tok


@pytest.fixture
def interactions() -> pd.DataFrame:
    rows = []
    for user in range(1, 11):
        for t, item in enumerate(range(1, 6)):
            rows.append([user, item, t])
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])


class TestTIGERModel:  # pylint: disable=redefined-outer-name
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_init(self, trained_tokenizer: SIDTokenizer) -> None:
        model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
        )
        assert model.hidden_units == 16
        assert model.num_blocks == 1

    def test_predict_output_format(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
            beam_size=5,
            top_k=3,
        )
        result = model.predict(interactions, top_k=3)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"user_id", "item_id", "score", "rank"}
        # Each user should have at most top_k recommendations
        for _uid, group in result.groupby("user_id"):
            assert len(group) <= 3

    def test_save_and_load(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "tokenizer.pt"))
            assert os.path.exists(os.path.join(tmpdir, "model.pt"))
            assert os.path.exists(os.path.join(tmpdir, "config.json"))

            loaded = TIGERModel.load(tmpdir, device="cpu")
            assert loaded.hidden_units == model.hidden_units
            assert loaded.num_blocks == model.num_blocks
            assert loaded.num_heads == model.num_heads

    def test_predict_batching(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
            beam_size=5,
            top_k=3,
            eval_batch_size=2,
        )
        result = model.predict(interactions, top_k=3)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"user_id", "item_id", "score", "rank"}
        result_user_ids = set(result["user_id"].unique())
        input_user_ids = set(interactions["user_id"].unique())
        assert result_user_ids.issubset(input_user_ids)
        for _uid, group in result.groupby("user_id"):
            assert list(group["rank"]) == list(range(1, len(group) + 1))
            assert len(group) <= 3

    def test_fit_runs(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        torch.use_deterministic_algorithms(True)
        model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            max_epochs=1,
            batch_size=16,
            eval_batch_size=16,
            num_workers=0,
            beam_size=4,
            top_k=3,
            device="cpu",
        )
        # Split interactions for train/val
        train_df = interactions[interactions["user_id"] <= 7]
        val_df = interactions[interactions["user_id"] > 7]
        model.fit(train_df, val_df)
        # After fit, model should be able to predict
        result = model.predict(interactions, top_k=3)
        assert len(result) > 0
        torch.use_deterministic_algorithms(False)

    def test_evaluate_uses_unique_item_count_for_coverage(
        self,
        trained_tokenizer: SIDTokenizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
        )
        interactions = pd.DataFrame(
            [
                [1, 10, 1],
                [1, 20, 2],
                [1, 30, 3],
                [2, 10, 1],
                [2, 30, 2],
                [2, 50, 3],
            ],
            columns=["user_id", "item_id", "timestamp"],
        )
        captured = {}

        class FakeTrainer:
            def test(self, lightning_model, dataloaders):
                captured["num_items"] = lightning_model.num_items
                return [{"Coverage@10": 0.0}]

        monkeypatch.setattr("rectools.semantic.tiger.model.pl.Trainer", lambda: FakeTrainer())

        model.evaluate(interactions)

        assert captured["num_items"] == 4

    def test_predict_is_reproducible_with_random_seed(self, trained_tokenizer: SIDTokenizer) -> None:
        collision_sid = (0, 0)
        trained_tokenizer.id2sid = {
            10: collision_sid,
            20: collision_sid,
            30: (1, 1),
        }
        trained_tokenizer.sid2id.clear()

        interactions = pd.DataFrame(
            [
                [1, 10, 1],
                [1, 30, 2],
            ],
            columns=["user_id", "item_id", "timestamp"],
        )

        first_model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
            random_seed=42,
        )
        second_model = TIGERModel(
            tokenizer=trained_tokenizer,
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            max_length=10,
            device="cpu",
            random_seed=42,
        )

        generated_codes = torch.tensor([[[0, 0], [0, 0], [1, 1]]], dtype=torch.long)
        generated_scores = torch.tensor([[0.9, 0.8, 0.7]], dtype=torch.float32)

        def fake_generate(*args, **kwargs):
            return generated_codes, generated_scores

        first_model.model.generate = fake_generate  # type: ignore[method-assign]
        second_model.model.generate = fake_generate  # type: ignore[method-assign]

        first_result = first_model.predict(interactions, top_k=3)
        second_result = second_model.predict(interactions, top_k=3)

        assert first_result.equals(second_result)
