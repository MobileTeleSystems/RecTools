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

import numpy as np
import pandas as pd
import pytest
from pytorch_lightning import seed_everything

from rectools.semantic.data_handling.dataset import PaddingCollateFn, TIGERDataset
from rectools.semantic.tokenizer.emb_dataset import EmbDataset
from rectools.semantic.tokenizer.model import SIDTokenizer


@pytest.fixture
def trained_tokenizer() -> SIDTokenizer:
    seed_everything(42, workers=True)
    rng = np.random.RandomState(42)
    item_ids = list(range(1, 21))
    embeddings = rng.randn(20, 16).astype(np.float32)
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
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "timestamp": list(range(1, 13)),
        }
    )


class TestTIGERDataset:  # pylint: disable=redefined-outer-name
    def test_eval_mode_len(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        ds = TIGERDataset(
            interactions=interactions,
            tokenizer=trained_tokenizer,
            codebook_sizes=[4, 4],
            max_length=10,
            codeword_offset=2,
            bos_token_id=1,
            only_last=True,
            eval_mode=True,
        )
        # 3 users, each gets one sample
        assert len(ds) == 3

    def test_eval_mode_getitem(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        ds = TIGERDataset(
            interactions=interactions,
            tokenizer=trained_tokenizer,
            codebook_sizes=[4, 4],
            max_length=10,
            codeword_offset=2,
            bos_token_id=1,
            only_last=True,
            eval_mode=True,
        )
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert isinstance(sample["input_ids"], np.ndarray)
        assert isinstance(sample["labels"], (int, np.integer))

    def test_train_mode_only_last(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        ds = TIGERDataset(
            interactions=interactions,
            tokenizer=trained_tokenizer,
            codebook_sizes=[4, 4],
            max_length=10,
            codeword_offset=2,
            bos_token_id=1,
            only_last=True,
            eval_mode=False,
        )
        assert len(ds) == 3
        sample = ds[0]
        assert "input_ids" in sample
        assert "dec_input" in sample
        assert "labels" in sample

    def test_train_mode_all_subsequences(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        ds = TIGERDataset(
            interactions=interactions,
            tokenizer=trained_tokenizer,
            codebook_sizes=[4, 4],
            max_length=10,
            codeword_offset=2,
            bos_token_id=1,
            only_last=False,
            eval_mode=False,
        )
        # Each user has 4 items -> 3 subsequences per user -> 9 total
        assert len(ds) == 9

    def test_max_users(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        ds = TIGERDataset(
            interactions=interactions,
            tokenizer=trained_tokenizer,
            codebook_sizes=[4, 4],
            max_length=10,
            codeword_offset=2,
            bos_token_id=1,
            only_last=True,
            eval_mode=True,
            max_users=2,
        )
        assert len(ds) == 2

    def test_dec_input_starts_with_bos(self, trained_tokenizer: SIDTokenizer, interactions: pd.DataFrame) -> None:
        ds = TIGERDataset(
            interactions=interactions,
            tokenizer=trained_tokenizer,
            codebook_sizes=[4, 4],
            max_length=10,
            codeword_offset=2,
            bos_token_id=1,
            only_last=True,
            eval_mode=False,
        )
        sample = ds[0]
        assert sample["dec_input"][0] == 1  # BOS token


class TestPaddingCollateFn:
    def test_pads_sequences(self) -> None:
        batch = [
            {"input_ids": np.array([1, 2, 3]), "labels": np.array([10, 11])},
            {"input_ids": np.array([4, 5]), "labels": np.array([12, 13])},
        ]
        collate = PaddingCollateFn(padding_value=0, labels_padding_value=-100)
        result = collate(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["labels"].shape == (2, 2)
        # Second sequence should be padded
        assert result["input_ids"][1, 2].item() == 0

    def test_scalar_labels(self) -> None:
        batch = [
            {"input_ids": np.array([1, 2, 3]), "labels": 10},
            {"input_ids": np.array([4, 5, 6]), "labels": 20},
        ]
        collate = PaddingCollateFn(padding_value=0)
        result = collate(batch)
        assert result["labels"].shape == (2,)
        assert result["labels"].tolist() == [10, 20]
