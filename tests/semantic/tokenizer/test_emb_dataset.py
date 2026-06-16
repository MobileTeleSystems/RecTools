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
import pytest

from rectools.semantic.tokenizer.emb_dataset import EmbDataset


class TestEmbDataset:
    @pytest.fixture
    def dataset(self) -> EmbDataset:
        item_ids = [10, 20, 30, 40]
        embeddings = np.random.RandomState(42).randn(4, 8).astype(np.float32)
        return EmbDataset(item_ids, embeddings)

    def test_len(self, dataset: EmbDataset) -> None:
        assert len(dataset) == 4

    def test_dim(self, dataset: EmbDataset) -> None:
        assert dataset.dim == 8

    def test_getitem_int(self, dataset: EmbDataset) -> None:
        item = dataset[0]
        assert item["item_id"] == 10
        assert item["embed"].shape == (8,)

    def test_getitem_slice(self, dataset: EmbDataset) -> None:
        batch = dataset[0:2]
        assert batch["item_id"] == [10, 20]
        assert batch["embed"].shape == (2, 8)

    def test_getitem_full_slice(self, dataset: EmbDataset) -> None:
        batch = dataset[:]
        assert len(batch["item_id"]) == 4

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="item_ids length"):
            EmbDataset([1, 2, 3], np.zeros((4, 8)))
