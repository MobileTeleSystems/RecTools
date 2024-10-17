#  Copyright 2024 MTS (Mobile Telesystems)
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


import typing as tp

import numpy as np
import pandas as pd
import pytest
import torch
from lightning_fabric import seed_everything

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.sasrec import CatFeaturesItemNet, IdEmbeddingsItemNet, ItemNetConstructor, SASRecModel
from tests.models.utils import assert_second_fit_refits_model

from .data import INTERACTIONS


@pytest.mark.filterwarnings("ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestIdEmbeddingsItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset(self) -> Dataset:
        ds = Dataset.construct(INTERACTIONS)
        return ds

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, dataset: Dataset, n_factors: int) -> None:
        id_embeddings = IdEmbeddingsItemNet.from_dataset(dataset, n_factors=n_factors, dropout_rate=0.5)

        actual_n_items = id_embeddings.n_items
        actual_embedding_dim = id_embeddings.ids_emb.embedding_dim

        assert actual_n_items == INTERACTIONS[Columns.Item].nunique()
        assert actual_embedding_dim == n_factors

    def test_device(self, dataset: Dataset) -> None:
        id_embeddings = IdEmbeddingsItemNet.from_dataset(dataset, n_factors=5, dropout_rate=0.5)
        assert id_embeddings.device == torch.device("cpu")


@pytest.mark.filterwarnings("ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestCatFeaturesItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)


@pytest.mark.filterwarnings("ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestItemNetConstructor:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)
