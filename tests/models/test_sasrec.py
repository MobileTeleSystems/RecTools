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
from rectools.dataset.features import SparseFeatures
from rectools.models.sasrec import CatFeaturesItemNet, IdEmbeddingsItemNet, ItemNetBase, ItemNetConstructor
from tests.testing_utils import assert_feature_set_equal

from .data import DATASET, INTERACTIONS


class TestIdEmbeddingsItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    def test_device(self) -> None:
        id_embeddings = IdEmbeddingsItemNet.from_dataset(DATASET, n_factors=5, dropout_rate=0.5)
        assert id_embeddings.device == torch.device("cpu")

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, n_factors: int) -> None:
        item_id_embeddings = IdEmbeddingsItemNet.from_dataset(DATASET, n_factors=n_factors, dropout_rate=0.5)

        actual_n_items = item_id_embeddings.n_items
        actual_embedding_dim = item_id_embeddings.ids_emb.embedding_dim

        assert actual_n_items == DATASET.item_id_map.size
        assert actual_embedding_dim == n_factors

    @pytest.mark.parametrize(
        "n_items,n_factors",
        (
            (
                2,
                10,
            ),
            (
                4,
                100,
            ),
        ),
    )
    def test_embedding_shape_after_model_pass(self, n_items: int, n_factors: int) -> None:
        items = torch.from_numpy(np.random.choice(DATASET.item_id_map.internal_ids, size=n_items, replace=False))
        item_id_embeddings = IdEmbeddingsItemNet.from_dataset(DATASET, n_factors=n_factors, dropout_rate=0.5)

        expected_item_ids = item_id_embeddings(items)
        assert expected_item_ids.shape == (n_items, n_factors)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCatFeaturesItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset_item_features(self) -> Dataset:
        item_features = pd.DataFrame(
            [
                [11, "f1", "f1val1"],
                [11, "f2", "f2val1"],
                [12, "f1", "f1val1"],
                [12, "f2", "f2val2"],
                [13, "f1", "f1val1"],
                [13, "f2", "f2val3"],
                [14, "f1", "f1val2"],
                [14, "f2", "f2val1"],
                [15, "f1", "f1val2"],
                [15, "f2", "f2val2"],
                [17, "f1", "f1val2"],
                [17, "f2", "f2val3"],
                [16, "f1", "f1val2"],
                [16, "f2", "f2val3"],
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
                [17, "f3", 5],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    def test_device(self, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)
        assert cat_item_embeddings.device == torch.device("cpu")

    def test_feature_catalogue(self, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)
        expected_feature_catalogue = torch.arange(
            0, cat_item_embeddings.n_cat_features, device=cat_item_embeddings.device
        )
        assert torch.equal(cat_item_embeddings.feature_catalogue, expected_feature_catalogue)

    def test_get_dense_item_features(self, dataset_item_features: Dataset) -> None:
        items = torch.from_numpy(
            dataset_item_features.item_id_map.convert_to_internal(INTERACTIONS[Columns.Item].unique())
        )
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)

        actual_feature_dense = cat_item_embeddings.get_dense_item_features(items)
        expected_feature_dense = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
            ],
            device=cat_item_embeddings.device,
        )

        assert torch.equal(actual_feature_dense, expected_feature_dense)

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, n_factors: int, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5
        )

        actual_item_features = cat_item_embeddings.item_features
        actual_n_items = cat_item_embeddings.n_items
        actual_n_cat_features = cat_item_embeddings.n_cat_features
        actual_embedding_dim = cat_item_embeddings.category_embeddings.embedding_dim

        expected_item_features = dataset_item_features.item_features
        # TODO: remove after adding Dense Features support
        if isinstance(expected_item_features, SparseFeatures):
            expected_cat_item_features = expected_item_features.get_cat_features()

            assert_feature_set_equal(actual_item_features, expected_cat_item_features)
            assert actual_n_items == dataset_item_features.item_id_map.size
            assert actual_n_cat_features == len(expected_cat_item_features.names)
            assert actual_embedding_dim == n_factors

    @pytest.mark.parametrize(
        "n_items,n_factors",
        (
            (
                2,
                10,
            ),
            (
                4,
                100,
            ),
        ),
    )
    def test_embedding_shape_after_model_pass(
        self, dataset_item_features: Dataset, n_items: int, n_factors: int
    ) -> None:
        items = torch.from_numpy(
            np.random.choice(dataset_item_features.item_id_map.internal_ids, size=n_items, replace=False)
        )
        cat_item_embeddings = IdEmbeddingsItemNet.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5
        )

        expected_item_ids = cat_item_embeddings(items)
        assert expected_item_ids.shape == (n_items, n_factors)

    def test_raises_when_dataset_no_features(self) -> None:
        with pytest.raises(ValueError):
            CatFeaturesItemNet.from_dataset(DATASET, n_factors=10, dropout_rate=0.5)

    # TODO: remove after adding Dense Features support
    def test_raises_when_item_features_dense(self) -> None:
        item_features = pd.DataFrame(
            [
                [11, 1, 1],
                [12, 1, 2],
                [13, 1, 3],
                [14, 2, 1],
                [15, 2, 2],
                [17, 2, 3],
            ],
            columns=[Columns.Item, "f1", "f2"],
        )
        ds = Dataset.construct(
            INTERACTIONS, item_features_df=item_features, cat_item_features=["f1", "f2"], make_dense_item_features=True
        )
        with pytest.raises(ValueError):
            CatFeaturesItemNet.from_dataset(ds, n_factors=10, dropout_rate=0.5)

    def test_raises_when_item_features_numeric(self) -> None:
        item_features = pd.DataFrame(
            [
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
                [17, "f3", 5],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
        )
        with pytest.raises(ValueError):
            CatFeaturesItemNet.from_dataset(ds, n_factors=10, dropout_rate=0.5)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestItemNetConstructor:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset_item_features(self) -> Dataset:
        item_features = pd.DataFrame(
            [
                [11, "f1", "f1val1"],
                [11, "f2", "f2val1"],
                [12, "f1", "f1val1"],
                [12, "f2", "f2val2"],
                [13, "f1", "f1val1"],
                [13, "f2", "f2val3"],
                [14, "f1", "f1val2"],
                [14, "f2", "f2val1"],
                [15, "f1", "f1val2"],
                [15, "f2", "f2val2"],
                [17, "f1", "f1val2"],
                [17, "f2", "f2val3"],
                [16, "f1", "f1val2"],
                [16, "f2", "f2val3"],
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
                [17, "f3", 5],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    def test_device(self) -> None:
        item_net = ItemNetConstructor.from_dataset(
            DATASET, n_factors=10, dropout_rate=0.5, item_net_block_types=(IdEmbeddingsItemNet,)
        )
        assert item_net.device == torch.device("cpu")

    def test_catalogue(self) -> None:
        item_net = ItemNetConstructor.from_dataset(
            DATASET, n_factors=10, dropout_rate=0.5, item_net_block_types=(IdEmbeddingsItemNet,)
        )
        expected_feature_catalogue = torch.arange(0, item_net.n_items, device=item_net.device)
        assert torch.equal(item_net.catalogue, expected_feature_catalogue)

    @pytest.mark.parametrize(
        "item_net_block_types,n_factors",
        (
            (
                (IdEmbeddingsItemNet,),
                8,
            ),
            (
                (IdEmbeddingsItemNet, CatFeaturesItemNet),
                16,
            ),
        ),
    )
    def test_get_all_embeddings(
        self, dataset_item_features: Dataset, item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]], n_factors: int
    ) -> None:
        item_net = ItemNetConstructor.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )
        assert item_net.get_all_embeddings().shape == (item_net.n_items, n_factors)

    @pytest.mark.parametrize(
        "item_net_block_types",
        (
            (IdEmbeddingsItemNet,),
            (IdEmbeddingsItemNet, CatFeaturesItemNet),
        ),
    )
    def test_create_from_dataset(
        self, dataset_item_features: Dataset, item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]]
    ) -> None:
        item_net = ItemNetConstructor.from_dataset(
            dataset_item_features, n_factors=10, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )

        actual_n_items = item_net.n_items
        actual_item_net_blocks = len(item_net.item_net_blocks)

        assert actual_n_items == dataset_item_features.item_id_map.size
        assert actual_item_net_blocks == len(item_net_block_types)

    @pytest.mark.parametrize(
        "item_net_block_types,n_items,n_factors",
        (
            (
                (IdEmbeddingsItemNet,),
                2,
                16,
            ),
            (
                (IdEmbeddingsItemNet, CatFeaturesItemNet),
                4,
                8,
            ),
        ),
    )
    def test_embedding_shape_after_model_pass(
        self,
        dataset_item_features: Dataset,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        n_items: int,
        n_factors: int,
    ) -> None:
        items = torch.from_numpy(
            np.random.choice(dataset_item_features.item_id_map.internal_ids, size=n_items, replace=False)
        )
        item_net = ItemNetConstructor.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )

        expected_embeddings = item_net(items)

        assert expected_embeddings.shape == (n_items, n_factors)

    def test_raise_when_no_item_net_blocks(self) -> None:
        with pytest.raises(ValueError):
            ItemNetConstructor.from_dataset(DATASET, n_factors=10, dropout_rate=0.5, item_net_block_types=[])
