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
from pytorch_lightning import seed_everything

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.dataset.features import SparseFeatures
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet, ItemNetBase, ItemNetConstructor
from tests.testing_utils import assert_feature_set_equal

from ..data import DATASET, INTERACTIONS


class TestIdEmbeddingsItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, n_factors: int) -> None:
        item_id_embeddings = IdEmbeddingsItemNet.from_dataset(DATASET, n_factors=n_factors, dropout_rate=0.5)

        actual_n_items = item_id_embeddings.n_items
        actual_embedding_dim = item_id_embeddings.ids_emb.embedding_dim

        assert actual_n_items == DATASET.item_id_map.size
        assert actual_embedding_dim == n_factors

    @pytest.mark.parametrize("n_items,n_factors", ((2, 10), (4, 100)))
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

    def test_feature_catalog(self, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)
        assert isinstance(cat_item_embeddings, CatFeaturesItemNet)
        expected_feature_catalog = torch.arange(0, cat_item_embeddings.n_cat_features)
        assert torch.equal(cat_item_embeddings.feature_catalog, expected_feature_catalog)

    def test_get_dense_item_features(self, dataset_item_features: Dataset) -> None:
        items = torch.from_numpy(
            dataset_item_features.item_id_map.convert_to_internal(INTERACTIONS[Columns.Item].unique())
        )
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)

        assert isinstance(cat_item_embeddings, CatFeaturesItemNet)

        actual_feature_dense = cat_item_embeddings.get_dense_item_features(items)
        expected_feature_dense = torch.tensor(
            [
                [1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 1, 0, 0, 1],
            ],
            dtype=torch.float,
        )

        assert torch.equal(actual_feature_dense, expected_feature_dense)

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, n_factors: int, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5
        )

        assert isinstance(cat_item_embeddings, CatFeaturesItemNet)

        actual_item_features = cat_item_embeddings.item_features
        actual_n_items = cat_item_embeddings.n_items
        actual_n_cat_features = cat_item_embeddings.n_cat_features
        actual_embedding_dim = cat_item_embeddings.category_embeddings.embedding_dim

        expected_item_features = dataset_item_features.item_features

        assert isinstance(expected_item_features, SparseFeatures)
        expected_cat_item_features = expected_item_features.get_cat_features()

        assert_feature_set_equal(actual_item_features, expected_cat_item_features)
        assert actual_n_items == dataset_item_features.item_id_map.size
        assert actual_n_cat_features == len(expected_cat_item_features.names)
        assert actual_embedding_dim == n_factors

    @pytest.mark.parametrize(
        "n_items,n_factors",
        ((2, 10), (4, 100)),
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

    @pytest.mark.parametrize(
        "item_features,cat_item_features,make_dense_item_features",
        (
            (None, (), False),
            (
                pd.DataFrame(
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
                ),
                (),
                False,
            ),
            (
                pd.DataFrame(
                    [
                        [11, 1, 1],
                        [12, 1, 2],
                        [13, 1, 3],
                        [14, 2, 1],
                        [15, 2, 2],
                        [17, 2, 3],
                    ],
                    columns=[Columns.Item, "f1", "f2"],
                ),
                ["f1", "f2"],
                True,
            ),
        ),
    )
    def test_when_cat_item_features_is_none(
        self,
        item_features: tp.Optional[pd.DataFrame],
        cat_item_features: tp.Iterable[str],
        make_dense_item_features: bool,
    ) -> None:
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=cat_item_features,
            make_dense_item_features=make_dense_item_features,
        )
        cat_features_item_net = CatFeaturesItemNet.from_dataset(ds, n_factors=10, dropout_rate=0.5)
        assert cat_features_item_net is None


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
                [16, "f1", "f1val2"],
                [16, "f2", "f2val3"],
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
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

    def test_catalog(self) -> None:
        item_net = ItemNetConstructor.from_dataset(
            DATASET, n_factors=10, dropout_rate=0.5, item_net_block_types=(IdEmbeddingsItemNet,)
        )
        expected_feature_catalog = torch.arange(0, DATASET.item_id_map.size)
        assert torch.equal(item_net.catalog, expected_feature_catalog)

    @pytest.mark.parametrize(
        "item_net_block_types,n_factors",
        (
            ((IdEmbeddingsItemNet,), 8),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), 16),
            ((CatFeaturesItemNet,), 16),
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
        "item_net_block_types,make_dense_item_features,expected_n_item_net_blocks",
        (
            ((IdEmbeddingsItemNet,), False, 1),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), False, 2),
            ((IdEmbeddingsItemNet,), True, 1),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), True, 1),
        ),
    )
    def test_correct_number_of_item_net_blocks(
        self,
        dataset_item_features: Dataset,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        make_dense_item_features: bool,
        expected_n_item_net_blocks: int,
    ) -> None:
        if make_dense_item_features:
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
                make_dense_user_features=make_dense_item_features,
            )
        else:
            ds = dataset_item_features

        item_net: ItemNetConstructor = ItemNetConstructor.from_dataset(
            ds, n_factors=10, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )

        actual_n_items = item_net.n_items
        actual_n_item_net_blocks = len(item_net.item_net_blocks)

        assert actual_n_items == dataset_item_features.item_id_map.size
        assert actual_n_item_net_blocks == expected_n_item_net_blocks

    @pytest.mark.parametrize(
        "item_net_block_types,n_items,n_factors",
        (
            ((IdEmbeddingsItemNet,), 2, 16),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), 4, 8),
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
        item_net: ItemNetConstructor = ItemNetConstructor.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )

        expected_embeddings = item_net(items)

        assert expected_embeddings.shape == (n_items, n_factors)

    @pytest.mark.parametrize(
        "item_net_block_types,item_features,make_dense_item_features",
        (
            ([], None, False),
            ((CatFeaturesItemNet,), None, False),
            (
                (CatFeaturesItemNet,),
                pd.DataFrame(
                    [
                        [11, 1, 1],
                        [12, 1, 2],
                        [13, 1, 3],
                        [14, 2, 1],
                        [15, 2, 2],
                        [17, 2, 3],
                    ],
                    columns=[Columns.Item, "f1", "f2"],
                ),
                True,
            ),
            (
                (CatFeaturesItemNet,),
                pd.DataFrame(
                    [
                        [11, "f3", 0],
                        [12, "f3", 1],
                        [13, "f3", 2],
                        [14, "f3", 3],
                        [15, "f3", 4],
                        [17, "f3", 5],
                        [16, "f3", 6],
                    ],
                    columns=[Columns.Item, "feature", "value"],
                ),
                False,
            ),
        ),
    )
    def test_raise_when_no_item_net_blocks(
        self,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        item_features: tp.Optional[pd.DataFrame],
        make_dense_item_features: bool,
    ) -> None:
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            make_dense_item_features=make_dense_item_features,
        )
        with pytest.raises(ValueError):
            ItemNetConstructor.from_dataset(
                ds, n_factors=10, dropout_rate=0.5, item_net_block_types=item_net_block_types
            )