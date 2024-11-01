import typing as tp
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything

from rectools.columns import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.models.sasrec import IdEmbeddingsItemNet, SASRecDataPreparator, SASRecModel, SequenceDataset
from tests.models.utils import assert_second_fit_refits_model
from tests.testing_utils import assert_id_map_equal, assert_interactions_set_equal


class TestSASRecModel:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def dataset_hot_users_items(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df[:-4])

    @pytest.fixture
    def trainer(self) -> Trainer:
        return Trainer(
            max_epochs=2,
            min_epochs=2,
            deterministic=True,
            accelerator="cpu",
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 14, 13, 17, 12, 14, 13],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 12, 14, 12, 11, 14, 12, 17, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    # TODO: tests do not pass for multiple GPUs
    def test_u2i(self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=filter_viewed)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 30, 30, 40],
                        Columns.Item: [17, 13, 17, 13],
                        Columns.Rank: [1, 1, 2, 1],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 17, 11, 11, 13, 17, 17, 11, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(
        self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        items_to_recommend = np.array([11, 13, 17])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [12, 17, 11, 14, 11, 13, 17, 12, 14],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [17, 11, 14, 11, 13, 17, 12, 14, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                np.array([15, 13, 14]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 17, 17, 17],
                        Columns.Item: [14, 13, 15, 13, 15, 14, 15, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self,
        dataset: Dataset,
        trainer: Trainer,
        filter_itself: bool,
        whitelist: tp.Optional[np.ndarray],
        expected: pd.DataFrame,
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        target_items = np.array([12, 14, 17])
        actual = model.recommend_to_items(
            target_items=target_items,
            dataset=dataset,
            k=3,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, dataset_hot_users_items: Dataset, trainer: Trainer) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        assert_second_fit_refits_model(model, dataset_hot_users_items, pre_fit_callback=self._seed_everything)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [14, 12, 17],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [13, 14, 12],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_recommend_for_cold_user_with_hot_item(
        self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([20])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20, 20],
                        Columns.Item: [17, 15, 14, 12, 17],
                        Columns.Rank: [1, 2, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 20, 20, 20],
                        Columns.Item: [13, 12, 14, 13, 14, 12],
                        Columns.Rank: [1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_warn_when_hot_user_has_cold_items_in_recommend(
        self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 20, 50])
        with pytest.warns() as record:
            actual = model.recommend(
                users=users,
                dataset=dataset,
                k=3,
                filter_viewed=filter_viewed,
                on_unsupported_targets="warn",
            )
            pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
            pd.testing.assert_frame_equal(
                actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
                actual,
            )
        assert str(record[0].message) == "1 target users were considered cold because of missing known items"
        assert (
            str(record[1].message)
            == """
                Model `<class 'rectools.models.sasrec.SASRecModel'>` doesn't support recommendations for cold users,
                but some of given users are cold: they are not in the `dataset.user_id_map`
            """
        )


class TestSequenceDataset:

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 4, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 8, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.mark.parametrize(
        "expected_sessions, expected_weights",
        (([[14, 11, 12, 13], [15, 12, 11], [11, 17], [16]], [[1, 1, 4, 1], [1, 2, 1], [1, 8], [1]]),),
    )
    def test_from_interactions(
        self, interactions_df: pd.DataFrame, expected_sessions: List[List[int]], expected_weights: List[List[float]]
    ) -> None:
        actual = SequenceDataset.from_interactions(interactions_df)
        assert len(actual.sessions) == len(expected_sessions)
        assert all(
            actual_list == expected_list for actual_list, expected_list in zip(actual.sessions, expected_sessions)
        )
        assert len(actual.weights) == len(expected_weights)
        assert all(actual_list == expected_list for actual_list, expected_list in zip(actual.weights, expected_weights))


class TestSASRecDataPreparator:

    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def data_preparator(self) -> SASRecDataPreparator:
        return SASRecDataPreparator(session_max_len=3, batch_size=4, dataloader_num_workers=0)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([30, 40, 10]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 1, 1.0, "2021-11-25"],
                            [1, 2, 1.0, "2021-11-25"],
                            [0, 3, 2.0, "2021-11-26"],
                            [1, 4, 1.0, "2021-11-26"],
                            [0, 2, 1.0, "2021-11-27"],
                            [2, 5, 1.0, "2021-11-28"],
                            [2, 2, 1.0, "2021-11-29"],
                            [2, 3, 1.0, "2021-11-29"],
                            [2, 6, 1.0, "2021-11-30"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_process_dataset_train(
        self,
        dataset: Dataset,
        data_preparator: SASRecDataPreparator,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        actual = data_preparator.process_dataset_train(dataset)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([10, 20]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 6, 1.0, "2021-11-30"],
                            [0, 2, 1.0, "2021-11-29"],
                            [0, 3, 1.0, "2021-11-29"],
                            [0, 5, 1.0, "2021-11-28"],
                            [1, 6, 9.0, "2021-11-28"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_transform_dataset_u2i(
        self,
        dataset: Dataset,
        data_preparator: SASRecDataPreparator,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        users = [10, 20]
        actual = data_preparator.transform_dataset_u2i(dataset, users)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([10, 30, 40, 50, 20]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 6, 1.0, "2021-11-30"],
                            [0, 2, 1.0, "2021-11-29"],
                            [0, 3, 1.0, "2021-11-29"],
                            [1, 2, 1.0, "2021-11-27"],
                            [1, 3, 2.0, "2021-11-26"],
                            [1, 1, 1.0, "2021-11-25"],
                            [2, 2, 1.0, "2021-11-25"],
                            [2, 4, 1.0, "2021-11-26"],
                            [0, 5, 1.0, "2021-11-28"],
                            [4, 6, 9.0, "2021-11-28"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_tranform_dataset_i2i(
        self,
        dataset: Dataset,
        data_preparator: SASRecDataPreparator,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        actual = data_preparator.transform_dataset_i2i(dataset)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    @pytest.mark.parametrize(
        "train_batch",
        (
            (
                [
                    torch.tensor([[5, 2, 3], [0, 1, 3], [0, 0, 2]]),
                    torch.tensor([[2, 3, 6], [0, 3, 2], [0, 0, 4]]),
                    torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
                ]
            ),
        ),
    )
    def test_get_dataloader_train(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, train_batch: List
    ) -> None:
        dataset = data_preparator.process_dataset_train(dataset)
        dataloader = data_preparator.get_dataloader_train(dataset)
        actual = next(iter(dataloader))
        for i, value in enumerate(actual):
            assert torch.equal(value, train_batch[i])

    @pytest.mark.parametrize(
        "recommend_batch",
        ((torch.tensor([[2, 3, 6], [1, 3, 2], [0, 2, 4], [0, 0, 6]])),),
    )
    def test_get_dataloader_recommend(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, recommend_batch: torch.Tensor
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataset = data_preparator.transform_dataset_i2i(dataset)
        dataloader = data_preparator.get_dataloader_recommend(dataset)
        actual = next(iter(dataloader))
        assert torch.equal(actual, recommend_batch)
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
