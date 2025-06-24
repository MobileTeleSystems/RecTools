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

import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools.columns import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.dataset.features import DenseFeatures
from rectools.models.nn.transformers.data_preparator import SequenceDataset, TransformerDataPreparatorBase
from tests.testing_utils import assert_feature_set_equal, assert_id_map_equal, assert_interactions_set_equal


class TestSequenceDataset:

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30", 0],
                [10, 11, 1, "2021-11-29", 1],
                [10, 12, 4, "2021-11-29", 1],
                [30, 11, 1, "2021-11-27", 0],
                [30, 12, 2, "2021-11-26", 1],
                [30, 15, 1, "2021-11-25", 1],
                [40, 11, 1, "2021-11-25", 2],
                [40, 17, 8, "2021-11-26", 1],
                [50, 16, 1, "2021-11-25", 0],
                [10, 14, 1, "2021-11-28", 0],
            ],
            columns=Columns.Interactions + ["extra_column"],
        )
        return interactions_df

    @pytest.mark.parametrize(
        "expected_sessions, expected_weights, expected_extra_column",
        (
            (
                [[14, 11, 12, 13], [15, 12, 11], [11, 17], [16]],
                [[1, 1, 4, 1], [1, 2, 1], [1, 8], [1]],
                [[0, 1, 1, 0], [1, 1, 0], [2, 1], [0]],
            ),
        ),
    )
    def test_from_interactions(
        self,
        interactions_df: pd.DataFrame,
        expected_sessions: tp.List[tp.List[int]],
        expected_weights: tp.List[tp.List[float]],
        expected_extra_column: tp.Dict[str, tp.List[tp.Any]],
    ) -> None:
        actual = SequenceDataset.from_interactions(interactions=interactions_df, sort_users=True)
        assert len(actual.sessions) == len(expected_sessions)
        assert all(
            actual_list == expected_list for actual_list, expected_list in zip(actual.sessions, expected_sessions)
        )
        assert len(actual.weights) == len(expected_weights)
        assert all(actual_list == expected_list for actual_list, expected_list in zip(actual.weights, expected_weights))
        assert actual.extras is not None
        assert len(actual.extras["extra_column"]) == len(expected_extra_column)
        assert all(
            actual_list == expected_list
            for actual_list, expected_list in zip(actual.extras["extra_column"], expected_extra_column)
        )


class TestTransformerDataPreparatorBase:

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30", 0],
                [10, 11, 1, "2021-11-29", 2],
                [10, 12, 1, "2021-11-29", 3],
                [30, 11, 1, "2021-11-27", 4],
                [30, 12, 2, "2021-11-26", 1],
                [30, 15, 1, "2021-11-25", 0],
                [40, 11, 1, "2021-11-25", 1],
                [40, 17, 1, "2021-11-26", 1],
                [50, 16, 1, "2021-11-25", 2],
                [10, 14, 1, "2021-11-28", 2],
                [10, 16, 1, "2021-11-27", 1],
                [20, 13, 9, "2021-11-28", 1],
            ],
            columns=Columns.Interactions + ["extra_column"],
        )
        return interactions_df

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df, keep_extra_cols=True)

    @pytest.fixture
    def dataset_dense_item_features(self, interactions_df: pd.DataFrame) -> Dataset:
        item_features = pd.DataFrame(
            [
                [11, 1, 1],
                [12, 1, 2],
                [13, 1, 3],
                [14, 2, 1],
                [15, 2, 2],
                [16, 2, 2],
                [17, 2, 3],
            ],
            columns=[Columns.Item, "f1", "f2"],
        )
        ds = Dataset.construct(
            interactions_df,
            item_features_df=item_features,
            make_dense_item_features=True,
        )
        return ds

    @pytest.fixture
    def data_preparator(self) -> TransformerDataPreparatorBase:
        return TransformerDataPreparatorBase(
            session_max_len=4,
            batch_size=4,
            dataloader_num_workers=0,
        )

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([30, 40, 10]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 1, 1.0, "2021-11-25", 0],
                            [1, 2, 1.0, "2021-11-25", 1],
                            [0, 3, 2.0, "2021-11-26", 1],
                            [1, 4, 1.0, "2021-11-26", 1],
                            [0, 2, 1.0, "2021-11-27", 4],
                            [2, 5, 1.0, "2021-11-28", 2],
                            [2, 2, 1.0, "2021-11-29", 2],
                            [2, 3, 1.0, "2021-11-29", 3],
                            [2, 6, 1.0, "2021-11-30", 0],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime, "extra_column"],
                    ),
                ),
            ),
        ),
    )
    def test_process_dataset_train(
        self,
        dataset: Dataset,
        data_preparator: TransformerDataPreparatorBase,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.extra_cols = ["extra_column"]
        data_preparator.process_dataset_train(dataset)
        actual = data_preparator.train_dataset
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    def test_process_dataset_train_with_dense_item_features(
        self,
        dataset_dense_item_features: Dataset,
        data_preparator: TransformerDataPreparatorBase,
    ) -> None:
        data_preparator.process_dataset_train(dataset_dense_item_features)
        actual = data_preparator.train_dataset.item_features
        expected_values = np.array(
            [
                [0, 0],
                [2, 2],
                [1, 1],
                [1, 2],
                [2, 3],
                [2, 1],
                [1, 3],
            ],
            dtype=np.float32,
        )
        expected_names = ("f1", "f2")
        expected = DenseFeatures(expected_values, expected_names)
        assert_feature_set_equal(actual, expected)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([10, 20]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 6, 1.0, "2021-11-30", 0],
                            [0, 2, 1.0, "2021-11-29", 2],
                            [0, 3, 1.0, "2021-11-29", 3],
                            [0, 5, 1.0, "2021-11-28", 2],
                            [1, 6, 9.0, "2021-11-28", 1],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime, "extra_column"],
                    ),
                ),
            ),
        ),
    )
    def test_transform_dataset_u2i(
        self,
        dataset: Dataset,
        data_preparator: TransformerDataPreparatorBase,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.extra_cols = ["extra_column"]
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
        data_preparator: TransformerDataPreparatorBase,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        actual = data_preparator.transform_dataset_i2i(dataset)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)
