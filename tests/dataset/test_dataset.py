#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

# pylint: disable=attribute-defined-outside-init

import typing as tp
from datetime import datetime

import pandas as pd
import pytest
from scipy import sparse

from rectools import Columns
from rectools.dataset import Dataset, DenseFeatures, Features, IdMap, Interactions, SparseFeatures
from tests.testing_utils import (
    assert_feature_set_equal,
    assert_id_map_equal,
    assert_interactions_set_equal,
    assert_sparse_matrix_equal,
)


class TestDataset:
    def setup_method(self) -> None:
        self.interactions_df = pd.DataFrame(
            [
                ["u1", "i1", 2, "2021-09-09"],
                ["u1", "i2", 2, "2021-09-05"],
                ["u1", "i1", 6, "2021-08-09"],
                ["u2", "i1", 7, "2020-09-09"],
                ["u2", "i5", 9, "2021-09-03"],
                ["u3", "i1", 2, "2021-09-09"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        )
        self.expected_user_id_map = IdMap.from_values(["u1", "u2", "u3"])
        self.expected_item_id_map = IdMap.from_values(["i1", "i2", "i5"])
        self.expected_interactions = Interactions(
            pd.DataFrame(
                [
                    [0, 0, 2.0, datetime(2021, 9, 9)],
                    [0, 1, 2.0, datetime(2021, 9, 5)],
                    [0, 0, 6.0, datetime(2021, 8, 9)],
                    [1, 0, 7.0, datetime(2020, 9, 9)],
                    [1, 2, 9.0, datetime(2021, 9, 3)],
                    [2, 0, 2.0, datetime(2021, 9, 9)],
                ],
                columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
            ),
        )

    def assert_dataset_equal_to_expected(
        self,
        actual: Dataset,
        expected_user_features: tp.Optional[Features],
        expected_item_features: tp.Optional[Features],
        expected_user_id_map: tp.Optional[IdMap] = None,
        expected_item_id_map: tp.Optional[IdMap] = None,
        expected_interactions: tp.Optional[Interactions] = None,
    ) -> None:
        expected_user_id_map = expected_user_id_map or self.expected_user_id_map
        expected_item_id_map = expected_item_id_map or self.expected_item_id_map
        expected_interactions = expected_interactions or self.expected_interactions

        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)
        assert_feature_set_equal(actual.user_features, expected_user_features)
        assert_feature_set_equal(actual.item_features, expected_item_features)

    def test_construct_without_features(self) -> None:
        dataset = Dataset.construct(self.interactions_df)
        self.assert_dataset_equal_to_expected(dataset, None, None)
        assert dataset.n_hot_users == 3
        assert dataset.n_hot_items == 3

    @pytest.mark.parametrize("user_id_col", ("id", Columns.User))
    @pytest.mark.parametrize("item_id_col", ("id", Columns.Item))
    def test_construct_with_features(self, user_id_col: str, item_id_col: str) -> None:
        user_features_df = pd.DataFrame(
            [
                ["u1", 77, 99],
                ["u2", 33, 55],
                ["u3", 22, 11],
            ],
            columns=[user_id_col, "f1", "f2"],
        )
        expected_user_features = DenseFeatures.from_dataframe(user_features_df, self.expected_user_id_map, user_id_col)
        item_features_df = pd.DataFrame(
            [
                ["i2", "f1", 3],
                ["i2", "f2", 20],
                ["i5", "f2", 20],
                ["i5", "f2", 30],
            ],
            columns=[item_id_col, "feature", "value"],
        )
        expected_item_features = SparseFeatures.from_flatten(
            item_features_df,
            self.expected_item_id_map,
            ["f2"],
            id_col=item_id_col,
        )
        dataset = Dataset.construct(
            self.interactions_df,
            user_features_df=user_features_df,
            make_dense_user_features=True,
            item_features_df=item_features_df,
            cat_item_features=["f2"],
        )
        self.assert_dataset_equal_to_expected(dataset, expected_user_features, expected_item_features)
        assert dataset.n_hot_users == 3
        assert dataset.n_hot_items == 3

        assert_feature_set_equal(dataset.get_hot_user_features(), expected_user_features)
        assert_feature_set_equal(dataset.get_hot_item_features(), expected_item_features)

    @pytest.mark.parametrize("user_id_col", ("id", Columns.User))
    @pytest.mark.parametrize("item_id_col", ("id", Columns.Item))
    def test_construct_with_features_with_warm_ids(self, user_id_col: str, item_id_col: str) -> None:
        user_features_df = pd.DataFrame(
            [
                ["u1", 77, 99],
                ["u2", 33, 55],
                ["u3", 22, 11],
                ["u4", 22, 11],
            ],
            columns=[user_id_col, "f1", "f2"],
        )
        expected_user_id_map = self.expected_user_id_map.add_ids(["u4"])
        expected_user_features = DenseFeatures.from_dataframe(user_features_df, expected_user_id_map, user_id_col)

        item_features_df = pd.DataFrame(
            [
                ["i2", "f1", 3],
                ["i2", "f2", 20],
                ["i5", "f2", 20],
                ["i5", "f2", 30],
                ["i7", "f2", 70],
            ],
            columns=[item_id_col, "feature", "value"],
        )
        expected_item_id_map = self.expected_item_id_map.add_ids(["i7"])
        expected_item_features = SparseFeatures.from_flatten(
            df=item_features_df,
            id_map=expected_item_id_map,
            cat_features=["f2"],
            id_col=item_id_col,
        )

        dataset = Dataset.construct(
            self.interactions_df,
            user_features_df=user_features_df,
            make_dense_user_features=True,
            item_features_df=item_features_df,
            cat_item_features=["f2"],
        )
        self.assert_dataset_equal_to_expected(
            dataset,
            expected_user_features,
            expected_item_features,
            expected_user_id_map,
            expected_item_id_map,
        )
        assert dataset.n_hot_users == 3
        assert dataset.n_hot_items == 3

        assert_feature_set_equal(dataset.get_hot_user_features(), expected_user_features.take([0, 1, 2]))
        assert_feature_set_equal(dataset.get_hot_item_features(), expected_item_features.take([0, 1, 2]))

    @pytest.mark.parametrize(
        "include_warm_users, include_warm_items, expected",
        (
            (False, False, [[1, 2], [1, 0]]),
            (True, False, [[1, 2], [1, 0], [0, 0]]),
            (False, True, [[1, 2, 0], [1, 0, 0]]),
            (True, True, [[1, 2, 0], [1, 0, 0], [0, 0, 0]]),
        ),
    )
    def test_get_user_item_matrix(
        self, include_warm_users: bool, include_warm_items: bool, expected: tp.List[tp.List[int]]
    ) -> None:
        interactions_df = pd.DataFrame(
            [
                ["u10", "i11", 1, "2021-09-09"],
                ["u10", "i12", 2, "2021-09-09"],
                ["u20", "i11", 1, "2021-09-05"],
            ],
            columns=Columns.Interactions,
        )
        user_features_df = pd.DataFrame(
            {
                Columns.User: ["u10", "u20", "u30", "u30"],
                "feature": ["f1", "f2", "f1", "f2"],
                "value": [1, 2, 1, 2],
            }
        )
        item_features_df = pd.DataFrame(
            {
                Columns.Item: ["i13"],
                "feature": ["feature"],
                "value": [100],
            }
        )
        dataset = Dataset.construct(
            interactions_df=interactions_df,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
        )
        user_item_matrix = dataset.get_user_item_matrix(
            include_warm_users=include_warm_users, include_warm_items=include_warm_items
        )
        expected_user_item_matrix = sparse.csr_matrix(expected)
        assert_sparse_matrix_equal(user_item_matrix, expected_user_item_matrix)

    @pytest.mark.parametrize(
        "include_warm_users, include_warm_items, expected",
        (
            (False, False, [[0, 0, 0], [1, 0, 5]]),
            (True, False, [[0, 0, 0], [1, 0, 5], [0, 0, 0]]),
            (False, True, [[0, 0, 0], [1, 0, 5]]),
            (True, True, [[0, 0, 0], [1, 0, 5], [0, 0, 0]]),
        ),
    )
    def test_get_user_item_matrix_for_extraordinary_dataset(
        self, include_warm_users: bool, include_warm_items: bool, expected: tp.List[tp.List[int]]
    ) -> None:
        user_id_map = IdMap.from_values(["u1", "u2", "u3"])
        item_id_map = IdMap.from_values(["i1", "i2", "i5"])
        interactions_df = pd.DataFrame(
            [
                ["u2", "i1", 1, "2021-09-09"],
                ["u2", "i5", 5, "2021-09-05"],
            ],
            columns=Columns.Interactions,
        )
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map)
        dataset = Dataset(user_id_map, item_id_map, interactions)
        user_item_matrix = dataset.get_user_item_matrix(
            include_warm_users=include_warm_users, include_warm_items=include_warm_items
        )
        expected_user_item_matrix = sparse.csr_matrix(expected)
        assert_sparse_matrix_equal(user_item_matrix, expected_user_item_matrix)

    @pytest.mark.parametrize("column", Columns.Interactions)
    def test_raises_when_no_columns_in_construct(self, column: str) -> None:
        with pytest.raises(KeyError) as e:
            Dataset.construct(self.interactions_df.drop(columns=column))
        err_text = e.value.args[0]
        assert column in err_text

    def test_raises_when_in_dense_features_absent_some_ids_that_present_in_interactions(self) -> None:
        user_features_df = pd.DataFrame(
            [
                ["u1", 77, 99],
                ["u2", 33, 55],
            ],
            columns=["user_id", "f1", "f2"],
        )
        with pytest.raises(ValueError, match=".+user.+all ids from interactions must be present in features table"):
            Dataset.construct(
                self.interactions_df,
                user_features_df=user_features_df,
                make_dense_user_features=True,
            )

    @pytest.mark.parametrize("include_weight", (True, False))
    @pytest.mark.parametrize("include_datetime", (True, False))
    def test_get_raw_interactions(self, include_weight: bool, include_datetime: bool) -> None:
        dataset = Dataset.construct(self.interactions_df)
        actual = dataset.get_raw_interactions(include_weight, include_datetime)
        expected = self.interactions_df.astype({Columns.Weight: "float64", Columns.Datetime: "datetime64[ns]"})
        if not include_weight:
            expected.drop(columns=Columns.Weight, inplace=True)
        if not include_datetime:
            expected.drop(columns=Columns.Datetime, inplace=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_rebuild_with_new_data_without_feature(self) -> None:
        dataset = Dataset.construct(self.interactions_df)
        new_interactions_df = pd.DataFrame(
            [
                ["u2", "i3", 5, "2021-09-03"],
                ["u4", "i1", 3, "2021-09-09"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        )
        new_dataset = dataset.rebuild_with_new_data(new_interactions_df)
        expected_user_id_map = IdMap.from_values(["u1", "u2", "u3", "u4"])
        expected_item_id_map = IdMap.from_values(["i1", "i2", "i5", "i3"])
        expected_interactions = Interactions(
            pd.DataFrame(
                [
                    [1, 3, 5.0, datetime(2021, 9, 3)],
                    [3, 0, 3.0, datetime(2021, 9, 9)],
                ],
                columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
            ),
        )

        self.assert_dataset_equal_to_expected(
            new_dataset,
            None,
            None,
            expected_item_id_map=expected_item_id_map,
            expected_user_id_map=expected_user_id_map,
            expected_interactions=expected_interactions,
        )

    def test_rebuild_with_new_data_with_feature(self) -> None:
        user_features_df = pd.DataFrame(
            [
                ["u1", 77, 99],
                ["u2", 33, 55],
                ["u3", 22, 11],
                ["u4", 22, 11],  # Warm user
            ],
            columns=[Columns.User, "f1", "f2"],
        )
        item_features_df = pd.DataFrame(
            [
                ["i2", "f1", 3],
                ["i2", "f2", 20],
                ["i5", "f2", 20],
                ["i5", "f2", 30],
                ["i7", "f2", 70],  # Warm item
            ],
            columns=[Columns.Item, "feature", "value"],
        )
        dataset = Dataset.construct(
            self.interactions_df,
            user_features_df=user_features_df,
            make_dense_user_features=True,
            item_features_df=item_features_df,
            cat_item_features=["f2"],
        )
        new_interactions_df = pd.DataFrame(
            [
                ["u2", "i8", 5, "2021-09-03"],  # Warm item in interactions
                ["u5", "i1", 3, "2021-09-09"],  # Warm user in interactions
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        )
        new_user_features_df = pd.DataFrame(
            [
                ["u1", 77, 99],
                ["u2", 33, 55],
                ["u3", 22, 11],
                ["u4", 22, 11],  # Warm user in old data
                ["u5", 55, 22],  # Warm user in new data
            ],
            columns=[Columns.User, "f1", "f2"],
        )
        new_item_features_df = pd.DataFrame(
            [
                ["i2", "f1", 3],
                ["i2", "f2", 20],
                ["i5", "f2", 20],
                ["i5", "f2", 30],
                ["i7", "f2", 70],  # Warm item in old data
                ["i8", "f2", 70],  # Warm item in new data
            ],
            columns=[Columns.Item, "feature", "value"],
        )
        new_dataset = dataset.rebuild_with_new_data(
            new_interactions_df,
            user_features_df=new_user_features_df,
            make_dense_user_features=True,
            item_features_df=new_item_features_df,
            cat_item_features=["f2"],
        )
        expected_user_id_map = IdMap.from_values(["u1", "u2", "u3", "u4", "u5"])
        expected_item_id_map = IdMap.from_values(["i1", "i2", "i5", "i7", "i8"])
        expected_interactions = Interactions(
            pd.DataFrame(
                [
                    [1, 4, 5.0, datetime(2021, 9, 3)],
                    [4, 0, 3.0, datetime(2021, 9, 9)],
                ],
                columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
            ),
        )

        expected_user_features = DenseFeatures.from_dataframe(new_user_features_df, expected_user_id_map, Columns.User)
        expected_item_features = SparseFeatures.from_flatten(
            new_item_features_df,
            expected_item_id_map,
            ["f2"],
            id_col=Columns.Item,
        )
        self.assert_dataset_equal_to_expected(
            new_dataset,
            expected_user_features,
            expected_item_features,
            expected_user_id_map,
            expected_item_id_map,
            expected_interactions,
        )
