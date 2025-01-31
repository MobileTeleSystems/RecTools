#  Copyright 2022-2025 MTS (Mobile Telesystems)
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

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rectools import Columns
from rectools.dataset import Dataset, DenseFeatures, Features, IdMap, Interactions, SparseFeatures
from rectools.dataset.features import DIRECT_FEATURE_VALUE
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
                ["u1", "i1", 2, "2021-09-09", 5],
                ["u1", "i2", 2, "2021-09-05", 6],
                ["u1", "i1", 6, "2021-08-09", 7],
                ["u2", "i1", 7, "2020-09-09", 8],
                ["u2", "i5", 9, "2021-09-03", 9],
                ["u3", "i1", 2, "2021-09-09", 10],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime, "extra_col"],
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
        self.expected_schema = {
            "n_interactions": 6,
            "users": {
                "n_hot": 3,
                "id_map": {
                    "external_ids": ["u1", "u2", "u3"],
                    "dtype": "|O",
                },
                "features": None,
            },
            "items": {
                "n_hot": 3,
                "id_map": {
                    "external_ids": ["i1", "i2", "i5"],
                    "dtype": "|O",
                },
                "features": None,
            },
        }

    def assert_dataset_equal_to_expected(
        self,
        actual: Dataset,
        expected_user_features: tp.Optional[Features],
        expected_item_features: tp.Optional[Features],
        expected_user_id_map: tp.Optional[IdMap] = None,
        expected_item_id_map: tp.Optional[IdMap] = None,
    ) -> None:
        expected_user_id_map = expected_user_id_map or self.expected_user_id_map
        expected_item_id_map = expected_item_id_map or self.expected_item_id_map

        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, self.expected_interactions)
        assert_feature_set_equal(actual.user_features, expected_user_features)
        assert_feature_set_equal(actual.item_features, expected_item_features)

    def test_construct_with_extra_cols(self) -> None:

        dataset = Dataset.construct(self.interactions_df, keep_extra_cols=True)
        actual = dataset.interactions
        expected = self.expected_interactions
        expected.df["extra_col"] = self.interactions_df["extra_col"]
        assert_interactions_set_equal(actual, expected)
        actual_schema = dataset.get_schema(add_item_id_map=True, add_user_id_map=True)
        assert actual_schema == self.expected_schema

    def test_construct_without_features(self) -> None:
        dataset = Dataset.construct(self.interactions_df)
        self.assert_dataset_equal_to_expected(dataset, None, None)
        assert dataset.n_hot_users == 3
        assert dataset.n_hot_items == 3
        actual_schema = dataset.get_schema(add_item_id_map=True, add_user_id_map=True)
        assert actual_schema == self.expected_schema

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

        expected_schema = {
            "n_interactions": 6,
            "users": {
                "n_hot": 3,
                "id_map": {
                    "external_ids": ["u1", "u2", "u3"],
                    "dtype": "|O",
                },
                "features": {
                    "dense": True,
                    "names": ["f1", "f2"],
                    "cat_cols": None,
                    "cat_n_stored_values": None,
                },
            },
            "items": {
                "n_hot": 3,
                "id_map": {
                    "external_ids": ["i1", "i2", "i5"],
                    "dtype": "|O",
                },
                "features": {
                    "dense": False,
                    "names": [["f1", DIRECT_FEATURE_VALUE], ["f2", 20], ["f2", 30]],
                    "cat_cols": [1, 2],
                    "cat_n_stored_values": 3,
                },
            },
        }
        actual_schema = dataset.get_schema(add_item_id_map=True, add_user_id_map=True)
        assert actual_schema == expected_schema

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
    @pytest.mark.parametrize("keep_extra_cols", (True, False))
    @pytest.mark.parametrize("include_extra_cols", (True, False))
    def test_get_raw_interactions(
        self, include_weight: bool, include_datetime: bool, keep_extra_cols: bool, include_extra_cols: bool
    ) -> None:
        dataset = Dataset.construct(self.interactions_df, keep_extra_cols=keep_extra_cols)
        actual = dataset.get_raw_interactions(include_weight, include_datetime, include_extra_cols)
        expected = self.interactions_df.astype({Columns.Weight: "float64", Columns.Datetime: "datetime64[ns]"})
        if not include_weight:
            expected.drop(columns=Columns.Weight, inplace=True)
        if not include_datetime:
            expected.drop(columns=Columns.Datetime, inplace=True)
        if not keep_extra_cols or not include_extra_cols:
            expected.drop(columns="extra_col", inplace=True)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.fixture
    def dataset_to_filter(self) -> Dataset:
        item_id_map = IdMap.from_values([10, 20, 30, 40, 50])
        user_id_map = IdMap.from_values([10, 11, 12, 13, 14])
        df = pd.DataFrame(
            [
                [0, 0, 1, "2021-09-01"],
                [4, 2, 1, "2021-09-02"],
                [2, 1, 1, "2021-09-02"],
                [2, 2, 1, "2021-09-03"],
                [3, 2, 1, "2021-09-03"],
                [3, 3, 1, "2021-09-03"],
                [3, 4, 1, "2021-09-04"],
                [1, 2, 1, "2021-09-04"],
                [3, 1, 1, "2021-09-05"],
                [4, 2, 1, "2021-09-05"],
                [3, 3, 1, "2021-09-06"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]"})
        interactions = Interactions(df)
        return Dataset(user_id_map, item_id_map, interactions)

    @pytest.fixture
    def dataset_with_features_to_filter(self, dataset_to_filter: Dataset) -> Dataset:
        user_features = DenseFeatures(
            values=np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]),
            names=("f1", "f2"),
        )
        item_features = SparseFeatures(
            values=sparse.csr_matrix(
                [
                    [3.2, 0, 1],
                    [2.4, 2, 0],
                    [0.0, 0, 1],
                    [1.0, 5, 1],
                    [2.0, 1, 1],
                ],
            ),
            names=(("f1", None), ("f2", 100), ("f2", 200)),
        )
        return Dataset(
            dataset_to_filter.user_id_map,
            dataset_to_filter.item_id_map,
            dataset_to_filter.interactions,
            user_features,
            item_features,
        )

    @pytest.mark.parametrize("keep_features_for_removed_entities", (True, False))
    @pytest.mark.parametrize(
        "keep_external_ids, expected_external_item_ids, expected_external_user_ids",
        ((True, np.array([10, 30, 20]), np.array([10, 14, 12])), (False, np.array([0, 2, 1]), np.array([0, 4, 2]))),
    )
    def test_filter_dataset_interactions_df_rows_without_features(
        self,
        dataset_to_filter: Dataset,
        keep_features_for_removed_entities: bool,
        keep_external_ids: bool,
        expected_external_item_ids: np.ndarray,
        expected_external_user_ids: np.ndarray,
    ) -> None:
        rows_to_keep = np.arange(4)
        filtered_dataset = dataset_to_filter.filter_interactions(
            rows_to_keep,
            keep_external_ids=keep_external_ids,
            keep_features_for_removed_entities=keep_features_for_removed_entities,
        )
        expected_interactions_2x_internal_df = pd.DataFrame(
            [
                [0, 0, 1, "2021-09-01"],
                [1, 1, 1, "2021-09-02"],
                [2, 2, 1, "2021-09-02"],
                [2, 1, 1, "2021-09-03"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]", Columns.Weight: float})
        np.testing.assert_equal(filtered_dataset.user_id_map.external_ids, expected_external_user_ids)
        np.testing.assert_equal(filtered_dataset.item_id_map.external_ids, expected_external_item_ids)
        pd.testing.assert_frame_equal(filtered_dataset.interactions.df, expected_interactions_2x_internal_df)
        assert filtered_dataset.user_features is None
        assert filtered_dataset.item_features is None

    @pytest.mark.parametrize(
        "keep_external_ids, keep_features_for_removed_entities, expected_external_item_ids, expected_external_user_ids",
        (
            (True, False, np.array([10, 30, 20]), np.array([10, 14, 12])),
            (False, False, np.array([0, 2, 1]), np.array([0, 4, 2])),
            (True, True, np.array([10, 30, 20, 40, 50]), np.array([10, 14, 12, 11, 13])),
            (False, True, np.array([0, 2, 1, 3, 4]), np.array([0, 4, 2, 1, 3])),
        ),
    )
    def test_filter_dataset_interactions_df_rows_with_features(
        self,
        dataset_with_features_to_filter: Dataset,
        keep_features_for_removed_entities: bool,
        keep_external_ids: bool,
        expected_external_item_ids: np.ndarray,
        expected_external_user_ids: np.ndarray,
    ) -> None:
        rows_to_keep = np.arange(4)
        filtered_dataset = dataset_with_features_to_filter.filter_interactions(
            rows_to_keep,
            keep_external_ids=keep_external_ids,
            keep_features_for_removed_entities=keep_features_for_removed_entities,
        )
        expected_interactions_2x_internal_df = pd.DataFrame(
            [
                [0, 0, 1, "2021-09-01"],
                [1, 1, 1, "2021-09-02"],
                [2, 2, 1, "2021-09-02"],
                [2, 1, 1, "2021-09-03"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]", Columns.Weight: float})
        np.testing.assert_equal(filtered_dataset.user_id_map.external_ids, expected_external_user_ids)
        np.testing.assert_equal(filtered_dataset.item_id_map.external_ids, expected_external_item_ids)
        pd.testing.assert_frame_equal(filtered_dataset.interactions.df, expected_interactions_2x_internal_df)

        # Check features
        old_user_features = dataset_with_features_to_filter.user_features
        old_item_features = dataset_with_features_to_filter.item_features
        new_user_features = filtered_dataset.user_features
        new_item_features = filtered_dataset.item_features
        assert new_user_features is not None and new_item_features is not None  # for mypy
        assert old_user_features is not None and old_item_features is not None  # for mypy

        kept_internal_user_ids = (
            dataset_with_features_to_filter.user_id_map.convert_to_internal(expected_external_user_ids)
            if keep_external_ids
            else expected_external_user_ids
        )
        kept_internal_item_ids = (
            dataset_with_features_to_filter.item_id_map.convert_to_internal(expected_external_item_ids)
            if keep_external_ids
            else expected_external_item_ids
        )
        np.testing.assert_equal(new_user_features.values, old_user_features.values[kept_internal_user_ids])
        assert new_user_features.names == old_user_features.names
        assert_sparse_matrix_equal(new_item_features.values, old_item_features.values[kept_internal_item_ids])
        assert new_item_features.names == old_item_features.names
