#  Copyright 2023 MTS (Mobile Telesystems)
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

from rectools import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.model_selection import Splitter


class TestSplitter:
    @pytest.fixture
    def interactions(self) -> Interactions:
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
        return Interactions(df)

    @pytest.fixture
    def dataset(self, interactions: Interactions) -> Dataset:
        item_id_map = IdMap.from_values([10, 20, 30, 40, 50])
        user_id_map = IdMap.from_values([10, 11, 12, 13, 14])
        return Dataset(user_id_map, item_id_map, interactions)

    def test_not_implemented(self, interactions: Interactions) -> None:
        s = Splitter()
        with pytest.raises(NotImplementedError):
            for _, _, _ in s.split(interactions):
                pass

    @pytest.mark.parametrize("collect_fold_stats", [False, True])
    def test_not_defined_fields(self, interactions: Interactions, collect_fold_stats: bool) -> None:
        s = Splitter()
        train_idx = np.array([1, 2, 3, 5, 7, 8])
        test_idx = np.array([4, 6, 9, 10])
        fold_info = {"info_from_split": 123}
        train_idx_new, test_idx_new, _ = s.filter(interactions, collect_fold_stats, train_idx, test_idx, fold_info)

        assert np.array_equal(train_idx, train_idx_new)
        assert sorted(test_idx_new) == [4]

    @pytest.mark.parametrize("prefer_warm_inference_over_cold", (True, False))
    @pytest.mark.parametrize(
        "keep_external_ids, expected_external_item_ids, expected_external_user_ids",
        ((True, np.array([10, 30, 20]), np.array([10, 14, 12])), (False, np.array([0, 2, 1]), np.array([0, 4, 2]))),
    )
    def test_get_train_dataset_without_features(
        self,
        dataset: Dataset,
        prefer_warm_inference_over_cold: bool,
        keep_external_ids: bool,
        expected_external_item_ids: np.ndarray,
        expected_external_user_ids: np.ndarray,
    ) -> None:
        train_ids = np.arange(4)
        train_dataset = Splitter.get_train_dataset(
            dataset,
            train_ids,
            keep_external_ids=keep_external_ids,
            prefer_warm_inference_over_cold=prefer_warm_inference_over_cold,
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
        np.testing.assert_equal(train_dataset.user_id_map.external_ids, expected_external_user_ids)
        np.testing.assert_equal(train_dataset.item_id_map.external_ids, expected_external_item_ids)
        pd.testing.assert_frame_equal(train_dataset.interactions.df, expected_interactions_2x_internal_df)
        assert train_dataset.user_features is None
        assert train_dataset.item_features is None


# TODO: add tests with features
# There are old tests for old func:

# class TestGen2xInternalIdsDataset:
#     def setup_method(self) -> None:
#         self.interactions_internal_df = pd.DataFrame(
#             [
#                 [0, 0, 1, 101],
#                 [0, 1, 1, 102],
#                 [0, 0, 1, 103],
#                 [3, 0, 1, 101],
#                 [3, 2, 1, 102],
#             ],
#             columns=Columns.Interactions,
#         ).astype({Columns.Datetime: "datetime64[ns]", Columns.Weight: float})

#         self.expected_interactions_2x_internal_df = pd.DataFrame(
#             [
#                 [0, 0, 1, 101],
#                 [0, 1, 1, 102],
#                 [0, 0, 1, 103],
#                 [1, 0, 1, 101],
#                 [1, 2, 1, 102],
#             ],
#             columns=Columns.Interactions,
#         ).astype({Columns.Datetime: "datetime64[ns]", Columns.Weight: float})

#     @pytest.mark.parametrize(
#         "prefer_warm_inference_over_cold, expected_user_ids, expected_item_ids",
#         (
#             (False, [0, 3], [0, 1, 2]),
#             (True, [0, 3, 1, 2], [0, 1, 2, 3]),
#         ),
#     )
#     def test_with_features(
#         self, prefer_warm_inference_over_cold: bool, expected_user_ids: tp.List[int], expected_item_ids: tp.List[int]
#     ) -> None:
#         user_features = DenseFeatures(
#             values=np.array([[1, 10], [2, 20], [3, 30], [4, 40]]),
#             names=("f1", "f2"),
#         )
#         item_features = SparseFeatures(
#             values=sparse.csr_matrix(
#                 [
#                     [3.2, 0, 1],
#                     [2.4, 2, 0],
#                     [0.0, 0, 1],
#                     [1.0, 5, 1],
#                 ],
#             ),
#             names=(("f1", None), ("f2", 100), ("f2", 200)),
#         )

#         dataset = _gen_2x_internal_ids_dataset(
#             self.interactions_internal_df, user_features, item_features, prefer_warm_inference_over_cold
#         )

#         np.testing.assert_equal(dataset.user_id_map.external_ids, np.array(expected_user_ids))
#         np.testing.assert_equal(dataset.item_id_map.external_ids, np.array(expected_item_ids))
#         pd.testing.assert_frame_equal(dataset.interactions.df, self.expected_interactions_2x_internal_df)

#         assert dataset.user_features is not None and dataset.item_features is not None  # for mypy
#         np.testing.assert_equal(dataset.user_features.values, user_features.values[expected_user_ids])
#         assert dataset.user_features.names == user_features.names
#         assert_sparse_matrix_equal(dataset.item_features.values, item_features.values[expected_item_ids])
#         assert dataset.item_features.names == item_features.names
