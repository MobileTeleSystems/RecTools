#  Copyright 2022 MTS (Mobile Telesystems)
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
from scipy.sparse import csr_matrix

from rectools import Columns, ExternalIds
from rectools.dataset import IdMap, SparseFeatures
from rectools.metrics import (
    PairwiseDistanceCalculator,
    PairwiseHammingDistanceCalculator,
    SparsePairwiseHammingDistanceCalculator,
)


class TestPairwiseHammingDistanceCalculator:
    def test_correct_distance_values(self) -> None:
        features_df = pd.DataFrame(
            [
                ["i1", 0, 0],
                ["i2", 0, 1],
                ["i3", 1, 1],
                ["i4", 0, np.nan],
            ],
            columns=[Columns.Item, "feature_1", "feature_2"],
        ).set_index(Columns.Item)
        distance_calculator = PairwiseHammingDistanceCalculator(features_df)

        expected = np.array([0, 1, 2, np.nan, np.nan])

        with pytest.warns(UserWarning, match="Some items has absent feature values"):
            actual = distance_calculator[["i1", "i1", "i1", "i1", "i1"], ["i1", "i2", "i3", "i4", "i5"]]
        assert np.array_equal(actual, expected, equal_nan=True)


@pytest.mark.filterwarnings("ignore:Some items absent in mapper")
@pytest.mark.filterwarnings("ignore:Some items has absent feature values")
class TestSparsePairwiseHammingDistanceCalculator:
    @pytest.mark.parametrize(
        "left,right,expected",
        (
            # Correct features, mapper, item case
            (["i1", "i1", "i1"], ["i1", "i2", "i3"], [0, 1, 2]),
            # Features contain and not contain nan case
            (["i1", "i1", "i1", "i1"], ["i1", "i2", "i3", "i4"], [0, 1, 2, np.nan]),
            # Comparison absence item case
            (["i1", "i1", "i1", "i1"], ["i1", "i2", "i3", "i6"], [0, 1, 2, np.nan]),
            # Comparison empty items lists case
            ([], [], []),
            # IndexError case
            (["i1"], ["i5"], []),
        ),
    )
    def test_correct_distance_values(self, left: tp.List[str], right: tp.List[str], expected: tp.List[float]) -> None:
        dense_features = [
            [0, 0],  # i1
            [0, 1],  # i2
            [1, 1],  # i3
            [0, np.nan],  # i4
        ]
        mapper = IdMap.from_values(["i1", "i2", "i3", "i4", "i5"])
        sparse_features = SparseFeatures(values=csr_matrix(dense_features), names=(("f1", "v1"), ("f2", "v2")))
        distance_calculator = SparsePairwiseHammingDistanceCalculator(sparse_features, mapper)
        if "i5" not in right:
            actual = distance_calculator[left, right]
            assert np.array_equal(actual, expected, equal_nan=True)
        else:
            with pytest.raises(IndexError):
                distance_calculator[left, right]  # pylint: disable=pointless-statement


class DummyPairwiseDistanceCalculator(PairwiseDistanceCalculator):
    def _get_distances_for_item_pairs(self, items_0: ExternalIds, items_1: ExternalIds) -> np.ndarray:
        return np.zeros(len(items_0))


# pylint: disable=expression-not-assigned
class TestPairwiseDistanceCalculatorBase:
    def test_raises_when_get_distance_for_not_a_pairs_of_items(self) -> None:
        with pytest.raises(IndexError):
            DummyPairwiseDistanceCalculator()[["i1"], ["i2"], ["i3"]]  # type: ignore

    def test_raises_when_get_distance_for_not_a_sequence_of_items(self) -> None:
        with pytest.raises(TypeError):
            DummyPairwiseDistanceCalculator()["i1", "i2"]

    def test_raises_when_different_lengths_of_indices_lists_for_item_pairs(self) -> None:
        with pytest.raises(IndexError):
            DummyPairwiseDistanceCalculator()[["i1", "i2"], ["i3"]]
