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

import typing as tp
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection import LastNSplitter

Converter = tp.Callable[[tp.Sequence[int]], tp.List[int]]


class TestLastNSplitter:
    @pytest.fixture
    def shuffle_arr(self) -> np.ndarray:
        return np.random.choice(np.arange(9), 9, replace=False)

    @pytest.fixture
    def to_shuffled(self, shuffle_arr: np.ndarray) -> Converter:
        inv_shuffle_arr = np.zeros_like(shuffle_arr)
        inv_shuffle_arr[shuffle_arr] = np.arange(shuffle_arr.size)

        def _shuffle(values: tp.Sequence[int]) -> tp.List[int]:
            return sorted(inv_shuffle_arr[values])

        return _shuffle

    @pytest.fixture
    def interactions(self, shuffle_arr: np.ndarray) -> Interactions:
        df = pd.DataFrame(
            [
                [1, 1, 1, "2021-09-01"],  # 0
                [1, 2, 1, "2021-09-02"],  # 1
                [1, 1, 1, "2021-09-03"],  # 2
                [1, 2, 1, "2021-09-04"],  # 3
                [1, 3, 1, "2021-09-05"],  # 4
                [2, 2, 1, "2021-08-20"],  # 5
                [2, 3, 1, "2021-09-05"],  # 6
                [2, 2, 1, "2021-09-06"],  # 7
                [3, 1, 1, "2021-09-05"],  # 8
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]"})
        return Interactions(df.iloc[shuffle_arr])

    def test_without_filtering(self, interactions: Interactions, to_shuffled: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        splitter = LastNSplitter(2, 2, False, False, False)
        actual = list(splitter.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3
        assert len(actual[1]) == 3

        assert sorted(actual[0][0]) == to_shuffled([0])
        assert sorted(actual[0][1]) == to_shuffled([1, 2, 5])

        assert sorted(actual[1][0]) == to_shuffled([0, 1, 2, 5])
        assert sorted(actual[1][1]) == to_shuffled([3, 4, 6, 7, 8])

        assert actual[0][2] == {
            "i_split": 1,
            "train": 1,
            "train_users": 1,
            "train_items": 1,
            "test": 3,
            "test_users": 2,
            "test_items": 2,
        }

    def test_filter_cold_users(self, interactions: Interactions, to_shuffled: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        splitter = LastNSplitter(2, 2, True, False, False)
        actual = list(splitter.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3
        assert len(actual[1]) == 3

        assert sorted(actual[0][0]) == to_shuffled([0])
        assert sorted(actual[0][1]) == to_shuffled([1, 2])

        assert sorted(actual[1][0]) == to_shuffled([0, 1, 2, 5])
        assert sorted(actual[1][1]) == to_shuffled([3, 4, 6, 7])

    def test_filter_cold_items(self, interactions: Interactions, to_shuffled: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        splitter = LastNSplitter(2, 2, False, True, False)
        actual = list(splitter.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3
        assert len(actual[1]) == 3

        assert sorted(actual[0][0]) == to_shuffled([0])
        assert sorted(actual[0][1]) == to_shuffled([2])

        assert sorted(actual[1][0]) == to_shuffled([0, 1, 2, 5])
        assert sorted(actual[1][1]) == to_shuffled([3, 7, 8])

    def test_filter_already_seen(self, interactions: Interactions, to_shuffled: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        splitter = LastNSplitter(2, 2, False, False, True)
        actual = list(splitter.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3
        assert len(actual[1]) == 3

        assert sorted(actual[0][0]) == to_shuffled([0])
        assert sorted(actual[0][1]) == to_shuffled([1, 5])

        assert sorted(actual[1][0]) == to_shuffled([0, 1, 2, 5])
        assert sorted(actual[1][1]) == to_shuffled([4, 6, 8])

    def test_filter_all(self, interactions: Interactions, to_shuffled: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        splitter = LastNSplitter(2, 2, True, True, True)
        actual = list(splitter.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3
        assert len(actual[1]) == 3

        assert sorted(actual[0][0]) == to_shuffled([0])
        assert sorted(actual[0][1]) == to_shuffled([])

        assert sorted(actual[1][0]) == to_shuffled([0, 1, 2, 5])
        assert sorted(actual[1][1]) == to_shuffled([])
