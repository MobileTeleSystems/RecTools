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


class TestLastNSplitters:
    @pytest.fixture
    def shuffle_arr(self) -> np.ndarray:
        return np.arange(0, 9)

    @pytest.fixture
    def norm(self, shuffle_arr: np.ndarray) -> Converter:
        inv_shuffle_arr = np.zeros_like(shuffle_arr)
        inv_shuffle_arr[shuffle_arr] = np.arange(shuffle_arr.size)

        def _shuffle(values: tp.Sequence[int]) -> tp.List[int]:
            return sorted(inv_shuffle_arr[values])

        return _shuffle

    @pytest.fixture
    def interactions(self, shuffle_arr: np.ndarray) -> Interactions:
        df = pd.DataFrame(
            [
                [1, 1, 1, "2021-09-01"],
                [1, 2, 1, "2021-09-02"],
                [1, 1, 1, "2021-08-20"],
                [1, 2, 1, "2021-09-04"],
                [2, 1, 1, "2021-08-20"],
                [2, 2, 1, "2021-08-20"],
                [2, 3, 1, "2021-09-05"],
                [2, 2, 1, "2021-09-06"],
                [3, 1, 1, "2021-09-05"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]"})
        return Interactions(df.iloc[shuffle_arr])

    @pytest.fixture
    def n(self) -> tp.List[int]:
        return [1, 2]

    def test_without_filtering(self, interactions: Interactions, norm: Converter, n: tp.List[int]) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, False, False, False)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == norm([0, 1, 2, 4, 5, 6])
        assert sorted(actual[0][1]) == norm([3, 7, 8])

        assert sorted(actual[1][0]) == norm([0, 2, 4, 5])
        assert sorted(actual[1][1]) == norm([1, 3, 6, 7, 8])
        assert actual[1][2] == {
            "n": 2,
            "Train": 4,
            "Train users": 2,
            "Train items": 2,
            "Test": 5,
            "Test users": 3,
            "Test items": 3,
        }

    def test_filter_cold_users(self, interactions: Interactions, norm: Converter, n: tp.List[int]) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, True, False, False)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == norm([0, 1, 2, 4, 5, 6])
        assert sorted(actual[0][1]) == norm([3, 7])

        assert sorted(actual[1][0]) == norm([0, 2, 4, 5])
        assert sorted(actual[1][1]) == norm([1, 3, 6, 7])

    def test_filter_cold_items(self, interactions: Interactions, norm: Converter, n: tp.List[int]) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, False, True, False)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == norm([0, 1, 2, 4, 5, 6])
        assert sorted(actual[0][1]) == norm([3, 7, 8])

        assert sorted(actual[1][0]) == norm([0, 2, 4, 5])
        assert sorted(actual[1][1]) == norm([1, 3, 7, 8])

    def test_filter_already_seen(self, interactions: Interactions, norm: Converter, n: tp.List[int]) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, False, False, True)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == norm([0, 1, 2, 4, 5, 6])
        assert sorted(actual[0][1]) == norm([8])

        assert sorted(actual[1][0]) == norm([0, 2, 4, 5])
        assert sorted(actual[1][1]) == norm([1, 3, 6, 8])

    def test_filter_all(self, interactions: Interactions, norm: Converter, n: tp.List[int]) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, True, True, True)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 2
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == norm([0, 1, 2, 4, 5, 6])
        assert sorted(actual[0][1]) == []

        assert sorted(actual[1][0]) == norm([0, 2, 4, 5])
        assert sorted(actual[1][1]) == norm([1, 3])

    @pytest.mark.parametrize("filter_cold_users", (True, False))
    @pytest.mark.parametrize("filter_cold_items", (True, False))
    @pytest.mark.parametrize("filter_already_seen", (True, False))
    def test_int_and_array_n(
        self, interactions: Interactions, filter_cold_users: bool, filter_cold_items: bool, filter_already_seen: bool
    ) -> None:
        n_array = np.array([2])
        lns = LastNSplitter(n_array, filter_cold_users, filter_cold_items, filter_already_seen)
        actual1 = list(lns.split(interactions, collect_fold_stats=True))

        n_int = 2
        lns = LastNSplitter(n_int, filter_cold_users, filter_cold_items, filter_already_seen)
        actual2 = list(lns.split(interactions, collect_fold_stats=True))

        assert len(actual1) == len(actual2)
        assert len(actual1[0]) == len(actual2[0])
        assert np.array_equal(actual1[0][0], actual2[0][0])
        assert np.array_equal(actual1[0][1], actual2[0][1])

    @pytest.mark.parametrize(
        "new_index", (
            ([0, 11, 11, 11, 4, 5, 16, 7, 11]),
            ([0, 11, 2, -3, -4, -5, 16, 7, 1]),
        ),
    )
    def test_complicated_index(self, interactions: Interactions, new_index: tp.List[int]) -> None:
        interactions_new_index = deepcopy(interactions)
        interactions_new_index.df.index = new_index
        lns = LastNSplitter(2, False, False, False)
        actual1 = list(lns.split(interactions))
        actual2 = list(lns.split(interactions_new_index))

        assert len(actual1) == len(actual2)
        assert len(actual1[0]) == len(actual2[0])

        assert sorted(actual1[0][0]) == sorted(actual2[0][0])
        assert sorted(actual1[0][1]) == sorted(actual2[0][1])

    @pytest.mark.parametrize(
        "n, expected_error_type, err_message",
        (
            (0, ValueError, "N must be positive, got 0"),
            (-1, ValueError, "N must be positive, got -1"),
            ([1, 0], ValueError, "N must be positive, got 0"),
            ([-1], ValueError, "N must be positive, got -1"),
        ),
    )
    def test_negative_n(
        self,
        interactions: Interactions,
        n: tp.Union[int, tp.Iterable[int]],
        expected_error_type: tp.Type[Exception],
        err_message: str,
    ) -> None:
        lns = LastNSplitter(n, False, False, False)
        with pytest.raises(expected_error_type, match=err_message):
            for _, _, _ in lns.split(interactions):
                pass
