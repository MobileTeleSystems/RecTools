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
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection import TimeRangeSplitter

T = tp.TypeVar("T")
Converter = tp.Callable[[tp.Sequence[int]], tp.List[int]]


class TestTimeRangeSplitter:
    @pytest.fixture
    def shuffle_arr(self) -> np.ndarray:
        return np.random.choice(np.arange(11), 11, replace=False)

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
                [1, 1, 1, datetime(2021, 9, 1, 18, 5)],  # 0
                [1, 2, 1, datetime(2021, 9, 2, 18, 5)],  # 1
                [2, 1, 1, datetime(2021, 9, 2, 18, 5)],  # 2
                [2, 2, 1, datetime(2021, 9, 3, 18, 5)],  # 3
                [3, 2, 1, datetime(2021, 9, 3, 18, 5)],  # 4
                [3, 3, 1, datetime(2021, 9, 3, 18, 5)],  # 5
                [3, 4, 1, datetime(2021, 9, 4, 18, 5)],  # 6
                [1, 2, 1, datetime(2021, 9, 4, 18, 5)],  # 7
                [3, 1, 1, datetime(2021, 9, 5, 18, 5)],  # 8
                [4, 2, 1, datetime(2021, 9, 5, 18, 5)],  # 9
                [3, 3, 1, datetime(2021, 9, 6, 18, 5)],  # 10
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]"})
        return Interactions(df.iloc[shuffle_arr])

    def test_without_filtering(self, interactions: Interactions, norm: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        splitter = TimeRangeSplitter("2D", 2, False, False, False)
        actual = list(splitter.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm([0, 1, 2])
        assert sorted(actual[0][1]) == norm([3, 4, 5, 6, 7])
        assert actual[0][2] == {
            "i_split": 0,
            "start": pd.Timestamp("2021-09-03 00:00:00", freq="2D"),
            "end": pd.Timestamp("2021-09-05 00:00:00", freq="2D"),
            "train": 3,
            "train_users": 2,
            "train_items": 2,
            "test": 5,
            "test_users": 3,
            "test_items": 3,
        }

        assert sorted(actual[1][0]) == norm([0, 1, 2, 3, 4, 5, 6, 7])
        assert sorted(actual[1][1]) == norm([8, 9, 10])

    def test_filter_cold_users(self, interactions: Interactions, norm: Converter) -> None:
        splitter = TimeRangeSplitter(
            "2D",
            2,
            filter_cold_users=True,
            filter_cold_items=False,
            filter_already_seen=False,
        )
        actual = list(splitter.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm([0, 1, 2])
        assert sorted(actual[0][1]) == norm([3, 7])
        assert sorted(actual[1][0]) == norm([0, 1, 2, 3, 4, 5, 6, 7])
        assert sorted(actual[1][1]) == norm([8, 10])

    def test_filter_cold_items(self, interactions: Interactions, norm: Converter) -> None:
        splitter = TimeRangeSplitter(
            "2D",
            2,
            filter_cold_users=False,
            filter_cold_items=True,
            filter_already_seen=False,
        )
        actual = list(splitter.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm([0, 1, 2])
        assert sorted(actual[0][1]) == norm([3, 4, 7])
        assert sorted(actual[1][0]) == norm([0, 1, 2, 3, 4, 5, 6, 7])
        assert sorted(actual[1][1]) == norm([8, 9, 10])

    def test_filter_already_seen(self, interactions: Interactions, norm: Converter) -> None:
        splitter = TimeRangeSplitter(
            "2D",
            2,
            filter_cold_users=False,
            filter_cold_items=False,
            filter_already_seen=True,
        )
        actual = list(splitter.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm([0, 1, 2])
        assert sorted(actual[0][1]) == norm([3, 4, 5, 6])
        assert sorted(actual[1][0]) == norm([0, 1, 2, 3, 4, 5, 6, 7])
        assert sorted(actual[1][1]) == norm([8, 9])

    def test_filter_all(self, interactions: Interactions, norm: Converter) -> None:
        splitter = TimeRangeSplitter(
            "2D",
            2,
            filter_cold_users=True,
            filter_cold_items=True,
            filter_already_seen=True,
        )
        actual = list(splitter.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm([0, 1, 2])
        assert sorted(actual[0][1]) == norm([3])
        assert sorted(actual[1][0]) == norm([0, 1, 2, 3, 4, 5, 6, 7])
        assert sorted(actual[1][1]) == norm([8])

    def test_hour_interval(self) -> None:
        df = pd.DataFrame(
            [
                [1, 1, 1, datetime(2021, 9, 1, 18, 5)],
                [1, 1, 1, datetime(2021, 9, 1, 18, 55)],
                [1, 1, 1, datetime(2021, 9, 1, 22, 15)],
                [1, 1, 1, datetime(2021, 9, 1, 23, 5)],
            ],
            columns=Columns.Interactions,
        ).astype({Columns.Datetime: "datetime64[ns]"})
        interactions = Interactions(df)
        splitter = TimeRangeSplitter("2H", 2, False, False, False)
        actual = list(splitter.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == [0, 1]
        assert sorted(actual[0][1]) == []
        assert sorted(actual[1][0]) == [0, 1]
        assert sorted(actual[1][1]) == [2, 3]

    @pytest.mark.parametrize("test_size", ("5a", "5h", "5W", "0D", "01D", "-5D", "D", "5"))
    def test_incorrect_test_size(self, test_size: str) -> None:
        with pytest.raises(ValueError):
            TimeRangeSplitter(test_size)

    def test_dt_on_units_border(self) -> None:
        df = pd.DataFrame(
            [
                [1, 1, 1, "2021-09-01"],
                [1, 1, 1, "2021-09-02"],
                [1, 1, 1, "2021-09-03"],
            ],
            columns=Columns.Interactions,
        ).astype({Columns.Datetime: "datetime64[ns]"})
        interactions = Interactions(df)
        splitter = TimeRangeSplitter("1D", 1, False, False, False)
        actual = list(splitter.split(interactions))
        assert len(actual) == 1
        assert sorted(actual[0][0]) == [0, 1]
        assert sorted(actual[0][1]) == [2]
