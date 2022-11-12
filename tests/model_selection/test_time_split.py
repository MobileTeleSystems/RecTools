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
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection import TimeRangeSplitter
from rectools.model_selection.time_split import DateRange, get_not_seen_mask

T = tp.TypeVar("T")
Converter = tp.Callable[[tp.Sequence[int]], tp.List[int]]


class TestTimeRangeSplit:
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
                [1, 1, 1, "2021-09-01"],
                [1, 2, 1, "2021-09-02"],
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
        return Interactions(df.iloc[shuffle_arr])

    @pytest.fixture
    def date_range(self) -> DateRange:
        return pd.date_range(date(2021, 9, 4), date(2021, 9, 6))

    @pytest.mark.parametrize("filter_cold_users", (True, False))
    @pytest.mark.parametrize("filter_cold_items", (True, False))
    @pytest.mark.parametrize("filter_already_seen", (True, False))
    @pytest.mark.parametrize(
        "date_range",
        (pd.date_range(date(2021, 9, 1), date(2021, 9, 1)), pd.date_range(date(2021, 8, 1), date(2021, 8, 10))),
    )
    def test_works_on_empty_range(
        self,
        interactions: Interactions,
        filter_cold_users: bool,
        filter_cold_items: bool,
        filter_already_seen: bool,
        date_range: pd.Series,
    ) -> None:
        trs = TimeRangeSplitter(
            date_range,
            filter_cold_users=filter_cold_users,
            filter_cold_items=filter_cold_items,
            filter_already_seen=filter_already_seen,
        )
        assert trs.get_n_splits(interactions) == 0
        assert list(trs.split(interactions)) == []

    def test_without_filtering(self, interactions: Interactions, date_range: DateRange, norm: Converter) -> None:
        interactions_copy = deepcopy(interactions)
        trs = TimeRangeSplitter(date_range, False, False, False)
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == norm([6, 7])
        assert actual[0][2] == {
            "Start date": pd.Timestamp("2021-09-04 00:00:00"),
            "End date": pd.Timestamp("2021-09-05 00:00:00"),
            "Train": 6,
            "Train users": 3,
            "Train items": 3,
            "Test": 2,
            "Test users": 2,
            "Test items": 2,
        }

        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8, 9])

    def test_filter_cold_users(self, interactions: Interactions, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplitter(
            date_range,
            filter_cold_users=True,
            filter_cold_items=False,
            filter_already_seen=False,
        )
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == norm([6, 7])
        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8])

    def test_filter_cold_items(self, interactions: Interactions, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplitter(
            date_range,
            filter_cold_users=False,
            filter_cold_items=True,
            filter_already_seen=False,
        )
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == norm([7])
        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8, 9])

    def test_filter_already_seen(self, interactions: Interactions, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplitter(
            date_range,
            filter_cold_users=False,
            filter_cold_items=False,
            filter_already_seen=True,
        )
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == norm([6])
        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8, 9])

    def test_filter_all(self, interactions: Interactions, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplitter(date_range)
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == []
        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8])


class TestGetNotSeenMask:
    @pytest.mark.parametrize(
        "train_users,train_items,test_users,test_items,expected",
        (
            ([], [], [], [], []),
            ([1, 2], [10, 20], [], [], []),
            ([], [], [1, 2], [10, 20], [True, True]),
            ([1, 2, 3, 4, 2, 3], [10, 20, 30, 40, 22, 30], [1, 2, 3, 2], [10, 20, 33, 20], [False, False, True, False]),
        ),
    )
    def test_correct(
        self,
        train_users: tp.List[int],
        train_items: tp.List[int],
        test_users: tp.List[int],
        test_items: tp.List[int],
        expected: tp.List[bool],
    ) -> None:
        actual = get_not_seen_mask(*(np.array(a) for a in (train_users, train_items, test_users, test_items)))
        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "train_users,train_items,test_users,test_items,expected_error_type",
        (
            ([1], [10, 20], [1], [10], ValueError),
            ([1], [10], [1, 2], [10], ValueError),
        ),
    )
    def test_with_incorrect_arrays(
        self,
        train_users: tp.List[int],
        train_items: tp.List[int],
        test_users: tp.List[int],
        test_items: tp.List[int],
        expected_error_type: tp.Type[Exception],
    ) -> None:
        with pytest.raises(expected_error_type):
            get_not_seen_mask(*(np.array(a) for a in (train_users, train_items, test_users, test_items)))
