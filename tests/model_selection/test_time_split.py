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
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.model_selection import TimeRangeSplit
from rectools.model_selection.time_split import DateRange

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
    def interactions(self, shuffle_arr: np.ndarray) -> pd.DataFrame:
        df = (
            pd.DataFrame(
                [
                    ["i5", 1, 1, "2021-09-01"],
                    ["i5", 1, 2, "2021-09-02"],
                    ["i4", 2, 1, "2021-09-02"],
                    ["i3", 2, 2, "2021-09-03"],
                    ["i2", 3, 2, "2021-09-03"],
                    ["i1", 3, 3, "2021-09-03"],
                    ["i0", 3, 4, "2021-09-04"],
                    ["i7", 1, 2, "2021-09-04"],
                    ["i8", 3, 1, "2021-09-05"],
                    ["i9", 4, 2, "2021-09-05"],
                    ["i9", 3, 3, "2021-09-06"],
                ],
                columns=["index", Columns.User, Columns.Item, Columns.Datetime],
            )
            .astype({Columns.Datetime: "datetime64[ns]"})
            .set_index("index")
        )
        return df.iloc[shuffle_arr]

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
        interactions: pd.DataFrame,
        filter_cold_users: bool,
        filter_cold_items: bool,
        filter_already_seen: bool,
        date_range: pd.Series,
    ) -> None:
        trs = TimeRangeSplit(
            date_range,
            filter_cold_users=filter_cold_users,
            filter_cold_items=filter_cold_items,
            filter_already_seen=filter_already_seen,
        )
        assert trs.get_n_splits(interactions) == 0
        assert list(trs.split(interactions)) == []

    def test_without_filtering(self, interactions: pd.DataFrame, date_range: DateRange, norm: Converter) -> None:
        interactions_copy = interactions.copy()
        trs = TimeRangeSplit(date_range, False, False, False)
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions, interactions_copy)
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == norm([6, 7])
        assert actual[0][2] == {
            "Start date": pd.Timestamp("2021-09-04 00:00:00", freq="D"),
            "End date": pd.Timestamp("2021-09-05 00:00:00", freq="D"),
            "Train": 6,
            "Train users": 3,
            "Train items": 3,
            "Test": 2,
            "Test users": 2,
            "Test items": 2,
        }

        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8, 9])

    def test_filter_cold_users(self, interactions: pd.DataFrame, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplit(
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

    def test_filter_cold_items(self, interactions: pd.DataFrame, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplit(
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

    def test_filter_already_seen(self, interactions: pd.DataFrame, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplit(
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

    def test_filter_all(self, interactions: pd.DataFrame, date_range: DateRange, norm: Converter) -> None:
        trs = TimeRangeSplit(date_range)
        assert trs.get_n_splits(interactions) == 2
        actual = list(trs.split(interactions))
        assert len(actual) == 2

        assert sorted(actual[0][0]) == norm(range(6))
        assert sorted(actual[0][1]) == []
        assert sorted(actual[1][0]) == norm(range(8))
        assert sorted(actual[1][1]) == norm([8])

    @pytest.mark.parametrize("column", (Columns.User, Columns.Item, Columns.Datetime))
    def test_raises_when_column_absent(self, interactions: pd.DataFrame, date_range: DateRange, column: str) -> None:
        trs = TimeRangeSplit(date_range)
        with pytest.raises(KeyError) as e:
            next(iter(trs.split(interactions.drop(columns=column))))
        err_text = e.value.args[0]
        assert column in err_text
