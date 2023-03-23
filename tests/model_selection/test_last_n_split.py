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
from rectools.model_selection.last_n_split import get_not_seen_mask


class TestLastNSplitters:
    @pytest.fixture
    def interactions(self) -> Interactions:
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
        return Interactions(df)

    @pytest.fixture
    def n(self) -> int:
        return 2

    def test_without_filtering(self, interactions: Interactions, n: int) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, False, False, False)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 1
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == [0, 2, 4, 5]
        assert sorted(actual[0][1]) == [1, 3, 6, 7, 8]
        assert actual[0][2] == {
            "Train": 4,
            "Train users": 2,
            "Train items": 2,
            "Test": 5,
            "Test users": 3,
            "Test items": 3,
        }

    def test_filter_cold_users(self, interactions: Interactions, n: int) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, True, False, False)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 1
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == [0, 2, 4, 5]
        assert sorted(actual[0][1]) == [1, 3, 6, 7]

    def test_filter_cold_items(self, interactions: Interactions, n: int) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, False, True, False)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 1
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == [0, 2, 4, 5]
        assert sorted(actual[0][1]) == [1, 3, 7, 8]

    def test_filter_already_seen(self, interactions: Interactions, n: int) -> None:
        interactions_copy = deepcopy(interactions)
        lns = LastNSplitter(n, False, False, True)
        actual = list(lns.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert len(actual) == 1
        assert len(actual[0]) == 3

        assert sorted(actual[0][0]) == [0, 2, 4, 5]
        assert sorted(actual[0][1]) == [1, 3, 6, 8]


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
