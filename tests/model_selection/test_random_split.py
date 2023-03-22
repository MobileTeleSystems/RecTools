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
from rectools.model_selection import RandomSplitter
from rectools.model_selection.random_split import get_not_seen_mask

T = tp.TypeVar("T")
Converter = tp.Callable[[tp.Sequence[int]], tp.List[int]]


class TestTimeRangeSplit:
    @pytest.fixture
    def shuffle_arr(self) -> np.ndarray:
        return np.random.choice(np.arange(11), 11, replace=False)

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
    def train_size(self) -> float:
        return 0.5

    @pytest.mark.parametrize("execution_number", range(5))
    def test_without_filtering(self, interactions: Interactions, train_size: float, execution_number: list) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(train_size, False, False, False)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)
        assert len(actual) == 1
        assert len(actual[0]) == 3

        assert actual[0][0].shape[0] == int(train_size * interactions.df.shape[0])
        assert actual[0][0].shape[0] + actual[0][1].shape[0] == interactions.df.shape[0]

        fold_info = actual[0][2]
        df = interactions.df
        train_users = df.iloc[actual[0][0]]["user_id"]
        train_items = df.iloc[actual[0][0]]["item_id"]
        test_users = df.iloc[actual[0][1]]["user_id"]
        test_items = df.iloc[actual[0][1]]["item_id"]

        assert fold_info["Train"] == train_users.size
        assert fold_info["Train users"] == pd.unique(train_users).size
        assert fold_info["Train items"] == pd.unique(train_items).size
        assert fold_info["Test"] == test_users.size
        assert fold_info["Test users"] == pd.unique(test_users).size
        assert fold_info["Test items"] == pd.unique(test_items).size

    @pytest.mark.parametrize("execution_number", range(5))
    def test_cold_users(self, interactions: Interactions, train_size: float, execution_number: list) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(train_size, True, False, False)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_users = pd.unique(df.iloc[actual[0][0]]["user_id"])
        test_users = pd.unique(df.iloc[actual[0][1]]["user_id"])

        assert np.intersect1d(train_users, test_users).shape[0] == test_users.shape[0]

    @pytest.mark.parametrize("execution_number", range(5))
    def test_cold_items(self, interactions: Interactions, train_size: float, execution_number: list) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(train_size, False, True, False)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_items = pd.unique(df.iloc[actual[0][0]]["item_id"])
        test_items = pd.unique(df.iloc[actual[0][1]]["item_id"])

        assert np.intersect1d(train_items, test_items).shape[0] == test_items.shape[0]

    @pytest.mark.parametrize("execution_number", range(5))
    def test_filter_already_seen(self, interactions: Interactions, train_size: float, execution_number: list) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(train_size, False, False, True)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_interactions = df.iloc[actual[0][0]][["user_id", "item_id"]]
        test_interactions = df.iloc[actual[0][1]][["user_id", "item_id"]]

        assert train_interactions.merge(test_interactions, how="inner").shape[0] == 0

    @pytest.mark.parametrize(
        "incorrect_train_size, expected_error_type",
        (
            (-0.1, ValueError),
            (1.5, ValueError),
        ),
    )
    def test_with_incorrect_train_size(
        self, interactions: Interactions, incorrect_train_size: float, expected_error_type: tp.Type[Exception]
    ) -> None:
        with pytest.raises(expected_error_type):
            RandomSplitter(incorrect_train_size, False, False, False)


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
