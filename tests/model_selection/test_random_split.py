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


class TestRandomSplitter:
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
    def test_size(self) -> float:
        return 0.25

    @pytest.mark.parametrize("execution_number", range(5))
    def test_without_filtering(self, interactions: Interactions, test_size: float, execution_number: int) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(test_size, 2, None, False, False, False)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)
        assert len(actual) == 2
        assert len(actual[0]) == 3

        assert actual[0][1].shape[0] == int(test_size * interactions.df.shape[0])
        assert actual[0][0].shape[0] + actual[0][1].shape[0] == interactions.df.shape[0]
        assert actual[1][1].shape[0] == int(test_size * interactions.df.shape[0])
        assert actual[1][0].shape[0] + actual[0][1].shape[0] == interactions.df.shape[0]

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
    def test_filter_cold_users(self, interactions: Interactions, test_size: float, execution_number: int) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(test_size, 1, None, True, False, False)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_users = pd.unique(df.iloc[actual[0][0]]["user_id"])
        test_users = pd.unique(df.iloc[actual[0][1]]["user_id"])

        assert np.intersect1d(train_users, test_users).shape[0] == test_users.shape[0]

    @pytest.mark.parametrize("execution_number", range(5))
    def test_filter_cold_items(self, interactions: Interactions, test_size: float, execution_number: int) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(test_size, 1, None, False, True, False)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_items = pd.unique(df.iloc[actual[0][0]]["item_id"])
        test_items = pd.unique(df.iloc[actual[0][1]]["item_id"])

        assert np.intersect1d(train_items, test_items).shape[0] == test_items.shape[0]

    @pytest.mark.parametrize("execution_number", range(5))
    def test_filter_already_seen(self, interactions: Interactions, test_size: float, execution_number: int) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(test_size, 1, None, False, False, True)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_interactions = df.iloc[actual[0][0]][["user_id", "item_id"]]
        test_interactions = df.iloc[actual[0][1]][["user_id", "item_id"]]

        assert train_interactions.merge(test_interactions, how="inner").shape[0] == 0

    @pytest.mark.parametrize("execution_number", range(5))
    def test_filter_all(self, interactions: Interactions, test_size: float, execution_number: int) -> None:
        interactions_copy = deepcopy(interactions)
        rs = RandomSplitter(test_size, 1, None, True, True, True)

        actual = list(rs.split(interactions, collect_fold_stats=True))
        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        df = interactions.df
        train_users = df.iloc[actual[0][0]]["user_id"].drop_duplicates()
        train_items = df.iloc[actual[0][0]]["item_id"].drop_duplicates()
        test_users = df.iloc[actual[0][1]]["user_id"].drop_duplicates()
        test_items = df.iloc[actual[0][1]]["item_id"].drop_duplicates()
        train_interactions = df.iloc[actual[0][0]][["user_id", "item_id"]].drop_duplicates()
        test_interactions = df.iloc[actual[0][1]][["user_id", "item_id"]].drop_duplicates()

        assert np.intersect1d(train_users, test_users).shape[0] == test_users.shape[0]
        assert np.intersect1d(train_items, test_items).shape[0] == test_items.shape[0]
        assert train_interactions.merge(test_interactions, how="inner").shape[0] == 0

    @pytest.mark.parametrize("execution_number", range(2))
    @pytest.mark.parametrize("random_state", (10, 42, 156))
    def test_random_state(
        self, interactions: Interactions, test_size: float, random_state: int, execution_number: int
    ) -> None:
        interactions_copy = deepcopy(interactions)

        rs1 = RandomSplitter(test_size, 1, random_state, True, True, True)
        actual1 = list(rs1.split(interactions, collect_fold_stats=True))

        rs2 = RandomSplitter(test_size, 1, random_state, True, True, True)
        actual2 = list(rs2.split(interactions, collect_fold_stats=True))

        pd.testing.assert_frame_equal(interactions.df, interactions_copy.df)

        assert np.array_equal(actual1[0][0], actual2[0][0])
        assert np.array_equal(actual1[0][1], actual2[0][1])

    @pytest.mark.parametrize(
        "incorrect_test_size, expected_error_type",
        (
            (-0.1, ValueError),
            (0.0, ValueError),
            (1.0, ValueError),
            (1.5, ValueError),
        ),
    )
    def test_with_incorrect_test_size(
        self, interactions: Interactions, incorrect_test_size: float, expected_error_type: tp.Type[Exception]
    ) -> None:
        with pytest.raises(expected_error_type):
            RandomSplitter(incorrect_test_size, 1, None, False, False, False)