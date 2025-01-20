#  Copyright 2024 MTS (Mobile Telesystems)
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

import pandas as pd
import pytest

from rectools.columns import Columns
from rectools.models.nn.transformer_data_preparator import SequenceDataset


class TestSequenceDataset:

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 4, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 8, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.mark.parametrize(
        "expected_sessions, expected_weights",
        (([[14, 11, 12, 13], [15, 12, 11], [11, 17], [16]], [[1, 1, 4, 1], [1, 2, 1], [1, 8], [1]]),),
    )
    def test_from_interactions(
        self,
        interactions_df: pd.DataFrame,
        expected_sessions: tp.List[tp.List[int]],
        expected_weights: tp.List[tp.List[float]],
    ) -> None:
        actual = SequenceDataset.from_interactions(interactions=interactions_df, sort_users=True)
        assert len(actual.sessions) == len(expected_sessions)
        assert all(
            actual_list == expected_list for actual_list, expected_list in zip(actual.sessions, expected_sessions)
        )
        assert len(actual.weights) == len(expected_weights)
        assert all(actual_list == expected_list for actual_list, expected_list in zip(actual.weights, expected_weights))
