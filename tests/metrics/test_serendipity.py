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

import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics.serendipity import Serendipity


class TestSerendipityCalculator:
    @pytest.fixture
    def interactions_train(self) -> pd.DataFrame:
        interactions_train = pd.DataFrame(
            [
                ["u1", "i1"],
                ["u1", "i2"],
                ["u2", "i1"],
                ["u2", "i2"],
                ["u3", "i1"],
            ],
            columns=[Columns.User, Columns.Item],
        )
        return interactions_train

    @pytest.fixture
    def interactions_test(self) -> pd.DataFrame:
        interactions_test = pd.DataFrame(
            [
                ["u1", "i1"],
                ["u1", "i2"],
                ["u2", "i2"],
                ["u2", "i3"],
                ["u3", "i2"],
                ["u4", "i2"],
            ],
            columns=[Columns.User, Columns.Item],
        )
        return interactions_test

    @pytest.fixture
    def recommendations(self) -> pd.DataFrame:
        recommendations = pd.DataFrame(
            [
                ["u1", "i1", 1],
                ["u1", "i2", 2],
                ["u2", "i2", 1],
                ["u2", "i3", 2],
                ["u3", "i3", 1],
                ["u4", "i2", 1],
                ["u4", "i3", 2],
            ],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        return recommendations

    @pytest.mark.parametrize(
        "k,expected",
        (
            (1, pd.Series(index=["u1", "u2", "u3", "u4"], data=[0, 0.25, 0, 0.25])),
            (2, pd.Series(index=["u1", "u2", "u3", "u4"], data=[0, 0.5, 0, 0.125])),
        ),
    )
    def test_correct_serendipity_values(
        self,
        recommendations: pd.DataFrame,
        interactions_train: pd.DataFrame,
        interactions_test: pd.DataFrame,
        k: int,
        expected: pd.Series,
    ) -> None:
        serendipity = Serendipity(k)

        actual = serendipity.calc_per_user(
            reco=recommendations,
            interactions=interactions_test,
            prev_interactions=interactions_train,
            catalog=["i1", "i2", "i3", "i4"],
        )
        pd.testing.assert_series_equal(actual, expected, check_names=False)

        actual_mean = serendipity.calc(
            reco=recommendations,
            interactions=interactions_test,
            prev_interactions=interactions_train,
            catalog=["i1", "i2", "i3", "i4"],
        )
        assert actual_mean == expected.mean()
