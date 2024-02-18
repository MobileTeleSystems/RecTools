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

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics.popularity import AvgRecPopularity


class TestAvgRecPopularity:
    @pytest.fixture
    def interactions(self) -> pd.DataFrame:
        interactions = pd.DataFrame(
            [["u1", "i1"], ["u1", "i2"], ["u2", "i1"], ["u2", "i3"], ["u3", "i1"], ["u3", "i2"]],
            columns=[Columns.User, Columns.Item],
        )
        return interactions

    @pytest.fixture
    def recommendations(self) -> pd.DataFrame:
        recommendations = pd.DataFrame(
            [
                ["u1", "i1", 1],
                ["u1", "i2", 2],
                ["u2", "i3", 1],
                ["u2", "i1", 2],
                ["u2", "i2", 3],
                ["u3", "i3", 1],
                ["u3", "i2", 2],
            ],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        return recommendations

    @pytest.mark.parametrize(
        "k,expected",
        (
            (1, pd.Series(index=["u1", "u2", "u3"], data=[3.0, 1.0, 1.0])),
            (3, pd.Series(index=["u1", "u2", "u3"], data=[2.5, 2.0, 1.5])),
        ),
    )
    def test_correct_arp_values(
        self, recommendations: pd.DataFrame, interactions: pd.DataFrame, k: int, expected: pd.Series
    ) -> None:
        arp = AvgRecPopularity(k)

        actual = arp.calc_per_user(recommendations, interactions)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

        actual_mean = arp.calc(recommendations, interactions)
        assert actual_mean == expected.mean()

    def test_when_no_interactions(
        self,
        recommendations: pd.DataFrame,
    ) -> None:
        expected = pd.Series(index=recommendations[Columns.User].unique(), data=[0.0, 0.0, 0.0])
        empty_interactions = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)
        arp = AvgRecPopularity(k=2)

        actual = arp.calc_per_user(recommendations, empty_interactions)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

        actual_mean = arp.calc(recommendations, empty_interactions)
        assert actual_mean == expected.mean()

    @pytest.mark.parametrize(
        "k,expected",
        (
            (1, pd.Series(index=["u1", "u2", "u3"], data=[3.0, 1.0, 1.0])),
            (3, pd.Series(index=["u1", "u2", "u3"], data=[2.5, np.divide(4, 3), 1.5])),
        ),
    )
    def test_when_new_item_in_reco(self, interactions: pd.DataFrame, k: int, expected: pd.Series) -> None:
        reco = pd.DataFrame(
            [
                ["u1", "i1", 1],
                ["u1", "i2", 2],
                ["u2", "i3", 1],
                ["u2", "i1", 2],
                ["u2", "i4", 3],
                ["u3", "i3", 1],
                ["u3", "i2", 2],
            ],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        arp = AvgRecPopularity(k)

        actual = arp.calc_per_user(reco, interactions)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

        actual_mean = arp.calc(reco, interactions)
        assert actual_mean == expected.mean()
