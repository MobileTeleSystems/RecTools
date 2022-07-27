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
from rectools.metrics.novelty import MeanInvUserFreq


class TestMeanInvUserFreq:
    @pytest.fixture
    def interactions(self) -> pd.DataFrame:
        interactions = pd.DataFrame(
            [
                ["u1", "i1"],
                ["u1", "i2"],
                ["u2", "i1"],
                ["u3", "i1"],
            ],
            columns=[Columns.User, Columns.Item],
        )
        return interactions

    @pytest.fixture
    def recommendations(self) -> pd.DataFrame:
        recommendations = pd.DataFrame(
            [
                ["u1", "i3", 1],
                ["u2", "i2", 1],
                ["u2", "i3", 2],
                ["u3", "i1", 1],
                ["u3", "i2", 2],
            ],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        return recommendations

    @pytest.mark.parametrize(
        "k,expected",
        (
            (1, pd.Series(index=["u1", "u2", "u3"], data=[-np.log2(1 / 3), -np.log2(1 / 3), 0])),
            (2, pd.Series(index=["u1", "u2", "u3"], data=[-np.log2(1 / 3), -np.log2(1 / 3), -np.log2(1 / 3) / 2])),
        ),
    )
    def test_correct_miuf_values(
        self, recommendations: pd.DataFrame, interactions: pd.DataFrame, k: int, expected: pd.Series
    ) -> None:
        miuf = MeanInvUserFreq(k)

        actual = miuf.calc_per_user(recommendations, interactions)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

        actual_mean = miuf.calc(recommendations, interactions)
        assert actual_mean == expected.mean()
