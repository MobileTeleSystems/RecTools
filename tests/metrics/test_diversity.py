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
from rectools.metrics import PairwiseHammingDistanceCalculator
from rectools.metrics.diversity import IntraListDiversity


class TestIntraListDiversity:
    @pytest.fixture
    def distance_calculator(self) -> PairwiseHammingDistanceCalculator:
        features_df = pd.DataFrame(
            [
                ["i1", 0, 0],
                ["i2", 0, 1],
                ["i3", 1, 1],
            ],
            columns=[Columns.Item, "feature_1", "feature_2"],
        ).set_index(Columns.Item)
        return PairwiseHammingDistanceCalculator(features_df)

    @pytest.fixture
    def recommendations(self) -> pd.DataFrame:
        recommendations = pd.DataFrame(
            [["u1", "i1", 1], ["u1", "i2", 2], ["u1", "i3", 3], ["u2", "i1", 1], ["u2", "i4", 2], ["u3", "i1", 1]],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        return recommendations

    @pytest.mark.parametrize(
        "k,expected",
        (
            (1, pd.Series(index=["u1", "u2", "u3"], data=[0, 0, 0])),
            (2, pd.Series(index=["u1", "u2", "u3"], data=[1, np.nan, 0])),
            (3, pd.Series(index=["u1", "u2", "u3"], data=[4 / 3, np.nan, 0])),
        ),
    )
    def test_correct_ild_values(
        self,
        distance_calculator: PairwiseHammingDistanceCalculator,
        recommendations: pd.DataFrame,
        k: int,
        expected: pd.Series,
    ) -> None:
        ild = IntraListDiversity(k, distance_calculator)

        actual = ild.calc_per_user(recommendations)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

        actual_mean = ild.calc(recommendations)
        assert actual_mean == expected.mean()
