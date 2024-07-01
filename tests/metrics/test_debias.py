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

import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import DebiasConfig
from rectools.metrics.base import merge_reco
from rectools.metrics.debias import DebiasableMetrikAtK


class TestDebias:
    @pytest.fixture
    def interactions(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                Columns.Item: [1, 2, 1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1, 1],
            }
        )
        return interactions_df

    @pytest.fixture
    def recommendations(self) -> pd.DataFrame:
        reco_df = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 7, 8, 9],
                Columns.Item: [1, 3, 1, 1, 2, 3, 4, 5, 1, 1, 2, 3, 1, 2, 1],
                Columns.Rank: [9, 1, 3, 1, 3, 5, 7, 9, 1, 1, 2, 3, 2, 1, 1],
            }
        )
        return reco_df

    @pytest.fixture
    def empty_interactions(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)

    @pytest.fixture
    def debias_metric(self) -> DebiasableMetrikAtK:
        return DebiasableMetrikAtK(k=10, debias_config=DebiasConfig(iqr_coef=1.5, random_state=32))

    def test_make_debias(
        self, interactions: pd.DataFrame, recommendations: pd.DataFrame, debias_metric: DebiasableMetrikAtK
    ) -> None:
        merged = merge_reco(recommendations, interactions)

        expected_result = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5, 7],
                Columns.Item: [1, 2, 1, 1, 2, 3, 4, 5, 6, 1, 2, 3, 1],
            }
        )
        expected_result = pd.merge(
            expected_result,
            recommendations,
            how="left",
            on=Columns.UserItem,
        )

        interactions_downsampling = debias_metric.make_debias(interactions)
        merged_downsampling = debias_metric.make_debias(merged)

        pd.testing.assert_frame_equal(
            interactions_downsampling.sort_values(Columns.UserItem, ignore_index=True),
            expected_result[Columns.UserItem],
        )
        pd.testing.assert_frame_equal(
            merged_downsampling.sort_values(Columns.UserItem, ignore_index=True), expected_result
        )

    def test_make_debias_with_empty_data(
        self, empty_interactions: pd.DataFrame, debias_metric: DebiasableMetrikAtK
    ) -> None:
        interactions_downsampling = debias_metric.make_debias(empty_interactions)

        pd.testing.assert_frame_equal(interactions_downsampling, empty_interactions, check_like=True)
