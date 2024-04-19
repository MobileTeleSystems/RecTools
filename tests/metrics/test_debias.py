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


from rectools import Columns
from rectools.metrics import calc_metrics, debias_wrapper
from rectools.metrics.base import DebiasMetric, merge_reco
from rectools.metrics.ranking import DebiasMAP, DebiasNDCG, DebiasMRR
from rectools.metrics.classification import DebiasPrecision, DebiasRecall, DebiasF1Beta, DebiasAccuracy, DebiasMCC


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
    def catalog(self) -> tp.List[int]:
        return list(range(10))

    def test_make_downsample(self, interactions: pd.DataFrame, recommendations: pd.DataFrame) -> None:
        metric_debias = DebiasMetric(iqr_coef=1.5, random_state=32)
        merged = merge_reco(recommendations, interactions)

        expected_result = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5, 7],
                Columns.Item: [2, 1, 1, 2, 3, 4, 5, 6, 1, 2, 3, 1, 1],
            }
        )
        expected_result = pd.merge(
            expected_result,
            recommendations,
            how="left",
            on=Columns.UserItem,
        )

        interactions_downsampling = metric_debias.make_downsample(interactions)
        merged_downsampling = metric_debias.make_downsample(merged)

        pd.testing.assert_frame_equal(interactions_downsampling, expected_result[Columns.UserItem], check_like=True)
        pd.testing.assert_frame_equal(merged_downsampling, expected_result, check_like=True)

    def test_make_downsample_with_empty_data(self, empty_interactions: pd.DataFrame) -> None:
        metric_debias = DebiasMetric(iqr_coef=1.5, random_state=32)
        interactions_downsampling = metric_debias.make_downsample(empty_interactions)

        pd.testing.assert_frame_equal(interactions_downsampling, empty_interactions, check_like=True)
        
    def test_calc(
        self,
        intecations: pd.DataFrame,
        recommendations: pd.DataFrame,
        catalog: tp.List[int],
    ) -> None:
        pass

    def test_calc_empty(
        self,
        intecations: pd.DataFrame,
        recommendations: pd.DataFrame,
        catalog: tp.List[int],
    ) -> None:
        pass
