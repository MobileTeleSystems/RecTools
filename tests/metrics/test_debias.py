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
from collections import defaultdict

import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import MAP, PAP, PartialAUC
from rectools.metrics.base import merge_reco
from rectools.metrics.debias import DebiasConfig, DebiasableMetrikAtK, calc_debiased_fit_task

DEBIAS_CONFIG_DEFAULT = DebiasConfig(iqr_coef=1.5, random_state=32)


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

    def test_debias_interactions(self, interactions: pd.DataFrame, recommendations: pd.DataFrame) -> None:
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

        interactions_downsampling = DebiasableMetrikAtK.debias_interactions(interactions, config=DEBIAS_CONFIG_DEFAULT)
        merged_downsampling = DebiasableMetrikAtK.debias_interactions(merged, config=DEBIAS_CONFIG_DEFAULT)

        pd.testing.assert_frame_equal(
            interactions_downsampling.sort_values(Columns.UserItem, ignore_index=True),
            expected_result[Columns.UserItem],
        )
        pd.testing.assert_frame_equal(
            merged_downsampling.sort_values(Columns.UserItem, ignore_index=True), expected_result
        )

    def test_debias_interactions_when_no_interactions(self, empty_interactions: pd.DataFrame) -> None:
        interactions_downsampling = DebiasableMetrikAtK.debias_interactions(
            empty_interactions, config=DEBIAS_CONFIG_DEFAULT
        )
        pd.testing.assert_frame_equal(interactions_downsampling, empty_interactions, check_like=True)

    @pytest.mark.parametrize(
        "metrics_fitted",
        (
            {
                "dMAP@1": MAP(k=1, debias_config=DEBIAS_CONFIG_DEFAULT),
                "dMAP@3": MAP(k=3, debias_config=DEBIAS_CONFIG_DEFAULT),
                "dMAP@2": MAP(k=2, debias_config=DebiasConfig(iqr_coef=1.6, random_state=32)),
                "dMAP@4": MAP(k=4, debias_config=DebiasConfig(iqr_coef=1.6, random_state=10)),
                "dMAP@5": MAP(k=5, debias_config=DebiasConfig(iqr_coef=1, random_state=10)),
                "MAP@1": MAP(k=1),
                "MAP@5": MAP(k=5),
            },
            {
                "dPartialAUC@1": PartialAUC(k=1, debias_config=DEBIAS_CONFIG_DEFAULT),
                "dPartialAUC@3": PartialAUC(k=3, debias_config=DEBIAS_CONFIG_DEFAULT),
                "dPartialAUC@2": PartialAUC(k=2, debias_config=DebiasConfig(iqr_coef=1.6, random_state=32)),
                "dPartialAUC@4": PartialAUC(k=4, debias_config=DebiasConfig(iqr_coef=1.6, random_state=10)),
                "dPartialAUC@5": PartialAUC(k=5, debias_config=DebiasConfig(iqr_coef=1, random_state=10)),
                "PartialAUC@1": PartialAUC(k=1),
                "PartialAUC@5": PartialAUC(k=5),
            },
            {
                "dPAP@1": PAP(k=1, debias_config=DEBIAS_CONFIG_DEFAULT),
                "dPAP@3": PAP(k=3, debias_config=DEBIAS_CONFIG_DEFAULT),
                "dPAP@2": PAP(k=2, debias_config=DebiasConfig(iqr_coef=1.6, random_state=32)),
                "dPAP@4": PAP(k=4, debias_config=DebiasConfig(iqr_coef=1.6, random_state=10)),
                "dPAP@5": PAP(k=5, debias_config=DebiasConfig(iqr_coef=1, random_state=10)),
                "PAP@1": PAP(k=1),
                "PAP@5": PAP(k=5),
            },
        ),
    )
    def test_calc_debiased_fit_task(
        self, metrics_fitted: tp.Dict[str, DebiasableMetrikAtK], interactions: pd.DataFrame
    ) -> None:
        debiased_fit_task = calc_debiased_fit_task(metrics=metrics_fitted.values(), interactions=interactions)

        unique_debias_config_expected = set()
        k_max_expected: tp.Dict[DebiasConfig, int] = defaultdict(int)
        for metric in metrics_fitted.values():
            unique_debias_config_expected.add(metric.debias_config)
            k_max_expected[metric.debias_config] = max(k_max_expected[metric.debias_config], metric.k)

        num_unique_configs_expected = len(unique_debias_config_expected)

        assert len(debiased_fit_task) == num_unique_configs_expected
        assert set(debiased_fit_task.keys()) == unique_debias_config_expected
        for value in k_max_expected:
            assert debiased_fit_task[value][0] == k_max_expected[value]
