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

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import (
    MAP,
    MCC,
    MRR,
    NDCG,
    Accuracy,
    F1Beta,
    MeanInvUserFreq,
    Precision,
    Recall,
    calc_metrics,
    debias_wrapper,
)
from rectools.metrics.base import DebiasMetric, merge_reco
from rectools.metrics.classification import (
    DebiasAccuracy,
    DebiasClassificationMetric,
    DebiasMCC,
    DebiasSimpleClassificationMetric,
)
from rectools.metrics.ranking import DebiasMAP, DebiasRankingMetric


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

    @pytest.fixture
    def interactions_downsampling(self, interactions: pd.DataFrame) -> pd.DataFrame:
        metric_debias = DebiasMetric(iqr_coef=1.5, random_state=32)
        return metric_debias.make_downsample(interactions)

    @pytest.fixture
    def merged_downsampling(self, interactions: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
        metric_debias = DebiasMetric(iqr_coef=1.5, random_state=32)
        merged = merge_reco(recommendations, interactions)
        return metric_debias.make_downsample(merged)

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

    @pytest.mark.parametrize(
        "metric",
        (
            Precision(k=1),
            Precision(k=3),
            Recall(k=1),
            Recall(k=3),
            F1Beta(k=1),
            F1Beta(k=3),
            Accuracy(k=1),
            Accuracy(k=3),
            MCC(k=1),
            MCC(k=3),
            MAP(k=1),
            MAP(k=3),
            NDCG(k=1),
            NDCG(k=3),
            MRR(k=1),
            MRR(k=3),
        ),
    )
    def test_debias_metric_calc(
        self,
        metric: tp.Union[DebiasClassificationMetric, DebiasSimpleClassificationMetric, DebiasRankingMetric],
        interactions: pd.DataFrame,
        recommendations: pd.DataFrame,
        interactions_downsampling: pd.DataFrame,
        catalog: tp.List[int],
    ) -> None:
        debias_metric = debias_wrapper(metric, iqr_coef=1.5, random_state=32)

        if isinstance(debias_metric, DebiasAccuracy) or isinstance(debias_metric, DebiasMCC):
            expected_result_per_user = metric.calc_per_user(  # type: ignore
                recommendations, interactions_downsampling, catalog
            )
            result_per_user = debias_metric.calc_per_user(recommendations, interactions, catalog)

            expected_result_mean = metric.calc(recommendations, interactions_downsampling, catalog)  # type: ignore
            result_mean = debias_metric.calc(recommendations, interactions, catalog)
        else:
            expected_result_per_user = metric.calc_per_user(recommendations, interactions_downsampling)  # type: ignore
            result_per_user = debias_metric.calc_per_user(recommendations, interactions)

            expected_result_mean = metric.calc(recommendations, interactions_downsampling)  # type: ignore
            result_mean = debias_metric.calc(recommendations, interactions)

        pd.testing.assert_series_equal(result_per_user, expected_result_per_user)
        assert result_mean == expected_result_mean

    @pytest.mark.parametrize(
        "metric",
        (
            Precision(k=3),
            Recall(k=3),
            F1Beta(k=3),
            Accuracy(k=3),
            MCC(k=3),
            MAP(k=3),
            NDCG(k=3),
            MRR(k=3),
        ),
    )
    def test_when_no_interactions(
        self,
        metric: tp.Union[DebiasClassificationMetric, DebiasClassificationMetric, DebiasRankingMetric],
        recommendations: pd.DataFrame,
        empty_interactions: pd.DataFrame,
        catalog: tp.List[int],
    ) -> None:
        debias_metric = debias_wrapper(metric, iqr_coef=1.5, random_state=32)
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        if isinstance(debias_metric, DebiasAccuracy) or isinstance(debias_metric, DebiasMCC):
            result_metric_per_user = debias_metric.calc_per_user(recommendations, empty_interactions, catalog)
            assert np.isnan(debias_metric.calc(recommendations, empty_interactions, catalog))
        else:
            result_metric_per_user = debias_metric.calc_per_user(recommendations, empty_interactions)
            assert np.isnan(debias_metric.calc(recommendations, empty_interactions))
        pd.testing.assert_series_equal(result_metric_per_user, expected_metric_per_user)

    def test_when_metric_no_classification_or_ranking(self) -> None:
        with pytest.raises(TypeError):
            debias_wrapper(MeanInvUserFreq(k=1), iqr_coef=1.5, random_state=32)  # type: ignore

    def test_map_fit(self, interactions: pd.DataFrame, recommendations: pd.DataFrame) -> None:
        merged = merge_reco(recommendations, interactions)
        debias_metric = debias_wrapper(MAP(k=3), iqr_coef=1.5, random_state=32)
        merged_downsampling = debias_metric.make_downsample(merged)

        expected_fitted = MAP.fit(merged_downsampling, 3)
        if isinstance(debias_metric, DebiasMAP):
            actual_fitted = debias_metric.fit(merged, 3)

        np.allclose(actual_fitted.precision_at_k.A, expected_fitted.precision_at_k.A)
        np.testing.assert_equal(actual_fitted.users, expected_fitted.users)
        np.testing.assert_equal(actual_fitted.n_relevant_items, expected_fitted.n_relevant_items)

    def test_calc_metrics(
        self,
        interactions: pd.DataFrame,
        recommendations: pd.DataFrame,
        interactions_downsampling: pd.DataFrame,
        catalog: tp.List[int],
    ) -> None:
        debias_metrics = {
            "debias_precision@3": debias_wrapper(Precision(k=3), iqr_coef=1.5, random_state=32),
            "debias_recall@3": debias_wrapper(Recall(k=3), iqr_coef=1.5, random_state=32),
            "debias_f1beta@3": debias_wrapper(F1Beta(k=3), iqr_coef=1.5, random_state=32),
            "debias_accuracy@3": debias_wrapper(Accuracy(k=3), iqr_coef=1.5, random_state=32),
            "debias_mcc@3": debias_wrapper(MCC(k=3), iqr_coef=1.5, random_state=32),
            "debias_map@3": debias_wrapper(MAP(k=3), iqr_coef=1.5, random_state=32),
            "debias_ndcg@3": debias_wrapper(NDCG(k=3, log_base=3), iqr_coef=1.5, random_state=32),
            "debias_mrr@3": debias_wrapper(MRR(k=3), iqr_coef=1.5, random_state=32),
        }

        metrics = {
            "debias_precision@3": Precision(k=3),
            "debias_recall@3": Recall(k=3),
            "debias_f1beta@3": F1Beta(k=3),
            "debias_accuracy@3": Accuracy(k=3),
            "debias_mcc@3": MCC(k=3),
            "debias_map@3": MAP(k=3),
            "debias_ndcg@3": NDCG(k=3, log_base=3),
            "debias_mrr@3": MRR(k=3),
        }

        actual = calc_metrics(
            metrics=debias_metrics,  # type: ignore[arg-type]
            reco=recommendations,
            interactions=interactions,
            catalog=catalog,
        )
        expected = calc_metrics(
            metrics=metrics, reco=recommendations, interactions=interactions_downsampling, catalog=catalog
        )
        assert actual == expected
