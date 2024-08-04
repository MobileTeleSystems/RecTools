#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import DebiasConfig, debias_interactions
from rectools.metrics.base import merge_reco
from rectools.metrics.ranking import MAP, MRR, NDCG, RankingMetric

EMPTY_INTERACTIONS = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)
DEBIAS_CONFIG = DebiasConfig(iqr_coef=1.5, random_state=32)


class TestMAP:
    @pytest.mark.parametrize(
        "k,divide_by_k,expected_ap",
        (
            (1, False, [0, 0, 1 / 6, 1, 1 / 3, 0, 0, 0]),
            (3, False, [0, 1 / 3, 1 / 6 * (1 / 1 + 2 / 3), 1, 1, 0, 1 * (1 / 2), 0]),
            (1, True, [0, 0, 1, 1, 1, 0, 0, 0]),
            (3, True, [0, 1 / 9, 1 / 3 * (1 / 1 + 2 / 3), 1 / 3, 1, 0, 1 / 3 * (1 / 2), 0]),
        ),
    )
    def test_calc(self, k: int, divide_by_k: bool, expected_ap: tp.List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 7, 8, 9],
                Columns.Item: [1, 3, 1, 1, 2, 3, 4, 5, 1, 1, 2, 3, 1, 2, 1],
                Columns.Rank: [9, 1, 3, 1, 3, 5, 7, 9, 1, 1, 2, 3, 2, 1, 1],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                Columns.Item: [1, 2, 1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1, 1],
            }
        )

        metric = MAP(k=k, divide_by_k=divide_by_k)
        expected_metric_per_user = pd.Series(
            expected_ap,
            index=pd.Series([1, 2, 3, 4, 5, 6, 7, 8], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert metric.calc(reco, interactions) == expected_metric_per_user.mean()

    @pytest.mark.parametrize("metric", (MAP(k=3), MAP(k=3, divide_by_k=True)))
    def test_when_no_interactions(self, metric: MAP) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))

    def test_when_duplicates_in_interactions(self) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 3, 1, 2, 3],
                Columns.Rank: [1, 2, 3, 1, 2, 3],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 1, 1, 2, 3],
            }
        )
        metric = MAP(k=3)
        expected_metric_per_user = pd.Series(
            [3.5 / 3, 3 / 3],
            index=pd.Series([1, 2], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)


class TestNDCG:

    _idcg_at_3 = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)

    @pytest.mark.parametrize(
        "k,expected_ndcg",
        (
            (1, [0, 0, 1, 1, 0]),
            (3, [0, 0, 1, 1 / _idcg_at_3, 0.5 / _idcg_at_3]),
        ),
    )
    def test_calc(self, k: int, expected_ndcg: tp.List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6],
                Columns.Item: [1, 2, 1, 2, 3, 1, 1, 2, 3, 5, 1],
                Columns.Rank: [9, 1, 1, 2, 3, 1, 3, 7, 9, 1, 1],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5],
                Columns.Item: [1, 1, 1, 2, 3, 1, 1, 2, 3, 4],
            }
        )

        metric = NDCG(k=k)
        expected_metric_per_user = pd.Series(
            expected_ndcg,
            index=pd.Series([1, 2, 3, 4, 5], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert np.allclose(metric.calc(reco, interactions), expected_metric_per_user.mean())

    def test_when_no_interactions(self) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        metric = NDCG(k=3)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))


class TestMRR:
    @pytest.mark.parametrize(
        "k,expected_mrr",
        (
            (1, [0, 0, 1, 1, 0]),
            (3, [0, 0, 1, 1, 1 / 3]),
        ),
    )
    def test_calc(self, k: int, expected_mrr: tp.List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5],
                Columns.Item: [1, 2, 1, 2, 3, 1, 1, 2, 3, 5],
                Columns.Rank: [9, 1, 1, 2, 3, 1, 3, 7, 9, 1],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5],
                Columns.Item: [1, 1, 1, 2, 3, 1, 1, 2, 3, 4],
            }
        )

        metric = MRR(k=k)
        expected_metric_per_user = pd.Series(
            expected_mrr,
            index=pd.Series([1, 2, 3, 4, 5], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert np.allclose(metric.calc(reco, interactions), expected_metric_per_user.mean())

    def test_when_no_interactions(self) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        metric = MRR(k=3)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))

    def test_when_duplicates_in_interactions(self) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 3, 1, 2, 3],
                Columns.Rank: [1, 2, 3, 4, 5, 6],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 1, 1, 2, 3],
            }
        )
        metric = MRR(k=3)
        expected_metric_per_user = pd.Series(
            [1, 0],
            index=pd.Series([1, 2], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)


class TestDebiasableRankingMetric:
    def setup_method(self) -> None:
        self.reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 7, 8, 9],
                Columns.Item: [1, 3, 1, 1, 2, 3, 4, 5, 1, 1, 2, 3, 1, 2, 1],
                Columns.Rank: [9, 1, 3, 1, 3, 5, 7, 9, 1, 1, 2, 3, 2, 1, 1],
            }
        )
        self.interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                Columns.Item: [1, 2, 1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1, 1],
            }
        )
        self.merged = merge_reco(self.reco, self.interactions)

    @pytest.mark.parametrize(
        "metric, debiased_metric",
        (
            (MAP(k=3), MAP(k=3, debias_config=DEBIAS_CONFIG)),
            (NDCG(k=3), NDCG(k=3, debias_config=DEBIAS_CONFIG)),
            (MRR(k=3), MRR(k=3, debias_config=DEBIAS_CONFIG)),
        ),
    )
    def test_calc(self, metric: RankingMetric, debiased_metric: RankingMetric) -> None:
        debiased_interactions = debias_interactions(self.interactions, config=debiased_metric.debias_config)

        expected_metric_per_user = metric.calc_per_user(self.reco, debiased_interactions)
        actual_metric_per_user = debiased_metric.calc_per_user(self.reco, self.interactions)
        actual_metric = debiased_metric.calc(self.reco, self.interactions)

        pd.testing.assert_series_equal(actual_metric_per_user, expected_metric_per_user)
        assert actual_metric == expected_metric_per_user.mean()

    @pytest.mark.parametrize(
        "debiased_metric",
        (
            MAP(k=3, debias_config=DEBIAS_CONFIG),
            NDCG(k=3, debias_config=DEBIAS_CONFIG),
            MRR(k=3, debias_config=DEBIAS_CONFIG),
        ),
    )
    def test_when_no_interactions(self, debiased_metric: RankingMetric) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)

        pd.testing.assert_series_equal(
            debiased_metric.calc_per_user(self.reco, EMPTY_INTERACTIONS),
            expected_metric_per_user,
        )
        assert np.isnan(debiased_metric.calc(self.reco, EMPTY_INTERACTIONS))

    @pytest.mark.parametrize(
        "metric",
        (
            MAP(k=3),
            MAP(k=3, debias_config=DEBIAS_CONFIG),
        ),
    )
    def test_raise_when_correct_is_debias(self, metric: MAP) -> None:
        fitted = metric.fit(self.merged, metric.k)
        result = metric.calc_from_fitted(fitted, is_debiased=metric.debias_config is not None)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "metric",
        (MAP(k=3, debias_config=DEBIAS_CONFIG),),
    )
    def test_raise_when_incorrect_is_debias(self, metric: MAP) -> None:
        fitted = metric.fit(self.merged, metric.k)
        with pytest.raises(ValueError):
            metric.calc_from_fitted(fitted)
