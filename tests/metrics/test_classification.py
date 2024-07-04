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
from rectools.metrics import MCC, Accuracy, DebiasConfig, F1Beta, HitRate, Precision, Recall
from rectools.metrics.base import MetricAtK, merge_reco
from rectools.metrics.classification import (
    ClassificationMetric,
    SimpleClassificationMetric,
    calc_classification_metrics,
    calc_confusions,
)

RECO = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 2, 2, 3, 4],
        Columns.Item: [1, 2, 3, 1, 2, 1, 1],
        Columns.Rank: [1, 2, 3, 1, 2, 1, 1],
    }
)
INTERACTIONS = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 3, 4, 5],
        Columns.Item: [4, 2, 3, 1, 2, 2],
    }
)
CATALOG = list(range(10))
EMPTY_INTERACTIONS = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)
DEBIAS_CONFIG = DebiasConfig(iqr_coef=1.5, random_state=32)


class TestPrecision:
    def setup_method(self) -> None:
        self.metric = Precision(k=2)
        self.r_precision = Precision(k=2, r_precision=True)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [0.5, 0.5, 0.0, 0.0],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS) == expected_metric_per_user.mean()

        expected_metric_per_user_r_prec = pd.Series(
            [0.5, 1, 0.0, 0.0],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(
            self.r_precision.calc_per_user(RECO, INTERACTIONS), expected_metric_per_user_r_prec
        )
        assert self.r_precision.calc(RECO, INTERACTIONS) == expected_metric_per_user_r_prec.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS))
        pd.testing.assert_series_equal(
            self.r_precision.calc_per_user(RECO, EMPTY_INTERACTIONS), expected_metric_per_user
        )
        assert np.isnan(self.r_precision.calc(RECO, EMPTY_INTERACTIONS))


class TestRecall:
    def setup_method(self) -> None:
        self.metric = Recall(k=2)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [1 / 3, 1, 0, 0],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS) == expected_metric_per_user.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS))


class TestAccuracy:
    def setup_method(self) -> None:
        self.metric = Accuracy(k=2)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [0.7, 0.9, 0.7, 0.7],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS, CATALOG), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS, CATALOG) == expected_metric_per_user.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(
            self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS, CATALOG),
            expected_metric_per_user,
        )
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS, CATALOG))


class TestCalcClassificationMetrics:
    def test_raises_when_unexpected_metric_type(self) -> None:
        metric = MetricAtK(k=1)
        with pytest.raises(TypeError):
            calc_classification_metrics(
                {"m": metric},  # type: ignore
                pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank]),
            )

    def test_raises_when_no_catalog_set_when_needed(self) -> None:
        metric = ClassificationMetric(k=1)
        with pytest.raises(ValueError):
            calc_classification_metrics({"m": metric}, pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank]))


class TestF1Beta:
    def setup_method(self) -> None:
        self.metric = F1Beta(k=2, beta=2 ** (1 / 2))

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [0.375, 0.75, 0, 0],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS) == expected_metric_per_user.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS))


class TestMCC:
    def setup_method(self) -> None:
        self.metric = MCC(k=2)

    def test_calc(self) -> None:

        # tp = pd.Series([1, 1, 0, 0])
        # tn = pd.Series([6, 8, 7, 7])
        # fp = pd.Series([1, 1, 2, 2])
        # fn = pd.Series([2, 0, 1, 1])

        expected_metric_per_user = pd.Series(
            [1 / (21 ** (1 / 2)), 2 / 3, -1 / 6, -1 / 6],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS, CATALOG), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS, CATALOG) == expected_metric_per_user.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(
            self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS, CATALOG),
            expected_metric_per_user,
        )
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS, CATALOG))


class TestHitRate:
    def setup_method(self) -> None:
        self.metric = HitRate(k=2)

    def test_calc(self) -> None:

        # tp = pd.Series([1, 1, 0, 0])
        # tn = pd.Series([6, 8, 7, 7])
        # fp = pd.Series([1, 1, 2, 2])
        # fn = pd.Series([2, 0, 1, 1])

        expected_metric_per_user = pd.Series(
            [1, 1, 0, 0], index=pd.Series([1, 3, 4, 5], name=Columns.User, dtype=int), dtype=float
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS) == expected_metric_per_user.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(
            self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS),
            expected_metric_per_user,
        )
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS))


class TestDebiasMetric:
    @pytest.mark.parametrize(
        "metric, metric_debias",
        (
            (Precision(k=2), Precision(k=2, debias_config=DEBIAS_CONFIG)),
            (Precision(k=2), Precision(k=2, debias_config=DEBIAS_CONFIG)),
            (Recall(k=2), Recall(k=2, debias_config=DEBIAS_CONFIG)),
            (Accuracy(k=2), Accuracy(k=2, debias_config=DEBIAS_CONFIG)),
            (F1Beta(k=2), F1Beta(k=2, debias_config=DEBIAS_CONFIG)),
            (MCC(k=2), MCC(k=2, debias_config=DEBIAS_CONFIG)),
            (HitRate(k=2), HitRate(k=2, debias_config=DEBIAS_CONFIG)),
        ),
    )
    def test_calc(
        self,
        metric: tp.Union[ClassificationMetric, SimpleClassificationMetric],
        metric_debias: tp.Union[ClassificationMetric, SimpleClassificationMetric],
    ) -> None:
        downsample_interactions = metric_debias.make_debias(interactions_for_debiasing=INTERACTIONS)

        if isinstance(metric, ClassificationMetric):
            expected_metric_per_user_downsample = metric.calc_per_user(RECO, downsample_interactions, CATALOG)
        else:
            expected_metric_per_user_downsample = metric.calc_per_user(RECO, downsample_interactions)

        if isinstance(metric_debias, ClassificationMetric):
            result_metric_per_user = metric_debias.calc_per_user(RECO, INTERACTIONS, CATALOG)
            result_calc = metric_debias.calc(RECO, INTERACTIONS, CATALOG)
        else:
            result_metric_per_user = metric_debias.calc_per_user(RECO, INTERACTIONS)
            result_calc = metric_debias.calc(RECO, INTERACTIONS)

        pd.testing.assert_series_equal(result_metric_per_user, expected_metric_per_user_downsample)
        assert result_calc == expected_metric_per_user_downsample.mean()

    @pytest.mark.parametrize(
        "metric_debias",
        (
            (Precision(k=2, debias_config=DEBIAS_CONFIG)),
            (Precision(k=2, debias_config=DEBIAS_CONFIG)),
            (Recall(k=2, debias_config=DEBIAS_CONFIG)),
            (Accuracy(k=2, debias_config=DEBIAS_CONFIG)),
            (F1Beta(k=2, debias_config=DEBIAS_CONFIG)),
            (MCC(k=2, debias_config=DEBIAS_CONFIG)),
            (HitRate(k=2, debias_config=DEBIAS_CONFIG)),
        ),
    )
    def test_when_no_interactions(
        self, metric_debias: tp.Union[ClassificationMetric, SimpleClassificationMetric]
    ) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)

        if isinstance(metric_debias, ClassificationMetric):
            calc_per_user_result = metric_debias.calc_per_user(RECO, EMPTY_INTERACTIONS, CATALOG)
            calc_result = metric_debias.calc(RECO, EMPTY_INTERACTIONS, CATALOG)
        else:
            calc_per_user_result = metric_debias.calc_per_user(RECO, EMPTY_INTERACTIONS)
            calc_result = metric_debias.calc(RECO, EMPTY_INTERACTIONS)

        pd.testing.assert_series_equal(calc_per_user_result, expected_metric_per_user)
        assert np.isnan(calc_result)

    @pytest.mark.parametrize(
        "metric",
        (
            Precision(k=3, debias_config=DEBIAS_CONFIG),
            Accuracy(k=3, debias_config=DEBIAS_CONFIG),
        ),
    )
    def test_check_debias(self, metric: tp.Union[ClassificationMetric, SimpleClassificationMetric]) -> None:
        merged = merge_reco(RECO, INTERACTIONS)
        downsample_merged = metric.make_debias(merged)
        confusion_df = calc_confusions(downsample_merged, k=2)
        with pytest.raises(ValueError):
            if isinstance(metric, ClassificationMetric):
                metric.calc_from_confusion_df(confusion_df, CATALOG)
            else:
                metric.calc_from_confusion_df(confusion_df)
