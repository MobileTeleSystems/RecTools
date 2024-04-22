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

# pylint: disable=attribute-defined-outside-init

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import MCC, Accuracy, F1Beta, Precision, Recall, HitRate
from rectools.metrics.base import MetricAtK
from rectools.metrics.classification import ClassificationMetric, calc_classification_metrics

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


class TestPrecision:
    def setup(self) -> None:
        self.metric = Precision(k=2)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [0.5, 0.5, 0.0, 0.0],
            index=pd.Series([1, 3, 4, 5], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, INTERACTIONS), expected_metric_per_user)
        assert self.metric.calc(RECO, INTERACTIONS) == expected_metric_per_user.mean()

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(self.metric.calc(RECO, EMPTY_INTERACTIONS))


class TestRecall:
    def setup(self) -> None:
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
    def setup(self) -> None:
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
    def setup(self) -> None:
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
    def setup(self) -> None:
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
    def setup(self) -> None:
        self.metric = HitRate(k=2)

    def test_calc(self) -> None:

        # tp = pd.Series([1, 1, 0, 0])
        # tn = pd.Series([6, 8, 7, 7])
        # fp = pd.Series([1, 1, 2, 2])
        # fn = pd.Series([2, 0, 1, 1])

        expected_metric_per_user = pd.Series(
            [1, 1, 0, 0],
            index=pd.Series([1, 3, 4, 5], name=Columns.User, dtype=int), dtype=float
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
