# pylint: disable=attribute-defined-outside-init

import pandas as pd

from rectools import Columns
from rectools.metrics import ItemCoverage, NumRetrieved

RECO = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 1, 2, 2, 3, 4, 4],
        Columns.Item: [1, 2, 3, 4, 1, 2, 1, 1, 5],
        Columns.Rank: [1, 2, 3, 4, 1, 2, 1, 1, 2],
    }
)

CATALOG = list(range(10))


class TestItemCoverage:
    def setup(self) -> None:
        self.metric = ItemCoverage(k=3)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [0.3, 0.2, 0.1, 0.2],
            index=pd.Series([1, 2, 3, 4], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO, CATALOG), expected_metric_per_user)
        assert self.metric.calc(RECO, CATALOG) == 0.4


class TestNumRetrieved:
    def setup(self) -> None:
        self.metric = NumRetrieved(k=3)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [3, 2, 1, 2],
            index=pd.Series([1, 2, 3, 4], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO), expected_metric_per_user)
        assert self.metric.calc(RECO) == expected_metric_per_user.mean()
