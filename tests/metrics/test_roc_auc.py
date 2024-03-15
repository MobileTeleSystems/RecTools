# pylint: disable=attribute-defined-outside-init

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics import LAUC
from rectools.metrics.base import Catalog
from rectools.metrics.classification import make_confusions

TP = "__TP"
FP = "__FP"
FN = "__FN"
TN = "__TN"

EMPTY_USER_RECO_LIST = np.array([], dtype=float)

RECO_SIMPLE = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 2, 2, 2, 3, 3, 3],
        Columns.Item: [1, 2, 3, 4, 5, 6, 3, 5, 7],
        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
    }
)

INTERACTIONS_SIMPLE = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 2, 2, 2, 2],
        Columns.Item: [4, 2, 8, 4, 5, 6, 7],
    }
)

CATALOG_SIMPLE = list(range(1, 9))

EMPTY_INTERACTIONS = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)


class TestRoc:
    def setup(self) -> None:
        self.metric_simple = LAUC(k=3)

    @staticmethod
    def _make_extended_confusions(
        reco: pd.DataFrame, interactions: pd.DataFrame, k: int, catalog: Catalog
    ) -> pd.DataFrame:
        confusion_df = make_confusions(reco, interactions, k)
        if TN not in confusion_df:
            confusion_df[TN] = len(catalog) - k - confusion_df[FN]
        return confusion_df

    def test_auc(self) -> None:
        expected_auc = (7 / 15 + 0.875) / 2
        eps = 1e-6
        assert np.abs(expected_auc - self.metric_simple.calc(RECO_SIMPLE, INTERACTIONS_SIMPLE, CATALOG_SIMPLE)) < eps
        assert self.metric_simple.calc_per_user(RECO_SIMPLE, INTERACTIONS_SIMPLE, CATALOG_SIMPLE)[2] == 0.875

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(
            self.metric_simple.calc_per_user(RECO_SIMPLE, EMPTY_INTERACTIONS, CATALOG_SIMPLE), expected_metric_per_user
        )
        assert np.isnan(self.metric_simple.calc(RECO_SIMPLE, EMPTY_INTERACTIONS, CATALOG_SIMPLE))
