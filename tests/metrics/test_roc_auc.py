import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics import AUC
from rectools.metrics.base import Catalog
from rectools.metrics.classification import make_confusions

TP = "__TP"
FP = "__FP"
FN = "__FN"
TN = "__TN"


USER_RECO = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 1, 1, 1, 1],
        Columns.Item: [1, 2, 3, 4, 5, 6, 7],
        Columns.Rank: [1, 2, 3, 4, 5, 6, 7],
    }
)

USER_RECO_SIMPLE = pd.DataFrame(
    {
        Columns.User: [1, 1, 1],
        Columns.Item: [1, 2, 3],
        Columns.Rank: [1, 2, 3],
        Columns.Score: [3, 2, 1],
    }
)

USER_INTERACTIONS = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 1],
        Columns.Item: [4, 2, 8, 9],
    }
)

USER_INTERACTIONS_SIMPLE = pd.DataFrame(
    {
        Columns.User: [1, 1, 1],
        Columns.Item: [4, 2, 8],
    }
)

CATALOG = list(range(20))
CATALOG_SIMPLE = list(range(1, 9))


class TestRoc:
    def setup(self) -> None:
        self.k = 7
        self.metric = AUC(self.k)

    def _make_extended_confusions(
        self, reco: pd.DataFrame, interactions: pd.DataFrame, k: int, catalog: Catalog
    ) -> tp.Tuple[np.array, np.array]:
        confusion_df = make_confusions(reco, interactions, k)
        if TN not in confusion_df:
            confusion_df[TN] = len(catalog) - k - confusion_df[FN]
        return confusion_df

    def test_roc_optimization(self) -> None:

        expected_tpr = np.zeros(self.k + 2)
        expected_fpr = np.zeros(self.k + 2)
        expected_tpr[self.k + 1] = 1.0
        expected_fpr[self.k + 1] = 1.0

        for roc_k in range(1, self.k + 1):
            confusion_df = self._make_extended_confusions(USER_RECO, USER_INTERACTIONS, roc_k, CATALOG)
            expected_tpr[roc_k] = confusion_df[TP] / (confusion_df[TP] + confusion_df[FN])
            expected_fpr[roc_k] = confusion_df[FP] / (confusion_df[FP] + confusion_df[TN])

        tpr, fpr = self.metric._calc_tpr_fpr(USER_RECO, USER_INTERACTIONS, CATALOG)

        assert np.array_equal(expected_tpr, tpr)
        assert np.array_equal(expected_fpr, fpr)

    def test_auc(self) -> None:
        expected_auc = 7 / 15
        assert expected_auc == self.metric.calc_for_user(1, USER_RECO_SIMPLE, USER_INTERACTIONS_SIMPLE, CATALOG_SIMPLE)
