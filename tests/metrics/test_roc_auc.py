"""
import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import LAUC
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
    }
)

USER_RECO_LIST = USER_RECO[Columns.Item]
USER_RECO_SIMPLE_LIST = USER_RECO_SIMPLE[Columns.Item]
EMPTY_USER_RECO_LIST = np.array([], dtype=float)

RECO_SIMPLE = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 2, 2, 2, 3, 3, 3],
        Columns.Item: [1, 2, 3, 4, 5, 6, 3, 5, 7],
        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
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

USER_INTERACTIONS_SET = set(USER_INTERACTIONS[Columns.Item].unique())
USER_INTERACTIONS_SIMPLE_SET = set(USER_INTERACTIONS_SIMPLE[Columns.Item].unique())

INTERACTIONS_SIMPLE = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 2, 2, 2, 2],
        Columns.Item: [4, 2, 8, 4, 5, 6, 7],
    }
)

CATALOG = list(range(20))
CATALOG_SIMPLE = list(range(1, 9))

EMPTY_INTERACTIONS = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)


class TestRoc:
    def setup(self) -> None:
        self.k = 7
        self.metric = LAUC(self.k)
        self.metric_simple = LAUC(k=3)

    @staticmethod
    def _make_extended_confusions(
        reco: pd.DataFrame, interactions: pd.DataFrame, k: int, catalog: Catalog
    ) -> pd.DataFrame:
        confusion_df = make_confusions(reco, interactions, k)
        if TN not in confusion_df:
            confusion_df[TN] = len(catalog) - k - confusion_df[FN]
        return confusion_df

    def test_tpr_fpr(self) -> None:

        expected_tpr = np.zeros(self.k + 2)
        expected_fpr = np.zeros(self.k + 2)
        expected_tpr[self.k + 1] = 1.0
        expected_fpr[self.k + 1] = 1.0

        for roc_k in range(1, self.k + 1):
            confusion_df = self._make_extended_confusions(USER_RECO, USER_INTERACTIONS, roc_k, CATALOG)
            expected_tpr[roc_k] = confusion_df[TP] / (confusion_df[TP] + confusion_df[FN])
            expected_fpr[roc_k] = confusion_df[FP] / (confusion_df[FP] + confusion_df[TN])

        tpr, fpr = self.metric.calc_user_tpr_fpr(USER_RECO_LIST, USER_INTERACTIONS_SET, CATALOG)

        assert np.array_equal(expected_tpr, tpr)
        assert np.array_equal(expected_fpr, fpr)

    def test_auc_for_user(self) -> None:
        expected_auc = 7 / 15
        eps = 1e-6
        assert (
            np.abs(
                expected_auc
                - self.metric_simple.calc_for_user(1, USER_RECO_SIMPLE, USER_INTERACTIONS_SIMPLE, CATALOG_SIMPLE)
            )
            < eps
        )

    def test_auc(self) -> None:
        expected_auc = (7 / 15 + 0.875) / 2
        eps = 1e-6
        assert np.abs(expected_auc - self.metric_simple.calc(RECO_SIMPLE, INTERACTIONS_SIMPLE, CATALOG_SIMPLE)) < eps

    @pytest.mark.xfail(raises=ValueError)
    def user_not_in_reco_users(self) -> None:
        with pytest.raises(ValueError):
            self.metric_simple.calc_for_user(10, RECO_SIMPLE, EMPTY_INTERACTIONS, CATALOG_SIMPLE)

    @pytest.mark.xfail(raises=ValueError)
    def k_is_more_than_recommended_items(self) -> None:
        with pytest.raises(ValueError):
            self.metric.calc_user_tpr_fpr(USER_RECO_SIMPLE_LIST, USER_INTERACTIONS_SIMPLE_SET, CATALOG)

    def test_when_no_interactions(self) -> None:
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(
            self.metric.calc_per_user(RECO_SIMPLE, EMPTY_INTERACTIONS, CATALOG_SIMPLE), expected_metric_per_user
        )
        assert np.isnan(self.metric.calc(RECO_SIMPLE, EMPTY_INTERACTIONS, CATALOG_SIMPLE))
"""
