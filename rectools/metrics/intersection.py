from typing import Optional

import attr
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK
from rectools.metrics.classification import Recall


@attr.s
class Intersection(MetricAtK):
    """
    Metric to measure intersection in user-item (or item-item) pairs between recommendation lists.

    Parameters
    ----------
    k : int
        Number of items in top of recommendations list that will be used to calculate metric.
    """

    def calc(self, reco: pd.DataFrame, ref_reco: pd.DataFrame, ref_k: Optional[int] = None) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        ref_reco : pd.DataFrame
            Reference recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        ref_k : Optional[int]
            Number of items in top of reference recommendations list that will be used to calculate metric.
            If k_res is None than ref_recos will be filtered with ref_k = self.k. Default: None.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, ref_reco, ref_k)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, ref_reco: pd.DataFrame, ref_k: Optional[int] = None) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        ref_reco : pd.DataFrame
            Reference recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        ref_k : Optional[int]
            Number of items in top of reference recommendations list that will be used to calculate metric.
            If k_res is None than ref_recos will be filtered with ref_k = self.k. Default: None.

        Returns
        -------
        pd.Series:
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco)
        self._check(ref_reco)

        if not ref_k:
            ref_k = self.k
        filtered_ref_reco = ref_reco[ref_reco[Columns.Rank] <= ref_k]

        recall = Recall(k=self.k)
        return recall.calc_per_user(reco, filtered_ref_reco[Columns.UserItem])
