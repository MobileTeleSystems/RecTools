from typing import Optional

import attr
import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK
from rectools.metrics.classification import Recall


@attr.s
class Intersection(MetricAtK):
    """
    Metric to measure intersection in user-item pairs between recommendation lists.

    The intersection@k equals the share of ``reco`` that is present in ``ref_reco``.

    This corresponds to the following algorithm:
        1) filter ``reco`` by ``k``
        2) filter ``ref_reco`` by ``ref_k``
        3) calculate the proportion of items in ``reco`` that are also present in ``ref_reco``
    The second and third steps are equivalent to computing Recall@ref_k when:
        - Interactions consists of ``reco`` without the `Columns.Rank` column.
        - Recommendation table is ``ref_reco``

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
        ref_k : int, optional
            Number of items in top of reference recommendations list that will be used to calculate metric.
            If ``ref_k`` is None than ``ref_reco`` will be filtered with ``ref_k = k``. Default: None.

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
        ref_k : int, optional
            Number of items in top of reference recommendations list that will be used to calculate metric.
            If ``ref_k`` is None than ``ref_reco`` will be filtered with ``ref_k = self.k``. Default: None.

        Returns
        -------
        pd.Series:
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco)
        assert set(ref_reco.columns) >= {Columns.User, Columns.Item, Columns.Rank}

        if ref_reco.shape[0] == 0:
            return pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)

        filtered_reco = reco[reco[Columns.Rank] <= self.k]

        if not ref_k:
            ref_k = self.k
        recall = Recall(k=ref_k)

        return recall.calc_per_user(ref_reco, filtered_reco[Columns.UserItem])
