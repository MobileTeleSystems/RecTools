from typing import Dict, Hashable, Optional, Union

import attr
import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK
from rectools.metrics.classification import Recall
from rectools.utils import select_by_type


@attr.s(auto_attribs=True)
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
    ref_k : int, optional
        Number of items in top of reference recommendations list that will be used to calculate metric.
        If ``ref_k`` is None than ``ref_reco`` will be filtered with ``ref_k = k``. Default: None.
    """

    ref_k: Optional[int] = attr.ib(default=None)

    def calc(self, reco: pd.DataFrame, ref_reco: pd.DataFrame) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        ref_reco : pd.DataFrame
            Reference recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, ref_reco)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, ref_reco: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        ref_reco : pd.DataFrame
            Reference recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        pd.Series:
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco)
        assert set(ref_reco.columns) >= {Columns.User, Columns.Item, Columns.Rank}

        if ref_reco.shape[0] == 0:
            return pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)

        if ref_reco is reco:
            return pd.Series(
                data=1,
                index=pd.Series(data=reco[Columns.User].unique(), name=Columns.User, dtype=int),
                dtype=np.float64,
            )

        filtered_reco = reco[reco[Columns.Rank] <= self.k]

        if self.ref_k is None:
            self.ref_k = self.k
        recall = Recall(k=self.ref_k)

        return recall.calc_per_user(ref_reco, filtered_reco[Columns.UserItem])


IntersectionMetric = Intersection


def calc_intersection_metrics(
    metrics: Dict[str, Intersection],
    reco: pd.DataFrame,
    ref_reco: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
) -> Dict[str, float]:
    """
    Calculate intersection metrics.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> PopularityMetric)
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    ref_reco : Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]
        Reference recommendations table(s) with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same as keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    intersection_metrics: Dict[str, Intersection] = select_by_type(metrics, Intersection)
    if isinstance(ref_reco, pd.DataFrame):
        for name, metric in intersection_metrics.items():
            results[name] = metric.calc(reco, ref_reco)
    else:
        for name, metric in intersection_metrics.items():
            for key, ref_r in ref_reco.items():
                results[f"{name}_{key}"] = metric.calc(reco, ref_r)

    return results
