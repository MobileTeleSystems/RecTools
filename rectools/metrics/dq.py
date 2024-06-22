#  Copyright 2024 MTS (Mobile Telesystems)
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

"""Recommendations data quality metrics."""

import typing as tp

import attr
import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK
from rectools.utils import fast_isin_for_sorted_test_elements, select_by_type


@attr.s
class _RecoDQMetric(MetricAtK):
    """
    Recommendations data quality metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    deep: bool, default `False`
        Whether to calculated detailed value of the metric for each user. Otherwise just the share of
        users with identified problems will be returned (this is the default behaviour).
    """

    deep: bool = attr.ib(default=False)

    def calc(self, reco: pd.DataFrame) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        raise NotImplementedError()


class RecoEmpty(_RecoDQMetric):
    """
    Empty rows in recommendations table when `k` recommendations are required for each user.
    This metric helps to identify situations when recommendation lists are not fully filled.

    Parameters
    ----------
    k : int
        Number required recommendations  for each user that will be used to calculate metric.
    deep: bool, default `False`
        Whether to calculated detailed value of the metric for each user. Otherwise just the share of
        users with identified empty rows will be returned (this is the default behaviour).

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    ...         Columns.Item: [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
    ...         Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> RecoEmpty(k=1).calc_per_user(reco).values
    array([0, 0, 0])
    >>> RecoEmpty(k=4).calc_per_user(reco).values
    array([1, 1, 0])
    >>> RecoEmpty(k=4, deep=True).calc_per_user(reco).values
    array([0.5 , 0.25, 0.  ])
    """

    def calc_per_user(self, reco: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        reco_k = reco.query(f"{Columns.Rank} <= @self.k").drop_duplicates(subset=[Columns.User, Columns.Rank])
        all_users = reco[Columns.User].unique()
        n_unique_per_user = reco_k.groupby(Columns.User).size().reindex(all_users, fill_value=0)

        if self.deep:
            return 1 - n_unique_per_user / self.k

        return (n_unique_per_user < self.k).astype("int")


class RecoDuplicated(_RecoDQMetric):
    """
    Duplicated items recommended to the same user in recommendations table.
    This metrics help to identify situations when recommendation lists have duplicated items for users.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    deep: bool, default `False`
        Whether to calculated detailed value of the metric for each user. Otherwise just the share of
        users with identified problems will be returned (this is the default behaviour).

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    ...         Columns.Item: [1, 2, 1, 1, 3, 1, 2, 2, 1, 5],
    ...         Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> RecoDuplicated(k=1).calc_per_user(reco).values
    array([0, 0, 0])
    >>> RecoDuplicated(k=4).calc_per_user(reco).values
    array([0, 1, 1])
    >>> RecoDuplicated(k=4, deep=True).calc_per_user(reco).values
    array([0. , 0.25, 0.5  ])
    """

    def calc_per_user(self, reco: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        reco_k = reco.query(f"{Columns.Rank} <= @self.k").copy()
        reco_k["__duplicated"] = reco_k.duplicated(subset=Columns.UserItem)

        if self.deep:
            return reco_k.groupby(Columns.User)["__duplicated"].sum().rename(None) / self.k

        return reco_k.groupby(Columns.User)["__duplicated"].any().astype("int").rename(None)


class UsersNotCovered(MetricAtK):
    """
    Recommendations data quality metric to calsulate share of test users that are not present in
    recommendations table.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2],
    ...         Columns.Item: [1, 2, 1],
    ...         Columns.Rank: [1, 2, 2],
    ...     }
    ... )
    ... interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 2, 3, 4],
    ...         Columns.Item: [1, 1, 1, 1],
    ...     }
    ... )
    >>> UsersNotCovered(k=1).calc_per_user(reco, interactions).values
    array([0, 1, 1, 1])
    >>> UsersNotCovered(k=2).calc_per_user(reco, interactions).values
    array([0, 0, 1, 1])
    """

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> float:
        """
        Calculate metric value from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco, interactions=interactions)
        target_users = interactions[Columns.User].unique()
        reco_users = np.unique(reco[reco[Columns.Rank] <= self.k][Columns.User])
        not_covered = fast_isin_for_sorted_test_elements(target_users, reco_users, invert=True)
        res = pd.Series(not_covered, index=pd.Series(target_users, name=Columns.User), dtype="int")
        return res


DQMetric = tp.Union[RecoEmpty, RecoDuplicated, UsersNotCovered]


def calc_dq_metrics(
    metrics: tp.Dict[str, DQMetric],
    reco: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tp.Dict[str, float]:
    """
    Calculate recommendations data quality metrics.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> DQMetric)
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    interactions : pd.DataFrame, optional
        Interactions table with columns `Columns.User`, `Columns.Item`.
        Obligatory only for some types of metrics.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same as keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    for dq_metric_cls in [RecoDuplicated, RecoEmpty]:
        reco_dq_metrics: tp.Dict[str, tp.Union[RecoDuplicated, RecoEmpty]] = select_by_type(metrics, dq_metric_cls)
        if reco_dq_metrics:
            for name, reco_dq_metric in reco_dq_metrics.items():
                results[name] = reco_dq_metric.calc(reco)

    cross_dq_metrics: tp.Dict[str, UsersNotCovered] = select_by_type(metrics, UsersNotCovered)
    if cross_dq_metrics:
        if interactions is None:
            raise ValueError("For calculating `UsersNotCovered` metric it's necessary to set 'interactions'")
        for name, cross_dq_metric in cross_dq_metrics.items():
            results[name] = cross_dq_metric.calc(reco, interactions)

    return results
