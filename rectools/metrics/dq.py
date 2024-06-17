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

"""Recommendation data quality metrics."""

import typing as tp

import attr
import pandas as pd
import numpy as np

from rectools import Columns
from rectools.metrics.base import MetricAtK
from rectools.utils import fast_isin_for_sorted_test_elements

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
        users with identified problems will be returned (this is the default behaviour).

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
    array([0., 0., 0.])
    >>> RecoEmpty(k=4).calc_per_user(reco).values
    array([1., 1., 0.])
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
        
        return (n_unique_per_user < self.k).astype("float")


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
    array([0., 0., 0.])
    >>> RecoDuplicated(k=4).calc_per_user(reco).values
    array([0., 1., 1.])
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
        
        return reco_k.groupby(Columns.User)["__duplicated"].any().astype("float").rename(None)
    

class TestUsersNotCovered(MetricAtK):
    """
    Recommendations data quality metric to calsulate share of test users that are not present in
    recommendations table.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
        
    Examples
    --------
    TODO
    """

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame,) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame,) -> pd.Series:
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
        test_users = interactions[Columns.User].unique()
        reco_users = np.unique(reco[Columns.User])
        not_covered = fast_isin_for_sorted_test_elements(test_users, reco_users, invert=True)
        return pd.Series(not_covered, index=test_users, dtype="float")


DQMetric = tp.Union[RecoEmpty, RecoDuplicated, TestUsersNotCovered]


def calc_dq_metrics(
    metrics: tp.Dict[str, DQMetric],
    reco: pd.DataFrame,
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

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same as keys in `metrics`
        and values are metric calculation results.
    """
    # TODO: add TestUsersNotCoverded
    results = {name: metric.calc(reco) for name, metric in metrics.items()}
    return results
