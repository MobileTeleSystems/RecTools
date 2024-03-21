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

"""Novelty metrics."""

import typing as tp

import attr
import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK
from rectools.utils import select_by_type


@attr.s
class MIUFFitted:
    """
    Container with meta data got from `MeanInvUserFreq.fit` method.

    Parameters
    ----------
    item_novelties : pd.DataFrame
        Table with columns `Columns.User`, `Columns.Item`, ``item_novelty``.
    users : np.ndarray
        Array of user ids.
    """

    item_novelties: pd.DataFrame = attr.ib()
    users: np.ndarray = attr.ib()


@attr.s
class MeanInvUserFreq(MetricAtK):
    r"""
    Mean Inverse User Frequency metric.

    Estimate mean novelty of items in recommendations,
    where "novelty" of item is inversely proportional
    to the number of users who interacted with it.

    .. math::
        MIUF@k = -(\sum_{i=1}^{k} \log_{2} (users(i) / n\_users)) / k

    where
    - `users(i)` is number of users that previously interacted with item with rank `i`.
    - `n_users` is the overall number of users in previous interactions.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 2, 2, 3, 3],
    ...         Columns.Item: [3, 2, 3, 1, 2],
    ...         Columns.Rank: [1, 1, 2, 1, 2],
    ...     }
    ... )
    >>> prev_interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 3],
    ...         Columns.Item: [1, 2, 1, 1],
    ...     }
    ... )
    >>> MeanInvUserFreq(k=1).calc_per_user(reco, prev_interactions).values
    array([1.5849625, 1.5849625, 0. ])
    >>> MeanInvUserFreq(k=3).calc_per_user(reco, prev_interactions).values
    array([1.5849625 , 1.5849625 , 0.79248125])
    """

    @classmethod
    def fit(
        cls,
        reco: pd.DataFrame,
        prev_interactions: pd.DataFrame,
        k_max: int,
    ) -> MIUFFitted:
        """
        Prepare intermediate data for effective calculation.

        You can use this method to prepare some intermediate data
        for later calculation. It can optimize calculations if
        you want calculate metric for different values of `k`.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        prev_interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.
        k_max : int
             `k` is number of items at the top of recommendations list that will be used to calculate metric.
             So `k_max` is maximum value of `k` parameter for which you want to calculate metric.

        Returns
        -------
        MIUFFitted
        """
        cls._check(reco, prev_interactions=prev_interactions)
        n_interacted_users = prev_interactions[Columns.User].nunique()
        n_users_per_item = prev_interactions.groupby(Columns.Item)[Columns.User].nunique()

        recommendations_ = reco.loc[reco[Columns.Rank] <= k_max].copy()
        recommendations_["n_users_per_item"] = recommendations_[Columns.Item].map(n_users_per_item)
        # cold items are treated as if they were consumed by a single user
        recommendations_["n_users_per_item"] = recommendations_["n_users_per_item"].fillna(1)
        recommendations_["item_novelty"] = -np.log2(recommendations_["n_users_per_item"] / n_interacted_users)

        item_novelties = recommendations_[[Columns.User, Columns.Rank, "item_novelty"]]
        users = reco[Columns.User].unique()
        return MIUFFitted(item_novelties, users)

    def calc(self, reco: pd.DataFrame, prev_interactions: pd.DataFrame) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        prev_interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, prev_interactions)
        return per_user.mean()

    def calc_from_fitted(self, fitted: MIUFFitted) -> float:
        """
        Calculate metric value from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : MIUFFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_fitted(fitted)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, prev_interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        prev_interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        fitted = self.fit(reco, prev_interactions, k_max=self.k)
        return self.calc_per_user_from_fitted(fitted)

    def calc_per_user_from_fitted(self, fitted: MIUFFitted) -> pd.Series:
        """
        Calculate metric values for all users from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : MIUFFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        miuf_at_k = (
            fitted.item_novelties.loc[fitted.item_novelties[Columns.Rank] <= self.k]
            .groupby(Columns.User)["item_novelty"]
            .agg("mean")
        )
        return miuf_at_k.reindex(fitted.users).rename(None)


NoveltyMetric = MeanInvUserFreq


def calc_novelty_metrics(
    metrics: tp.Dict[str, NoveltyMetric],
    reco: pd.DataFrame,
    prev_interactions: pd.DataFrame,
) -> tp.Dict[str, float]:
    """
    Calculate novelty metrics (only MeanInvUserFreq now).

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> NoveltyMetric)
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    prev_interactions : pd.DataFrame
        Table with previous user-item interactions,
        with columns `Columns.User`, `Columns.Item`.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same as keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    # MIUF
    miuf_metrics: tp.Dict[str, MeanInvUserFreq] = select_by_type(metrics, MeanInvUserFreq)
    if miuf_metrics:
        k_max = max(metric.k for metric in miuf_metrics.values())
        fitted = MeanInvUserFreq.fit(reco, prev_interactions, k_max)
        for name, metric in miuf_metrics.items():
            results[name] = metric.calc_from_fitted(fitted)

    return results
