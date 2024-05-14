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

"""Serendipity is designed to find balance between novelty and relevance."""

import typing as tp

import attr
import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics.base import Catalog, MetricAtK
from rectools.utils import select_by_type


@attr.s
class SerendipityFitted:
    """
    Container with meta data got from `Serendipity.fit` method.

    Parameters
    ----------
    serendipity_values : pd.DataFrame
        Table with serendipity value for every recommended item,
        with columns `Columns.User`, `Columns.Rank`, ``serendipity``,
    users : np.ndarray
        Array of user ids.
    """

    serendipity_values: pd.DataFrame = attr.ib()
    users: np.ndarray = attr.ib()


@attr.s
class Serendipity(MetricAtK):
    r"""
    Serendipity metric.

    Evaluates novelty and relevance together.

    .. math::
        Serendipity@k = (\sum_{i=1}^{k} max(p(i) - pu(i), 0) * rel(i)) / k

    where
        - :math:`p(i) = (n\_items + 1 - i) / n\_items`
          is probability to recommend item with rank ``i``
          to current user;
        - :math:`pu(i) = (n\_items + 1 - popularity(i)) / n_items`
          is probability to recommend item with rank ``i``
          to any user;
        - :math:`rel(i)` is an indicator function, it equals to ``1``
          if the item at rank ``i`` is relevant, ``0`` otherwise;
        - :math:`n\_items` is an overall number of items
          that could be used for recommendations.
        - :math:`popularity(i)` is popularity rank of the
          `i`-th item in recommendations list.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list
        that will be used to calculate metric.

    Notes
    -----
    Method is inspired by the article:
    https://gab41.lab41.org/recommender-systems-its-not-all-about-the-accuracy-562c7dceeaff

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: ["u1", "u1", "u2", "u2", "u3", "u4", "u4"],
    ...         Columns.Item: ["i1", "i2", "i2", "i3", "i3", "i2", "i3"],
    ...         Columns.Rank: [   1,    2,    1,    2,    1,    1,    2],
    ...     }
    ... )
    >>> interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: ["u1", "u1", "u2", "u2", "u3", "u4"],
    ...         Columns.Item: ["i1", "i2", "i2", "i3", "i2", "i2"],
    ...     }
    ... )
    >>> prev_interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: ["u1", "u1", "u2", "u2", "u3"],
    ...         Columns.Item: ["i1", "i2", "i1", "i2", "i1"],
    ...     }
    ... )
    >>> catalog = ("i1", "i2", "i3", "i4")
    >>> Serendipity(k=1).calc_per_user(reco, interactions, prev_interactions, catalog).values
    array([0. , 0.25, 0. , 0.25])
    >>> Serendipity(k=2).calc_per_user(reco, interactions, prev_interactions, catalog).values
    array([0.  , 0.5 , 0.  , 0.125])
    """

    @classmethod
    def fit(
        cls,
        reco: pd.DataFrame,
        interactions: pd.DataFrame,
        prev_interactions: pd.DataFrame,
        catalog: Catalog,
        k_max: int,
    ) -> "SerendipityFitted":
        """
        Prepare intermediate data for effective calculation.

        You can use this method to prepare some intermediate data
        for later calculation. It can optimize calculations if
        you want calculate metric value for different `k`
        or `distance_calculator`.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.
        prev_interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.
        k_max : int
             k is number of items at the top of recommendations list that will be used to calculate metric.
             So `k_max` is maximum value of `k` parameter for which you want to calculate metric.

        Returns
        -------
        SerendipityFitted
        """
        cls._check(reco, interactions=interactions, prev_interactions=prev_interactions)

        recommendations = reco.loc[reco[Columns.Rank] <= k_max]

        recommendations_ = pd.merge(
            recommendations,
            interactions[[Columns.User, Columns.Item]],
            how="left",
            indicator=True,
        )
        recommendations_["is_relevant"] = np.where(recommendations_["_merge"] == "both", 1, 0)

        n_items = len(catalog)
        item_popularity_ranks = cls._get_item_popularity_ranks(prev_interactions)
        recommendations_["rank_pop"] = recommendations_[Columns.Item].map(item_popularity_ranks)

        recommendations_["proba_user"] = (n_items + 1 - recommendations_[Columns.Rank]) / n_items
        recommendations_["proba_any_user"] = np.where(
            recommendations_["rank_pop"].notnull(),
            (n_items + 1 - recommendations_["rank_pop"]) / n_items,
            0.0,  # zero probability for cold items
        )
        recommendations_["proba_diff"] = np.maximum(
            recommendations_["proba_user"] - recommendations_["proba_any_user"], 0.0
        )
        recommendations_["serendipity"] = recommendations_["proba_diff"] * recommendations_["is_relevant"]

        serendipity_values = recommendations_[[Columns.User, Columns.Rank, "serendipity"]]
        users = recommendations[Columns.User].unique()

        return SerendipityFitted(serendipity_values, users)

    @staticmethod
    def _get_item_popularity_ranks(interactions: pd.DataFrame) -> pd.Series:
        item_interaction_counts = interactions[Columns.Item].value_counts()
        counts_unique = item_interaction_counts.unique()
        count_rank_mapping = pd.Series(index=counts_unique, data=np.arange(len(counts_unique)) + 1)
        return item_interaction_counts.map(count_rank_mapping)

    def calc_per_user_from_fitted(self, fitted: SerendipityFitted) -> pd.Series:
        """
        Calculate metric values for all users from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : SerendipityFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        serendipity_at_k = (
            fitted.serendipity_values.loc[fitted.serendipity_values[Columns.Rank] <= self.k]
            .groupby(Columns.User)["serendipity"]
            .agg("mean")
        )
        return serendipity_at_k.reindex(fitted.users).rename(None)

    def calc(
        self,
        reco: pd.DataFrame,
        interactions: pd.DataFrame,
        prev_interactions: pd.DataFrame,
        catalog: Catalog,
    ) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.
        prev_interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions, prev_interactions, catalog)
        return per_user.mean()

    def calc_from_fitted(self, fitted: SerendipityFitted) -> float:
        """
        Calculate metric value from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : SerendipityFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_fitted(fitted)
        return per_user.mean()

    def calc_per_user(
        self,
        reco: pd.DataFrame,
        interactions: pd.DataFrame,
        prev_interactions: pd.DataFrame,
        catalog: Catalog,
    ) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.
        prev_interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        fitted = self.fit(reco, interactions, prev_interactions, catalog, k_max=self.k)
        return self.calc_per_user_from_fitted(fitted)


SerendipityMetric = Serendipity


def calc_serendipity_metrics(
    metrics: tp.Dict[str, SerendipityMetric],
    reco: pd.DataFrame,
    interactions: pd.DataFrame,
    prev_interactions: pd.DataFrame,
    catalog: Catalog,
) -> tp.Dict[str, float]:
    """
    Calculate serendipity metrics.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> SerendipityMetric)
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    interactions : pd.DataFrame
        Interactions table with columns `Columns.User`, `Columns.Item`.
    prev_interactions : pd.DataFrame
        Table with previous user-item interactions,
        with columns `Columns.User`, `Columns.Item`.
    catalog : collection
        Collection of unique item ids that could be used for recommendations.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same as keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    serendipity_metrics: tp.Dict[str, Serendipity] = select_by_type(metrics, Serendipity)
    if serendipity_metrics:
        k_max = max(metric.k for metric in serendipity_metrics.values())
        fitted = Serendipity.fit(reco, interactions, prev_interactions, catalog, k_max)
        for name, metric in serendipity_metrics.items():
            results[name] = metric.calc_from_fitted(fitted)

    return results
