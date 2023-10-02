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

"""Diversity metrics."""

import typing as tp
from itertools import combinations

import attr
import numpy as np
import pandas as pd

from rectools import Columns
from rectools.utils import select_by_type

from .base import MetricAtK
from .distances import PairwiseDistanceCalculator


@attr.s
class ILDFitted:
    """
    Container with meta data got from `IntraListDiversity.fit` method.

    Parameters
    ----------
    recommended_items_paired : pd.DataFrame
        Table with recommended item pairs,
        with columns ``item_0``, ``item_1``, ``rank_0``, ``rank_1``.
    users : np.ndarray
        Array of user ids.
    """

    recommended_items_paired: pd.DataFrame = attr.ib()
    users: np.ndarray = attr.ib()


@attr.s
class IntraListDiversity(MetricAtK):
    r"""
    Intra-List Diversity metric.

    Estimate average pairwise distance
    between items in user recommendations.

    .. math::
        ILD@k = (\sum_{i=1}^{k+1} \sum_{j=1}^{k+1} d(i, j)) / (k * (k-1))

    where
    - ``d(i, j)`` is distance between
    recommended items with rank ``i`` and rank ``j``.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    distance_calculator : PairwiseDistanceCalculator
        Distance calculator, object that returns distance
        between any item pair.

    Examples
    --------
    >>> from rectools.metrics.distances import PairwiseHammingDistanceCalculator
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 1, 2, 2],
    ...         Columns.Item: [1, 2, 3, 1, 4],
    ...         Columns.Rank: [1, 2, 3, 1, 2],
    ...     }
    ... )
    >>> features_df = pd.DataFrame(
    ...     [
    ...         [1, 0, 0],
    ...         [2, 0, 1],
    ...         [3, 1, 1],
    ...         [4, 0, 0],
    ...     ],
    ...     columns=[Columns.Item, "feature_1", "feature_2"]
    ... ).set_index(Columns.Item)
    >>> calculator = PairwiseHammingDistanceCalculator(features_df)
    >>> IntraListDiversity(k=1, distance_calculator=calculator).calc_per_user(reco).values
    array([0, 0])
    >>> IntraListDiversity(k=2, distance_calculator=calculator).calc_per_user(reco).values
    array([1., 0.])
    >>> IntraListDiversity(k=3, distance_calculator=calculator).calc_per_user(reco).values
    array([1.33333333, 0.        ])
    """

    distance_calculator: PairwiseDistanceCalculator = attr.ib()

    @classmethod
    def fit(
        cls,
        reco: pd.DataFrame,
        k_max: int,
    ) -> "ILDFitted":
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
        k_max : int
             k is number of items at the top of recommendations list that will be used to calculate metric.
             So `k_max` is maximum value of `k` metric for that you want to calculate.

        Returns
        -------
        ILDFitted
        """
        cls._check(reco)

        recommendations = reco.loc[reco[Columns.Rank] <= k_max]
        users = recommendations[Columns.User].unique()

        recommended_items_paired = (
            recommendations.groupby(Columns.User)[Columns.Item]
            .apply(lambda x: list(combinations(x, 2)))
            .reset_index()
            .explode(Columns.Item)
            .rename(columns={Columns.Item: "item_pair"})
            .dropna()
        )
        recommended_item_ranks_paired = (
            recommendations.groupby(Columns.User)[Columns.Rank]
            .apply(lambda x: list(combinations(x, 2)))
            .reset_index()
            .explode(Columns.Rank)
            .rename(columns={Columns.Rank: "rank_pair"})
            .dropna()
        )

        recommended_items_paired["item_0"] = recommended_items_paired["item_pair"].map(lambda pair: pair[0])
        recommended_items_paired["item_1"] = recommended_items_paired["item_pair"].map(lambda pair: pair[1])
        recommended_items_paired["rank_0"] = recommended_item_ranks_paired["rank_pair"].map(lambda pair: pair[0])
        recommended_items_paired["rank_1"] = recommended_item_ranks_paired["rank_pair"].map(lambda pair: pair[1])
        del recommended_item_ranks_paired, recommended_items_paired["item_pair"]

        return ILDFitted(recommended_items_paired, users)

    def calc_per_user_from_fitted(self, fitted: ILDFitted) -> pd.Series:
        """
        Calculate metric values for all users from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : ILDFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        if len(fitted.recommended_items_paired) == 0:
            return pd.Series(index=fitted.users, data=0)

        recommended_items_paired = fitted.recommended_items_paired
        recommended_items_paired["dist"] = self.distance_calculator[
            recommended_items_paired["item_0"].values,
            recommended_items_paired["item_1"].values,
        ]

        ild_at_k = (
            recommended_items_paired.loc[
                (recommended_items_paired["rank_0"] <= self.k) & (recommended_items_paired["rank_1"] <= self.k)
            ]
            .groupby(Columns.User)["dist"]
            .agg("mean")
        )
        present_users = ild_at_k.index.values
        ild_at_k_full = ild_at_k.reindex(fitted.users)
        ild_at_k_full.loc[~ild_at_k_full.index.isin(present_users)] = 0
        return ild_at_k_full.rename(None)

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

    def calc_from_fitted(self, fitted: ILDFitted) -> float:
        """
        Calculate metric value from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : ILDFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_fitted(fitted)
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
        fitted = self.fit(reco, k_max=self.k)
        return self.calc_per_user_from_fitted(fitted)


DiversityMetric = IntraListDiversity


def calc_diversity_metrics(
    metrics: tp.Dict[str, DiversityMetric],
    reco: pd.DataFrame,
) -> tp.Dict[str, float]:
    """
    Calculate diversity metrics (only IntraListDiversity now).

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> DiversityMetric)
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same with keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    # ILD
    ild_metrics: tp.Dict[str, IntraListDiversity] = select_by_type(metrics, IntraListDiversity)
    if ild_metrics:
        k_max = max(metric.k for metric in ild_metrics.values())
        fitted = IntraListDiversity.fit(reco, k_max)
        for name, metric in ild_metrics.items():
            results[name] = metric.calc_from_fitted(fitted)

    return results
