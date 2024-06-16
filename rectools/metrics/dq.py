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

from rectools import Columns
from rectools.metrics.base import MetricAtK


@attr.s
class Completeness(MetricAtK):
    """
    Completeness metric.

    Calculate the ratio of provided recommendations to the total of required `k` for each user.
    This metrics help to identify situations when recommendation lists are not fully filled.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    ...         Columns.Item: [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
    ...         Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
    ...     }
    ... )
    >>> Completeness(k=1).calc_per_user(reco).values
    array([1., 1., 1.])
    >>> Completeness(k=4).calc_per_user(reco).values
    array([0.5 , 0.75, 1.  ])
    """

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
        reco_k = reco.query(f"{Columns.Rank} <= @self.k")
        num_recommended_per_user = reco_k.groupby(Columns.User).size()
        all_users = reco[Columns.User].unique()
        completeness_per_user = num_recommended_per_user.reindex(all_users, fill_value=0) / self.k
        return completeness_per_user


DQMetric = Completeness


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
    results = {name: metric.calc(reco) for name, metric in metrics.items()}
    return results
