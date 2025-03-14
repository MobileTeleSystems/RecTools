#  Copyright 2025 MTS (Mobile Telesystems)
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

"""Catalog statistics recommendations metrics."""

import typing as tp

import attr
import pandas as pd

from rectools import Columns

from .base import Catalog, MetricAtK


@attr.s
class CatalogCoverage(MetricAtK):
    """
    Count (or share) of items from catalog that is present in recommendations for all users.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    normalize: bool, default ``False``
        Flag, which says whether to normalize metric or not.
    """

    normalize: bool = attr.ib(default=False)

    def calc(self, reco: pd.DataFrame, catalog: Catalog) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        float
            Value of metric (aggregated for all users).
        """
        res = reco.loc[reco[Columns.Rank] <= self.k, Columns.Item].nunique()
        if self.normalize:
            return res / len(catalog)
        return res


CatalogMetric = CatalogCoverage


def calc_catalog_metrics(
    metrics: tp.Dict[str, CatalogMetric],
    reco: pd.DataFrame,
    catalog: Catalog,
) -> tp.Dict[str, float]:
    """
    Calculate metrics of catalog statistics for recommendations.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> CatalogMetric)
        Dict of metric objects to calculate,
        where key is a metric name and value is a metric object.
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    catalog : collection
        Collection of unique item ids that could be used for recommendations.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same as keys in `metrics`
        and values are metric calculation results.
    """
    return {metric_name: metric.calc(reco, catalog) for metric_name, metric in metrics.items()}
