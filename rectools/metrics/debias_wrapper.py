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

"""Debias wrapper module."""

import typing as tp

import pandas as pd

from rectools import Columns
from rectools.utils import is_instance

from .base import Catalog
from .classification import ClassificationMetric, SimpleClassificationMetric
from .ranking import RankingMetric
from .scoring import calc_metrics


class DebiasWrapper:
    """
    Metric wrapper that creates debiased validation in case of strong popularity bias in test data.

    Warning: Wrapper works for the following metric type ClassificationMetric, SimpleClassificationMetric, RankingMetric

    Parameters
    ----------
    metric : (ClassificationMetric | SimpleClassificationMetric | RankingMetric)
        Number of items at the top of recommendations list that will be used to calculate metric.
    iqr_coef : float,
        coef TODO
    random_state: float,
        TODO
    """

    def __init__(
        self,
        metric: tp.Union[ClassificationMetric, SimpleClassificationMetric, RankingMetric],
        iqr_coef: float = 1.5,
        random_state: int = 32,
    ) -> None:

        if not is_instance(metric, (ClassificationMetric, SimpleClassificationMetric, RankingMetric)):
            raise TypeError("Metric must be `ClassificationMetric` / `SimpleClassificationMetric` / `RankingMetric`")

        self.metric = metric

        self.iqr_coef = iqr_coef
        self.random_state = random_state

    def downsampling(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Downsampling.

        Parameters
        ----------
        interactions : pd.DataFrame
            Table with previous user-item interactions,
            with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.DataFrame
            downsampling interactions.
        """
        item_popularity = interactions[Columns.Item].value_counts()

        quantiles = item_popularity.quantile(q=[0.25, 0.75])
        q1, q3 = quantiles.iloc[0.25], quantiles.iloc[0.75]
        iqr = q3 - q1
        max_border = q3 + self.iqr_coef * iqr

        item_outside_max_border = item_popularity[item_popularity > max_border].index

        interactions_result = interactions[~interactions[Columns.Item].isin(item_outside_max_border)]
        interactions_downsampling = interactions[interactions[Columns.Item].isin(item_outside_max_border)]

        interactions_downsampling = (
            interactions_downsampling.groupby(Columns.Item, as_index=False)[Columns.User]
            .agg(lambda users: users.sample(2, random_state=self.random_state).tolist())
            .explode(Columns.User)
        )

        interactions_result = pd.concat([interactions_result, interactions_downsampling]).sort_values(
            Columns.User, ignore_index=True
        )

        return interactions_result

    def calc(
        self,
        reco: pd.DataFrame,
        interactions: pd.DataFrame,
        catalog: tp.Optional[Catalog] = None,  # Only for Classification
    ) -> float:
        """
        Calculate metric as debias.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame, optional
            Interactions table with columns `Columns.User`, `Columns.Item`.
            Obligatory only for some types of metrics.
        catalog : collection, optional
            Collection of unique item ids that could be used for recommendations.
            Obligatory only if `ClassificationMetric` instance present in `metrics`.

        Returns
        -------
        float
            values are metric calculation result.
        """
        interactions_downsample = self.downsampling(interactions)

        result = calc_metrics(
            metrics={f"debiased_metric@{self.metric.k}": self.metric},
            reco=reco,
            interactions=interactions_downsample,
            catalog=catalog,
        )[f"debiased_metric@{self.metric.k}"]

        return result
