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

from .classification import (
    MCC,
    Accuracy,
    ClassificationMetric,
    DebiasAccuracy,
    DebiasClassificationMetric,
    DebiasF1Beta,
    DebiasMCC,
    DebiasPrecision,
    DebiasRecall,
    DebiasSimpleClassificationMetric,
    F1Beta,
    Precision,
    Recall,
    SimpleClassificationMetric,
)
from .ranking import MAP, MRR, NDCG, DebiasMAP, DebiasMRR, DebiasNDCG, DebiasRankingMetric, RankingMetric


class DebiasWrapper:
    """
    Metric wrapper that creates debiased validation in case of strong popularity bias in test data.

    Warning: Wrapper works for the following metric type ClassificationMetric, SimpleClassificationMetric, RankingMetric

    Parameters
    ----------
    metric : (ClassificationMetric | SimpleClassificationMetric | RankingMetric)
        Number of items at the top of recommendations list that will be used to calculate metric.
    iqr_coef : float, default 1.5
        Coefficient for defining as the maximum value inside the border.
    random_state: float, default 32
        Pseudorandom number generator state to control the down-sampling.
    """

    def __new__(
        cls,
        metric: tp.Union[ClassificationMetric, SimpleClassificationMetric, RankingMetric],
        iqr_coef: float = 1.5,
        random_state: int = 32,
    ) -> tp.Union[DebiasClassificationMetric, DebiasSimpleClassificationMetric, DebiasRankingMetric]:
        """
        TODO
        """
        if not is_instance(metric, (ClassificationMetric, SimpleClassificationMetric, RankingMetric)):
            raise TypeError("Metric must be `ClassificationMetric` / `SimpleClassificationMetric` / `RankingMetric`")

        k = metric.k

        if isinstance(metric, Precision):
            debias_metric = DebiasPrecision(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, Recall):
            debias_metric = DebiasRecall(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, F1Beta):
            debias_metric = DebiasF1Beta(k=k, beta=metric.beta, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, MCC):
            debias_metric = DebiasMCC(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, Accuracy):
            debias_metric = DebiasAccuracy(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, MAP):
            debias_metric = DebiasMAP(k=k, divide_by_k=metric.divide_by_k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, NDCG):
            debias_metric = DebiasNDCG(k=k, log_base=metric.log_base, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, MRR):
            debias_metric = DebiasMRR(k=k, iqr_coef=iqr_coef, random_state=random_state)

        debias_metric.make_downsample = cls.make_downsample
        return debias_metric

    @classmethod
    def make_downsample(cls, interactions: pd.DataFrame, iqr_coef: float, random_state: int) -> pd.DataFrame:
        """
        Downsample the size of interactions, excluding some interactions with popular items.

        Algorithm: TODO

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
        max_border = q3 + iqr_coef * iqr

        item_outside_max_border = item_popularity[item_popularity > max_border].index

        interactions_result = interactions[~interactions[Columns.Item].isin(item_outside_max_border)]
        interactions_downsampling = interactions[interactions[Columns.Item].isin(item_outside_max_border)]

        interactions_downsampling = (
            interactions_downsampling.groupby(Columns.Item, as_index=False)[Columns.User]
            .agg(lambda users: users.sample(2, random_state=random_state).tolist())
            .explode(Columns.User)
        )

        interactions_result = pd.concat([interactions_result, interactions_downsampling]).sort_values(
            Columns.User, ignore_index=True
        )

        if Columns.Rank in interactions.columns:
            interactions_result = pd.merge(
                interactions_result,
                interactions,
                how="left",
                on=Columns.UserItem,
            )

        return interactions_result
