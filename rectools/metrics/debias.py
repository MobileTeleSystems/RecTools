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

import attr
import pandas as pd

from rectools import Columns
from rectools.utils import is_instance

from .base import Catalog
from .classification import (
    ClassificationMetric, 
    SimpleClassificationMetric, 
    Precision,
    Recall,
    F1Beta,
    MCC,
    Accuracy,
    # DebiasClassificationMetric,
    # DebiasSimpleClassificationMetric,
    # DebiasPrecisionWrapper, 
    # DebiasRecallMetric, 
    # DebiasF1BetaWrapper, 
    # DebiasMCCWrapper, 
    # DebiasAccuracyWrapper,
)
from .ranking import (
    RankingMetric, 
    MAP,
    NDCG,
    MRR,
    # DebiasRankingMetric
    # DebiasMAPWrapper, 
    # DebiasNDCGWrapper, 
    # DebiasMRRWrapper,
)


class DebiasWrapper:
    """
    Metric wrapper that creates debiased validation in case of strong popularity bias in test data.

    Warning: Wrapper works for the following metric type ClassificationMetric, SimpleClassificationMetric, RankingMetric

    TODO: formula

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
    ) -> tp.tp.Union[DebiasClassificationMetric, DebiasSimpleClassificationMetric, DebiasRankingMetric]:
        if not is_instance(metric, (ClassificationMetric, SimpleClassificationMetric, RankingMetric)):
            raise TypeError("Metric must be `ClassificationMetric` / `SimpleClassificationMetric` / `RankingMetric`")

        k = metrik.k

        if isinstance(metric, Precision):
            metric = DebiasPrecisionWrapper(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, Recall):
            metric = DebiasRecallWrapper(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, F1Beta):
            metric = DebiasF1BetaWrapper(k=k, beta=metrik.beta, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, MCC):
            metric = DebiasMCCWrapper(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, Accuracy):
            metric = DebiasAccuracyWrapper(k=k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, MAP):
            metric = DebiasMAPWrapper(k=k, divide_by_k=metric.divide_by_k, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, NDCG):
            metric = DebiasNDCGWrapper(k=k, log_base=metric.log_base, iqr_coef=iqr_coef, random_state=random_state)
        elif isinstance(metric, MRR):
            metric = DebiasMAPWrapper(k=k, iqr_coef=iqr_coef, random_state=random_state)
        return metric


# THINK ABOUT it
@attr.s
class DebiasMAPWrapper(MAP):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    @classmethod
    def fit(cls, merged: pd.DataFrame, k_max: int) -> MAPFitted:
        merged_wo_popularity = make_downsample(merged, self.iqr_coef, self.random_state)
        return MAP.fit(merge_reco=merged_wo_popularity, k_max=k_max)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity)


@attr.s
class DebiasNDCGWrapper(NDCG):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity)

    def calc_per_user_from_merged(self, merged: pd.DataFrame) -> pd.Series:
        merged_wo_popularity = make_downsample(merged, self.iqr_coef, self.random_state)
        super().calc_per_user_from_merged(merged=merged_wo_popularity)


@attr.s
class DebiasMRRWrapper(NDCG):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)    

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity)

    def calc_per_user_from_merged(self, merged: pd.DataFrame) -> pd.Series:
        merged_wo_popularity = make_downsample(merged, self.iqr_coef, self.random_state)
        super().calc_per_user_from_merged(merged=merged_wo_popularity)


@attr.s
class DebiasPrecisionWrapper(Precision):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity)


@attr.s
class DebiasRecallWrapper(Recall):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity)


@attr.s
class DebiasF1BetaWrapper(F1Beta):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity)


@attr.s
class DebiasAccuracyWrapper(Accuracy):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity, catalog=catalog)


@attr.s
class DebiasMCCaWrapper(MCC):

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> pd.Series:
        interactions_wo_popularity = make_downsample(interactions, self.iqr_coef, self.random_state)
        super().calc_per_user(reco=reco, interactions=interactions_wo_popularity, catalog=catalog)


DebiasRankingMetric = tp.Union[DebiasNDCGWrapper, DebiasMAPWrapper, DebiasMRRWrapper]
DebiasClassificationMetric: tp.Union[DebiasAccuracyWrapper, DebiasMCCWrapper]
DebiasSimpleClassificationMetric: tp.Union[DebiasPrecisionWrapper, DebiasRecallWrapper, DebiasF1BetaWrapper]


def make_downsample(interactions: pd.DataFrame, iqr_coef: float, random_state: int) -> pd.DataFrame:
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
