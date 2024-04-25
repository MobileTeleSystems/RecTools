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


def debias_wrapper(
    metric: tp.Union[ClassificationMetric, SimpleClassificationMetric, RankingMetric],
    iqr_coef: float = 1.5,
    random_state: int = 32,
) -> tp.Union[DebiasClassificationMetric, DebiasSimpleClassificationMetric, DebiasRankingMetric]:
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
    k = metric.k

    debias_metric = None
    if isinstance(metric, Precision):
        debias_metric = DebiasPrecision(k=k, iqr_coef=iqr_coef, random_state=random_state)
    elif isinstance(metric, Recall):
        debias_metric = DebiasRecall(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore
    elif isinstance(metric, F1Beta):
        debias_metric = DebiasF1Beta(  # type: ignore
            k=k, beta=metric.beta, iqr_coef=iqr_coef, random_state=random_state
        )
    elif isinstance(metric, MCC):
        debias_metric = DebiasMCC(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore
    elif isinstance(metric, Accuracy):
        debias_metric = DebiasAccuracy(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore
    elif isinstance(metric, MAP):
        debias_metric = DebiasMAP(  # type: ignore
            k=k, divide_by_k=metric.divide_by_k, iqr_coef=iqr_coef, random_state=random_state
        )
    elif isinstance(metric, NDCG):
        debias_metric = DebiasNDCG(  # type: ignore
            k=k, log_base=metric.log_base, iqr_coef=iqr_coef, random_state=random_state
        )
    elif isinstance(metric, MRR):
        debias_metric = DebiasMRR(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore

    if debias_metric is not None:
        return debias_metric

    raise TypeError(
        "`metric` must be either  `ClassificationMetric` (`Precision`, `Recall`, `F1Beta`) "
        "or `SimpleClassificationMetric` (`MCC`, `Accuracy`) "
        "or `RankingMetric` (`MAP`, `NDCG`, `MRR`)."
    )
    if isinstance(metric, Precision):
        debias_metric = DebiasPrecision(k=k, iqr_coef=iqr_coef, random_state=random_state)
    elif isinstance(metric, Recall):
        debias_metric = DebiasRecall(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore
    elif isinstance(metric, F1Beta):
        debias_metric = DebiasF1Beta(  # type: ignore
            k=k, beta=metric.beta, iqr_coef=iqr_coef, random_state=random_state
        )
    elif isinstance(metric, MCC):
        debias_metric = DebiasMCC(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore
    elif isinstance(metric, Accuracy):
        debias_metric = DebiasAccuracy(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore
    elif isinstance(metric, MAP):
        debias_metric = DebiasMAP(  # type: ignore
            k=k, divide_by_k=metric.divide_by_k, iqr_coef=iqr_coef, random_state=random_state
        )
    elif isinstance(metric, NDCG):
        debias_metric = DebiasNDCG(  # type: ignore
            k=k, log_base=metric.log_base, iqr_coef=iqr_coef, random_state=random_state
        )
    elif isinstance(metric, MRR):
        debias_metric = DebiasMRR(k=k, iqr_coef=iqr_coef, random_state=random_state)  # type: ignore

    if debias_metric is not None:
        return debias_metric

    raise TypeError(
        "`metric` must be either  `ClassificationMetric` (`Precision`, `Recall`, `F1Beta`) "
        "or `SimpleClassificationMetric` (`MCC`, `Accuracy`) "
        "or `RankingMetric` (`MAP`, `NDCG`, `MRR`)."
    )
