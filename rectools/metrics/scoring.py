#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

"""Metrics calculation module."""

import typing as tp
import warnings

import pandas as pd

from rectools.utils import select_by_type

from .auc import AucMetric, calc_auc_metrics
from .base import Catalog, MetricAtK, merge_reco
from .classification import ClassificationMetric, SimpleClassificationMetric, calc_classification_metrics
from .diversity import DiversityMetric, calc_diversity_metrics
from .dq import CrossDQMetric, RecoDQMetric, calc_cross_dq_metrics, calc_reco_dq_metrics
from .intersection import IntersectionMetric, calc_intersection_metrics
from .novelty import NoveltyMetric, calc_novelty_metrics
from .popularity import PopularityMetric, calc_popularity_metrics
from .ranking import RankingMetric, calc_ranking_metrics
from .serendipity import SerendipityMetric, calc_serendipity_metrics


def calc_metrics(  # noqa  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    metrics: tp.Dict[str, MetricAtK],
    reco: pd.DataFrame,
    interactions: tp.Optional[pd.DataFrame] = None,
    prev_interactions: tp.Optional[pd.DataFrame] = None,
    catalog: tp.Optional[Catalog] = None,
    ref_reco: tp.Optional[tp.Union[pd.DataFrame, tp.Dict[tp.Hashable, pd.DataFrame]]] = None,
) -> tp.Dict[str, float]:
    """
    Calculate metrics.

    Parameters
    ----------
    metrics : dict(str -> Metric)
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    interactions : pd.DataFrame, optional
        Interactions table with columns `Columns.User`, `Columns.Item`.
        Obligatory only for some types of metrics.
    prev_interactions : pd.DataFrame
        Table with previous user-item interactions,
        with columns `Columns.User`, `Columns.Item`.
        Obligatory only for some types of metrics.
    catalog : collection, optional
        Collection of unique item ids that could be used for recommendations.
        Obligatory only if `ClassificationMetric` or `SerendipityMetric` instances present in `metrics`.
    ref_reco : Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]], optional
        Reference recommendations table(s) with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        For multiple intersection calculations we can pass multiple models recommendations in a dict:
        ``ref_reco = {"one": ref_reco_one, "two": ref_reco_two}``
        Obligatory only if `IntersectionMetric` instances present in `metrics`.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same with keys in `metrics`
        and values are metric calculation results.

    Raises
    ------
    ValueError
        If obligatory argument for some metric not set.

    Examples
    --------
    >>> from rectools import Columns
    >>> from rectools.metrics import Accuracy, NDCG
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
    ...         Columns.Item: [7, 8, 1, 2, 1, 2, 3, 4, 1, 2, 3],
    ...         Columns.Rank: [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3],
    ...     }
    ... )
    >>> interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 3, 3, 3, 4, 4, 4],
    ...         Columns.Item: [1, 2, 1, 1, 3, 4, 1, 2, 3],
    ...         Columns.Datetime: [1, 1, 1, 1, 1, 2, 2, 2, 2],
    ...     }
    ... )
    >>> split_dt = 2
    >>> df_train = interactions.loc[interactions[Columns.Datetime] < split_dt]
    >>> df_test = interactions.loc[interactions[Columns.Datetime] >= split_dt]
    >>> metrics = {
    ...     'ndcg@1': NDCG(k=1),
    ...     'accuracy@1': Accuracy(k=1)
    ... }
    >>> calc_metrics(
    ...     metrics,
    ...     reco=reco,
    ...     interactions=df_test,
    ...     prev_interactions=df_train,
    ...     catalog=df_train[Columns.Item].unique()
    ... )
    {'accuracy@1': 0.3333333333333333, 'ndcg@1': 0.5}
    """
    merged = None
    results = {}
    expected_results_len = len(metrics)

    # Classification
    classification_metrics = select_by_type(metrics, (ClassificationMetric, SimpleClassificationMetric))
    if classification_metrics:
        if interactions is None:
            raise ValueError("For calculating classification metrics it's necessary to set 'interactions'")
        merged = merge_reco(reco, interactions)
        classification_values = calc_classification_metrics(classification_metrics, merged, catalog)
        results.update(classification_values)

    # Ranking
    ranking_metrics = select_by_type(metrics, RankingMetric)
    if ranking_metrics:
        if interactions is None:
            raise ValueError("For calculating ranking metrics it's necessary to set 'interactions'")
        merged = merged if merged is not None else merge_reco(reco, interactions)
        ranking_values = calc_ranking_metrics(ranking_metrics, merged)
        results.update(ranking_values)

    # AUC based ranking
    auc_metrics = select_by_type(metrics, AucMetric)
    if auc_metrics:
        if interactions is None:
            raise ValueError("For calculating AUC-like metrics it's necessary to set 'interactions'")
        auc_values = calc_auc_metrics(auc_metrics, reco, interactions)
        results.update(auc_values)

    # Novelty
    novelty_metrics = select_by_type(metrics, NoveltyMetric)
    if novelty_metrics:
        if prev_interactions is None:
            raise ValueError("For calculating novelty metrics it's necessary to set 'prev_interactions'")
        novelty_values = calc_novelty_metrics(novelty_metrics, reco, prev_interactions)
        results.update(novelty_values)

    # Popularity
    popularity_metrics = select_by_type(metrics, PopularityMetric)
    if popularity_metrics:
        if prev_interactions is None:
            raise ValueError("For calculating popularity metrics it's necessary to set 'prev_interactions'")
        popularity_values = calc_popularity_metrics(popularity_metrics, reco, prev_interactions)
        results.update(popularity_values)

    # Diversity
    diversity_metrics = select_by_type(metrics, DiversityMetric)
    if diversity_metrics:
        diversity_values = calc_diversity_metrics(diversity_metrics, reco)
        results.update(diversity_values)

    # Serendipity
    serendipity_metrics = select_by_type(metrics, SerendipityMetric)
    if serendipity_metrics:
        if interactions is None:
            raise ValueError("For calculating serendipity metrics it's necessary to set 'interactions'")
        if prev_interactions is None:
            raise ValueError("For calculating serendipity metrics it's necessary to set 'prev_interactions'")
        if catalog is None:
            raise ValueError("For calculating serendipity metrics it's necessary to set 'catalog'")
        serendipity_values = calc_serendipity_metrics(
            serendipity_metrics,
            reco,
            interactions,
            prev_interactions,
            catalog,
        )
        results.update(serendipity_values)

    # Intersection
    intersection_metrics = select_by_type(metrics, IntersectionMetric)
    if intersection_metrics:
        if not ref_reco:
            raise ValueError("For calculating intersection metrics it's necessary to set 'ref_reco'")
        intersection_values = calc_intersection_metrics(
            intersection_metrics,
            reco,
            ref_reco,
        )
        results.update(intersection_values)
        expected_results_len += len(intersection_values) - len(intersection_metrics)

    # DQ
    cross_dq_metrics = select_by_type(metrics, CrossDQMetric)
    if cross_dq_metrics:
        if interactions is None:
            raise ValueError("For calculating some of the required DQ metrics it's necessary to set 'interactions'")
        cross_dq_values = calc_cross_dq_metrics(cross_dq_metrics, reco, interactions)
        results.update(cross_dq_values)

    reco_dq_metrics = select_by_type(metrics, RecoDQMetric)
    if reco_dq_metrics:
        reco_dq_values = calc_reco_dq_metrics(reco_dq_metrics, reco)
        results.update(reco_dq_values)

    if len(results) < expected_results_len:
        warnings.warn("Custom metrics are not supported.")

    return results
