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

"""Debias module."""

import typing as tp
from collections import defaultdict

import attr
import pandas as pd

from rectools import Columns

from .base import MetricAtK


@attr.s(frozen=True)
class DebiasConfig:
    """
    Config for debiasing method parameters.

    Parameters
    ----------
    iqr_coef : float, default 1.5
        The interquartile range (IQR) coefficient required to calculate the maximum accepted popularity border
        (Q3 + iqr_coef * IQR), which is necessary to down-sample every item to a value that does not exceed it.
    random_state : int, optional, default None
        Pseudorandom number generator state to control the down-sampling.
    """

    iqr_coef: float = attr.ib(default=1.5)
    random_state: tp.Optional[int] = attr.ib(default=None)


@attr.s
class DebiasableMetrikAtK(MetricAtK):
    """
    Debiasing metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    debias_config : DebiasConfig, default None
        Config with debias method parameters (iqr_coef, random_state).
    """

    debias_config: DebiasConfig = attr.ib(default=None)

    def _check_debias(self, is_debiased: bool, obj_name: str) -> None:
        if not is_debiased and self.debias_config is not None:
            raise ValueError(
                "You have specified `debias_config` for metric "
                f"but `{obj_name}` is not de-biased. "
                f"Please make de-biasing for `{obj_name}` "
                "and specify `is_debiased` as `True` "
                "or otherwise use `calc` and `calc_per_user` methods for auto de-biasing."
            )


def debias_interactions(interactions: pd.DataFrame, config: DebiasConfig) -> pd.DataFrame:
    """
    Down-sample the size of interactions, excluding some interactions with popular items.

    Algorithm:

        1. Calculate item "popularity"
        (here: number of unique users that had interaction with the item) distribution from interactions;
        2. Find first (Q1) and third (Q3) quartiles in items "popularity" distribution;
        3. Calculate interquartile range (IQR) = Q3 - Q1;
        4. Calculate maximum value inside by formula: Q3 + iqr_coef * IQR;
        5. Down-sample for all exceeding items in interactions,
        randomly keeping the maximum group of users to a size not exceeding
        maximum value inside

    Parameters
    ----------
    interactions : pd.DataFrame
        Table with previous user-item interactions,
        with columns `Columns.User`, `Columns.Item`.
    config : DebiasConfig
        Config with debias method parameters (iqr_coef, random_state).

    Returns
    -------
    pd.DataFrame
        Downsampling interactions.
    """
    if len(interactions) == 0:
        return interactions

    interactions_for_debiasing = interactions.copy()

    num_users_interacted_with_item = interactions_for_debiasing.groupby(Columns.Item, sort=False)[
        Columns.User
    ].nunique()

    quantiles = num_users_interacted_with_item.quantile(q=[0.25, 0.75])
    q1, q3 = quantiles.loc[0.25], quantiles.loc[0.75]
    iqr = q3 - q1
    max_border = int(q3 + config.iqr_coef * iqr)

    item_outside_max_border = num_users_interacted_with_item[num_users_interacted_with_item > max_border].index

    mask_outside_max_border = interactions_for_debiasing[Columns.Item].isin(item_outside_max_border)
    interactions_result = interactions_for_debiasing[~mask_outside_max_border]
    interactions_downsampling = interactions_for_debiasing[mask_outside_max_border]

    interactions_downsampling = (
        interactions_downsampling.sample(frac=1.0, random_state=config.random_state)
        .groupby(Columns.Item)
        .head(max_border)
    )

    result_dfs = [interactions_result, interactions_downsampling]
    interactions_result = pd.concat(result_dfs, ignore_index=True)

    return interactions_result


def calc_debiased_fit_task(
    metrics: tp.Iterable[DebiasableMetrikAtK],
    interactions: pd.DataFrame,
    prev_debiased_interactions: tp.Optional[tp.Dict[DebiasConfig, pd.DataFrame]] = None,
) -> tp.Dict[DebiasConfig, tp.Tuple[int, pd.DataFrame]]:
    """
    Calculate for each of the unique debias configs `k_max` and debiased `interactions`
    to then apply them in the `fit` methods of the corresponding metrics.

    Parameters
    ----------
    metrics : tp.Iteraple[DebiasableMetrikAtK]
        Dict of metric objects to calculate, where key is metric name and value is metric object.
    interactions : pd.DataFrame
        Interactions or merging table with columns `Columns.User`, `Columns.Item`, `Columns.Rank` (for merging).
        Obligatory only for some types of metrics.
    prev_debiased_interactions : dict(DebiasConfig->pd.DataFrame]), optinonal
        Debiased interactions for certain debias configs calculated earlier.

    Returns
    -------
    dict(DebiasConfig->list[(int | pd.DataFrame)])
        Dictionary, where key is debias config
        and values are a tuple of the corresponding `k_max` and debiased `interactions`.
    """
    debiased_interactions = debias_for_metric_configs(metrics, interactions, prev_debiased_interactions)

    max_k_for_config: tp.Dict[DebiasConfig, int] = defaultdict(int)
    for metric in metrics:
        max_k_for_config[metric.debias_config] = max(max_k_for_config[metric.debias_config], metric.k)

    result = {
        config: (max_k_for_config[config], d_interactions) for config, d_interactions in debiased_interactions.items()
    }
    return result


def debias_for_metric_configs(
    metrics: tp.Iterable[DebiasableMetrikAtK],
    interactions: pd.DataFrame,
    prev_debiased_interactions: tp.Optional[tp.Dict[DebiasConfig, pd.DataFrame]] = None,
) -> tp.Dict[DebiasConfig, pd.DataFrame]:
    """
    Calculate for each of the unique debias configs debiased `interactions`.

    Parameters
    ----------
        metrics : tp.Iterable[DebiasableMetrikAtK]
            List of metrics to calculate debiased differential metrics for.
        interactions : pd.DataFrame
            List of interactions to calculate debiased differential metrics for.
        prev_debiased_interactions : dict(DebiasConfig->pd.DataFrame]), optinonal
            Debiased interactions for certain debias configs calculated earlier.

    Returns
    -------
    dict(DebiasConfig->pd.DataFrame])
        Dictionary, where key is debias config and values are debiased `interactions`.
    """
    configs_new = set(metric.debias_config for metric in metrics)
    if prev_debiased_interactions is not None:
        configs_new -= set(prev_debiased_interactions.keys())

    debiased_interactions = {
        config: debias_interactions(interactions, config) if config is not None else interactions
        for config in configs_new
    }
    if prev_debiased_interactions is not None:
        debiased_interactions = {**prev_debiased_interactions, **debiased_interactions}

    return debiased_interactions
