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

import attr
import pandas as pd

from rectools import Columns


@attr.s
class DebiasConfig:
    """
    Config with debias method parameters.

    Parameters
    ----------
    iqr_coef : float, default 1.5
        Coefficient for defining as the maximum value inside the border.
    random_state : float, default 32
        Pseudorandom number generator state to control the down-sampling.
    """

    iqr_coef: float = attr.ib(default=1.5)
    random_state: int = attr.ib(default=32)


def make_downsample(interactions: pd.DataFrame, debias_config: DebiasConfig) -> pd.DataFrame:
    """
    Downsample the size of interactions, excluding some interactions with popular items.

    Algorithm:

        1. Calculate item popularity distribution from interactions;
        2. Find first (Q1) and third (Q3) quartiles in items popularity distribution;
        3. Calculate IQR = Q3 - Q1;
        4. Calculate maximum value inside by formula: Q3 + iqr_coef * IQR;
        5. Down-sample for all exceeding items in interactions,
        randomly keeping the maximum group of users to a size not exceeding
        maximum value inside

    Parameters
    ----------
    interactions : pd.DataFrame
        Table with previous user-item interactions,
        with columns `Columns.User`, `Columns.Item`.
    iqr_coef : float, default 1.5
        Coefficient for defining as the maximum value inside the border.
    random_state: float, default 32
        Pseudorandom number generator state to control the down-sampling.

    Returns
    -------
    pd.DataFrame
        downsampling interactions.
    """
    if len(interactions) == 0:
        return interactions

    item_popularity = interactions[Columns.Item].value_counts()

    quantiles = item_popularity.quantile(q=[0.25, 0.75])
    q1, q3 = quantiles.loc[0.25], quantiles.loc[0.75]
    iqr = q3 - q1
    max_border = int(q3 + debias_config.iqr_coef * iqr)

    item_outside_max_border = item_popularity[item_popularity > max_border].index

    interactions_result = interactions[~interactions[Columns.Item].isin(item_outside_max_border)]
    interactions_downsampling = interactions[interactions[Columns.Item].isin(item_outside_max_border)]

    interactions_downsampling = (
        interactions_downsampling.groupby(Columns.Item, as_index=False)[Columns.User]
        .agg(lambda users: users.sample(max_border, random_state=debias_config.random_state).tolist())
        .explode(Columns.User)
    )

    interactions_result = pd.concat([interactions_result, interactions_downsampling]).sort_values(
        Columns.User, ignore_index=True
    )

    interactions_result[Columns.User] = interactions_result[Columns.User].astype(interactions[Columns.User].dtypes)
    interactions_result[Columns.Item] = interactions_result[Columns.Item].astype(interactions[Columns.Item].dtypes)

    if Columns.Rank in interactions.columns:
        interactions_result = pd.merge(
            interactions_result[Columns.UserItem],
            interactions,
            how="left",
            on=Columns.UserItem,
        )
        interactions_result[Columns.Rank] = interactions_result[Columns.Rank].astype(interactions[Columns.Rank].dtypes)

    return interactions_result
