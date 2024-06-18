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

import attr
import pandas as pd

from rectools import Columns

from .base import MetricAtK


@attr.s(frozen=True)
class DebiasConfig:
    """
    Config with debias method parameters.

    Parameters
    ----------
    iqr_coef : float, default 1.5
        Coefficient for defining as the maximum value inside the border.
    random_state : float, default None
        Pseudorandom number generator state to control the down-sampling.
    """

    iqr_coef: float = attr.ib(default=1.5)
    random_state: tp.Optional[int] = attr.ib(default=None)


@attr.s
class DibiasableMetrikAtK(MetricAtK):
    """
    Classification metric base class.

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

    def make_debias(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Downsample the size of interactions, excluding some interactions with popular items.

        Algorithm:

            1. Calculate item "popularity"
            (here: number of unique users that had interaction with the item) distribution from interactions;
            2. Find first (Q1) and third (Q3) quartiles in items "popularity" distribution;
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
            Downsampling interactions.
        """
        if len(interactions) == 0:
            return interactions

        num_users_interacted_with_item = interactions.drop_duplicates(subset=Columns.UserItem)[
            Columns.Item
        ].value_counts()

        quantiles = num_users_interacted_with_item.quantile(q=[0.25, 0.75])
        q1, q3 = quantiles.loc[0.25], quantiles.loc[0.75]
        iqr = q3 - q1
        max_border = int(q3 + self.debias_config.iqr_coef * iqr)

        item_outside_max_border = num_users_interacted_with_item[num_users_interacted_with_item > max_border].index

        mask_outside_max_border = interactions[Columns.Item].isin(item_outside_max_border)
        interactions_result = interactions[~mask_outside_max_border]
        interactions_downsampling = interactions[mask_outside_max_border]

        interactions_downsampling = (
            interactions_downsampling.sample(frac=1.0, random_state=self.debias_config.random_state)
            .groupby(Columns.Item)
            .head(max_border)
        )

        interactions_result = pd.concat([interactions_result, interactions_downsampling])

        interactions_result = pd.merge(
            interactions_result[Columns.UserItem],
            interactions,
            how="left",
            on=Columns.UserItem,
        )

        return interactions_result
