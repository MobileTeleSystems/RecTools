#  Copyright 2022 MTS (Mobile Telesystems)
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

"""Base metric module."""

import typing as tp
import warnings

import attr
import pandas as pd

from rectools import Columns

ExternalItemId = tp.Union[str, int]
Catalog = tp.Collection[ExternalItemId]


@attr.s(auto_attribs=True)
class MetricAtK:
    """
    Base class of metrics that depends on `k` -
    a number of top recommendations used to calculate a metric.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    """

    k: int

    @classmethod
    def _check(
        cls,
        reco: pd.DataFrame,
        interactions: tp.Optional[pd.DataFrame] = None,
        prev_interactions: tp.Optional[pd.DataFrame] = None,
    ) -> None:
        cls._check_columns(reco, "reco", (Columns.User, Columns.Item, Columns.Rank))
        cls._check_columns(interactions, "interactions", (Columns.User, Columns.Item))
        cls._check_columns(prev_interactions, "prev_interactions", (Columns.User, Columns.Item))

        if reco[Columns.Rank].dtype.kind not in ("i", "u"):
            warnings.warn(f"Expected integer dtype of '{Columns.Rank}' column in 'reco' dataframe.")
        if int(round(reco[Columns.Rank].min())) != 1:
            warnings.warn(f"Expected min value of '{Columns.Rank}' column in 'reco' dataframe to be equal to 1.")

    @staticmethod
    def _check_columns(df: tp.Optional[pd.DataFrame], name: str, required_columns: tp.Iterable[str]) -> None:
        if df is None:
            return
        required_columns = set(required_columns)
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(f"Missed columns {required_columns - actual_columns} in '{name}' dataframe")


def merge_reco(reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge recommendation table with interactions table.

    Parameters
    ----------
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    interactions : pd.DataFrame
        Interactions table with columns `Columns.User`, `Columns.Item`.

    Returns
    -------
    pd.DataFrame
        Result of merging.
    """
    merged = pd.merge(
        interactions.reindex(columns=Columns.UserItem),
        reco.reindex(columns=Columns.UserItem + [Columns.Rank]),
        on=Columns.UserItem,
        how="left",
    )
    return merged


def outer_merge_reco(reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge recommendation table with interactions table with outer join and provide full ranks.
    This method is useful for AUC based metrics.

    Parameters
    ----------
    reco : pd.DataFrame
        Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    interactions : pd.DataFrame
        Interactions table with columns `Columns.User`, `Columns.Item`.

    Returns
    -------
    pd.DataFrame
        Result of merging with added `__test_positive` boolean column.
    """
    prepared_interactions = interactions.drop_duplicates(Columns.UserItem).reindex(columns=Columns.UserItem).copy()
    prepared_interactions["__test_positive"] = True
    test_users = prepared_interactions[Columns.User].drop_duplicates()
    prepared_reco = reco.merge(test_users, on=Columns.User, how="inner").reindex(
        columns=Columns.UserItem + [Columns.Rank]
    )
    merged = pd.merge(
        prepared_interactions,
        prepared_reco,
        on=Columns.UserItem,
        how="outer",
    )
    max_rank = prepared_reco.groupby(Columns.User)[Columns.Rank].max()
    full_ranks = max_rank.apply(lambda a: list(range(1, a + 1))).explode().rename(Columns.Rank)
    ranked_reco = merged.merge(full_ranks, on=[Columns.User, Columns.Rank], how="outer").sort_values(
        [Columns.User, Columns.Rank]
    )
    ranked_reco["__test_positive"] = ranked_reco["__test_positive"].fillna(False)
    return ranked_reco
