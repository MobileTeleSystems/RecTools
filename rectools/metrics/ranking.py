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

"""Ranking recommendations metrics."""
import typing as tp

import attr
import numpy as np
import pandas as pd
from scipy import sparse

from rectools import Columns
from rectools.metrics.base import MetricAtK, merge_reco
from rectools.utils import log_at_base, select_by_type


@attr.s
class _RankingMetric(MetricAtK):
    """
    Simple classification metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    """

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        raise NotImplementedError()


@attr.s
class MAPFitted:
    """
    Container with meta data got from `MAP.fit` method.

    Parameters
    ----------
    precision_at_k : csr_matrix
        CSR matrix where rows corresponds to users,
        rows corresponds all possible k from 0 to `k_max`,
        and values are weighted precisions for relevant recommended items.
    users : np.ndarray
        Array of user ids.
    n_relevant_items : np.ndarray
        Tally of relevant items for each user.
        Users are in the same order as in `precision_at_k` matrix.
    """

    precision_at_k: sparse.csr_matrix = attr.ib()
    users: np.ndarray = attr.ib()
    n_relevant_items: np.ndarray = attr.ib()


@attr.s
class MAP(_RankingMetric):
    r"""
    Mean Average Precision at k (MAP@k).

    Mean AP calculates as mean value of AP among all users.

    Average Precision estimates precision of recommendations
    taking into account their order.

    .. math::
        AP@k = (\sum_{i=1}^{k+1} p@i * rel(i)) / divider

    where
        - `p@i` is ``precision at i``,
          see `Precision` metric documentation for details;
        - `rel(i)` is an indicator function, it equals to ``1``
          if the item at rank ``i`` is relevant, ``0`` otherwise;
        - `divider` can be equal to ``k`` or
          be equal to number of relevant items per user,
          depending on `divide_by_k` parameter.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    divide_by_k : bool, default False
        If ``True``, ``k`` will be used as divider in ``AP@k``.
        If ``False``, number of relevant items for each user will be used.

    Examples
    --------
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
    ...     }
    ... )
    >>> # Here
    >>> #    - for user ``1`` we return non-relevant recommendations;
    >>> #    - for user ``2`` we return 2 items and relevant is first;
    >>> #    - for user ``3`` we return 4 items, 1st, 3rd and 4th are relevant;
    >>> #    - for user ``4`` we return 3 items and all are relevant;
    >>> MAP(k=1).calc_per_user(reco, interactions).values
    array([0. , 1. , 0.33333333, 0.33333333])
    >>> MAP(k=3).calc_per_user(reco, interactions).values
    array([0. , 1. , 0.55555556, 1. ])
    >>> MAP(k=1, divide_by_k=True).calc_per_user(reco, interactions).values
    array([0., 1., 1., 1.])
    >>> MAP(k=3, divide_by_k=True).calc_per_user(reco, interactions).values
    array([0. , 0.33333333, 0.55555556, 1. ])
    """

    divide_by_k: bool = attr.ib(default=False)

    @classmethod
    def fit(cls, merged: pd.DataFrame, k_max: int) -> MAPFitted:
        """
        Prepare intermediate data for effective calculation.

        You can use this method to prepare some intermediate data
        for later calculation. It can optimize calculations if
        you want calculate metric value for different `k`.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.
        k_max : int
             k is number of items at the top of recommendations list that will be used to calculate metric.
             So `k_max` is maximum number of items for which you want to calculate metric.

        Returns
        -------
        MAPFitted
        """
        users = np.unique(merged[Columns.User])
        if users.size == 0:
            prec_at_k_csr = sparse.csr_matrix(np.array([]).reshape(0, 0))
            return MAPFitted(prec_at_k_csr, users, np.array([]))

        n_relevant_items = merged.groupby(Columns.User, sort=False)[Columns.Item].agg("size")[users].values

        user_to_idx_map = pd.Series(np.arange(users.size), index=users)
        df_prepared = merged.query(f"{Columns.Rank} <= @k_max")
        csr = sparse.csr_matrix(
            (
                np.ones(len(df_prepared)),
                (
                    df_prepared[Columns.User].map(user_to_idx_map),
                    df_prepared[Columns.Rank].round().astype(int),
                ),
            ),
            shape=(users.size, k_max + 1),  # +1 because numeration from 0, but ranks from 1
        )

        # Now let calc cumulative ranks - it's equal to number of relevant items at k
        # Here rows - users, columns - all possible k
        full_cumsum = np.cumsum(csr.data)
        n_row_elements = np.diff(csr.indptr)
        row_sums = np.asarray(csr.sum(axis=1)).ravel()
        sum_n_elements_in_prev_rows = np.repeat(
            # add [0] because no elements before first row
            np.concatenate((np.array([0]), np.cumsum(row_sums)[:-1])),
            n_row_elements,
        )

        n_relevant_items_at_k = csr
        n_relevant_items_at_k.data = full_cumsum - sum_n_elements_in_prev_rows

        # And finally calculate precision for every k
        counts = np.arange(k_max + 1)
        counts_indexed = counts[n_relevant_items_at_k.indices]
        prec_at_k = n_relevant_items_at_k
        prec_at_k.data = prec_at_k.data / counts_indexed

        return MAPFitted(prec_at_k, users, n_relevant_items)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco, interactions=interactions)
        merged_reco = merge_reco(reco, interactions)
        fitted = self.fit(merged_reco, k_max=self.k)
        return self.calc_per_user_from_fitted(fitted)

    def calc_per_user_from_fitted(self, fitted: MAPFitted) -> pd.Series:
        """
        Calculate metric values for all users from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : MAPFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        valid_precisions = fitted.precision_at_k[:, 1 : self.k + 1]
        sum_precisions = np.asarray(valid_precisions.sum(axis=1)).reshape(-1)
        if self.divide_by_k:
            sum_precisions = sum_precisions / self.k
        else:
            sum_precisions = sum_precisions / fitted.n_relevant_items
        avg_precisions = pd.Series(sum_precisions, index=pd.Series(fitted.users, name=Columns.User)).rename(None)
        return avg_precisions

    def calc_from_fitted(self, fitted: MAPFitted) -> float:
        """
        Calculate metric value from fitted data.

        For parameters used result of `fit` method.

        Parameters
        ----------
        fitted : MAPFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_fitted(fitted)
        return per_user.mean()


@attr.s
class NDCG(_RankingMetric):
    r"""
    Normalized Discounted Cumulative Gain at k (NDCG@k).

    Estimates relevance of recommendations taking in account their order.

    .. math::
        NDCG@k = DCG@k / IDCG@k
    where :math:`DCG@k = \sum_{i=1}^{k+1} rel(i) / log_{}(i+1)` -
    Discounted Cumulative Gain at k, main part of `NDCG@k`.

    The closer it is to the top the more weight it assigns to relevant items.
    Here:
    - `rel(i)` is an indicator function, it equals to ``1``
    if an item at rank `i` is relevant, ``0`` otherwise;
    - `log` - logarithm at any given base, usually ``2``.

    and :math:`IDCG@k = \sum_{i=1}^{k+1} (1 / log(i + 1))` -
    `Ideal DCG@k`, maximum possible value of `DCG@k`, used as
    normalization coefficient to ensure that `NDCG@k` values
    lie in ``[0, 1]``.

    Parameters
    ----------
     k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
     log_base : int, default ``2``
        Base of logarithm used to weight relevant items.

    Examples
    --------
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
    ...     }
    ... )
    >>> # Here
    >>> #    - for user ``1`` we return non-relevant recommendations;
    >>> #    - for user ``2`` we return 2 items and relevant is first;
    >>> #    - for user ``3`` we return 4 items, 1st, 3rd and 4th are relevant;
    >>> #    - for user ``4`` we return 3 items and all are relevant;
    >>> NDCG(k=1).calc_per_user(reco, interactions).values
    array([0., 1., 1., 1.])
    >>> NDCG(k=3).calc_per_user(reco, interactions).values
    array([0. , 0.46927873, 0.70391809, 1. ])
    """

    log_base: int = attr.ib(default=2)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco, interactions=interactions)
        merged_reco = merge_reco(reco, interactions)
        return self.calc_per_user_from_merged(merged_reco)

    def calc_from_merged(self, merged: pd.DataFrame) -> float:
        """
        Calculate metric value from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_merged(merged)
        return per_user.mean()

    def calc_per_user_from_merged(self, merged: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        dcg = (merged[Columns.Rank] <= self.k).astype(int) / log_at_base(merged[Columns.Rank] + 1, self.log_base)
        idcg = (1 / log_at_base(np.arange(1, self.k + 1) + 1, self.log_base)).sum()
        ndcg = (
            pd.DataFrame({Columns.User: merged[Columns.User], "__ndcg": dcg / idcg})
            .groupby(Columns.User, sort=False)["__ndcg"]
            .sum()
            .rename(None)
        )
        return ndcg


class MRR(_RankingMetric):
    r"""
    Mean Reciprocal Rank at k (MRR@k).

    MRR calculates as mean value of reciprocal rank
    of first relevant recommendation among all users.

    Estimates relevance of recommendations taking in account their order.

    .. math::
        MRR@K = \frac{1}{|U|} \sum_{i=1}^{|U|} \frac{1}{rank_{i}}

    where
        - :math:`{|U|}` is a number of unique users;
        - :math:`rank_{i}` is a rank of first relevant recommendation
          starting from ``1``.

    If a user doesn't have any relevant recommendation then his metric value will be ``0``.

    Parameters
    ----------
     k : int
        Number of items at the top of recommendations list that will be used to calculate metric.

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
    ...         Columns.Item: [7, 8, 1, 2, 2, 1, 3, 4, 7, 8, 3],
    ...         Columns.Rank: [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3],
    ...     }
    ... )
    >>> interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 3, 3, 3, 4, 4, 4],
    ...         Columns.Item: [1, 2, 1, 1, 3, 4, 1, 2, 3],
    ...     }
    ... )
    >>> # Here
    >>> #    - for user ``1`` we return non-relevant recommendations;
    >>> #    - for user ``2`` we return 2 items and relevant is first;
    >>> #    - for user ``3`` we return 4 items, 2nd, 3rd and 4th are relevant;
    >>> #    - for user ``4`` we return 3 items and relevant is last;
    >>> MRR(k=1).calc_per_user(reco, interactions).values
    array([0., 1., 0., 0.])
    >>> MRR(k=3).calc_per_user(reco, interactions).values
    array([0.        , 1.        , 0.5       , 0.33333333])
    """

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco, interactions=interactions)
        merged_reco = merge_reco(reco, interactions)
        return self.calc_per_user_from_merged(merged_reco)

    def calc_per_user_from_merged(self, merged: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        cutted_rank = np.where(merged[Columns.Rank] <= self.k, merged[Columns.Rank], np.nan)
        min_rank_per_user = (
            pd.DataFrame({Columns.User: merged[Columns.User], "__cutted_rank": cutted_rank})
            .groupby(Columns.User, sort=False)["__cutted_rank"]
            .min()
        )
        return (1.0 / min_rank_per_user).fillna(0).rename(None)

    def calc_from_merged(self, merged: pd.DataFrame) -> float:
        """
        Calculate metric value from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_merged(merged)
        return per_user.mean()


@attr.s
class PFound(_RankingMetric):
    r"""
    PFound at k (NDCG@k).

    Estimates relevance of recommendations taking in account their order.

    .. math::
        pFound@K = \sum_{i=1}^{k} pLook_{i}\ pRel_{i}

    where
        - :math:`pLook_{1} = 1`
        - :math:`pLook_{i} = pLook_{i-1}\ (1 - pRel_{i-1})\ (1 - pBreak)` is a probability
          of viewing the i-th recommendation from the list;
        - :math:`pRel_{i}` is a probability that the i-th item will be relevant;
        - :math:`PBreak = 0.15` (default value) is a probability that the user will stop browsing
          due to external reason.

    Parameters
    ----------
     k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
     p_break : float, default ``0.15``
        The probability that the user will stop browsing due to external reason.

    Examples
    --------
    >>> reco = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
    ...         Columns.Item: [7, 8, 1, 2, 1, 2, 3, 4, 1, 2, 3],
    ...         Columns.Rank: [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3],
    ...         Columns.Score: [0.9, 0.8, 0.92, 0.86, 0.9, 0.8, 0.7, 0.6, 0.86, 0.82, 0.8],
    ...     }
    ... )
    >>> interactions = pd.DataFrame(
    ...     {
    ...         Columns.User: [1, 1, 2, 3, 3, 3, 4, 4, 4],
    ...         Columns.Item: [1, 2, 1, 1, 3, 4, 1, 2, 3],
    ...     }
    ... )
    >>> # Here
    >>> #    - for user ``1`` we return non-relevant recommendations;
    >>> #    - for user ``2`` we return 2 items and relevant is first;
    >>> #    - for user ``3`` we return 4 items, 1st, 3rd and 4th are relevant;
    >>> #    - for user ``4`` we return 3 items and all are relevant;
    >>> PFound(k=1).calc_per_user(reco, interactions).values
    array([0.  , 0.92, 0.9 , 0.86])
    >>> PFound(k=3).calc_per_user(reco, interactions).values
    array([0.       , 0.92     , 0.950575 , 0.9721456])
    """

    p_break: float = attr.ib(default=0.15)

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`, `Columns.Score`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco, interactions=interactions)
        merged_reco = merge_reco(reco, interactions)
        return self.calc_per_user_from_merged(merged_reco)

    def calc_from_merged(self, merged: pd.DataFrame) -> float:
        """
        Calculate metric value from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_merged(merged)
        return per_user.mean()

    @staticmethod
    def add_missing_values(top_k: pd.DataFrame) -> pd.DataFrame:
        """
        Adding missing values for each user if the ranks have gaps
        because the evaluation of the value `pLook[i]`
        depends on the values `pLook[i - 1]`.

        Parameters
        ----------
        top_k : pd.DataFrame
            Result after selecting top k for each user.

        Returns
        -------
        pd.DataFrame
            Top k values without gaps for each user.

        Examples
        --------
        >>> top_k = pd.DataFrame(
        ...     {
        ...         Columns.User: [3, 3, 3, 4, 4, 5, 5],
        ...         Columns.Rank: [1, 2, 3, 1, 4, 3, 6],
        ...         Columns.Score: [0.88, 0.8, 0.76, 0.9, 0.6, 0.8, 0.5],
        ...     }
        ... )
        >>> PFound(k=10).add_missing_values(top_k)
            user_id  rank  score
        0         3     1   0.88
        1         3     2   0.80
        2         3     3   0.76
        3         4     1   0.90
        4         4     2   0.00
        5         4     3   0.00
        6         4     4   0.60
        7         5     1   0.00
        8         5     2   0.00
        9         5     3   0.80
        10        5     4   0.00
        11        5     5   0.00
        12        5     6   0.50
        """
        rank_per_user = top_k.groupby(Columns.User)[Columns.Rank].agg([len, list]).reset_index()
        rank_min_max = top_k.groupby(Columns.User)[Columns.Rank].agg(["min", "max"]).reset_index()
        top_k_missing_values_per_user = top_k[Columns.User].unique()[
            (rank_per_user["len"] != rank_min_max["max"] - rank_min_max["min"] + 1)
            | (((rank_min_max["max"] != 1) & (rank_min_max["min"] != 1)) & (rank_min_max["max"] == rank_min_max["min"]))
        ]

        missing_values = []
        for user in top_k_missing_values_per_user:
            missing_value_per_user = [
                (user, rank, 0)
                for rank in range(1, int(rank_min_max[rank_min_max[Columns.User] == user]["max"].iloc[0]))
                if rank not in rank_per_user[rank_min_max[Columns.User] == user]["list"].iloc[0]
            ]
            missing_values.extend(missing_value_per_user)

        top_k = (
            pd.concat([top_k, pd.DataFrame(missing_values, columns=top_k.columns)], ignore_index=True)
            .sort_values([Columns.User, Columns.Rank])
            .reset_index(drop=True)
        )

        return top_k

    def calc_per_user_from_merged(self, merged: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users from merged recommendations.

        Parameters
        ----------
        merged : pd.DataFrame
            Result of merging recommendations and interactions tables.
            Can be obtained using `merge_reco` function.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        if Columns.Score not in merged:
            raise KeyError("No 'Columns.Score' column for recommendations")

        if len(merged) != 0:
            user_without_relevance_item = pd.Series(
                data=0,
                index=merged[Columns.User].unique()[
                    (merged.groupby(Columns.User)[Columns.Rank].min() > self.k)
                    | (merged.groupby(Columns.User)[Columns.Rank].min().isna())
                ],
            )

            top_k = merged[merged[Columns.Rank] <= self.k]
            del top_k[Columns.Item]

            top_k = self.add_missing_values(top_k.copy())
            top_k["pLook"] = (1 - top_k.groupby(Columns.User)[Columns.Score].shift(fill_value=0)) * (1 - self.p_break)
            top_k.loc[top_k[Columns.Rank] == 1, "pLook"] = 1
            top_k["pLook"] = top_k.groupby(Columns.User)["pLook"].cumprod()
            top_k["pFound_by_item"] = top_k[Columns.Score] * top_k["pLook"]

            pfound_per_user = (
                top_k.groupby(Columns.User)["pFound_by_item"].sum().append(user_without_relevance_item).sort_index()
            )
            pfound_per_user.index.name = Columns.User

            return pfound_per_user
        else:
            return pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)


RankingMetric = tp.Union[NDCG, MAP, MRR, PFound]


def calc_ranking_metrics(
    metrics: tp.Dict[str, RankingMetric],
    merged: pd.DataFrame,
) -> tp.Dict[str, float]:
    """
    Calculate any ranking metrics (MAP, NDCG and MRR for now).

    Works with pre-prepared data.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> (MAP | NDCG | MRR))
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    merged : pd.DataFrame
        Result of merging recommendations and interactions tables.
        Can be obtained using `merge_reco` function.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same with keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    for ranking_metric_cls in [NDCG, MRR, PFound]:
        ranking_metrics: tp.Dict[str, tp.Union[NDCG, MRR, PFound]] = select_by_type(metrics, ranking_metric_cls)
        for name, metric in ranking_metrics.items():
            results[name] = metric.calc_from_merged(merged)

    map_metrics: tp.Dict[str, MAP] = select_by_type(metrics, MAP)
    if map_metrics:
        k_max = max(metric.k for metric in map_metrics.values())
        fitted = MAP.fit(merged, k_max)
        for name, map_metric in map_metrics.items():
            results[name] = map_metric.calc_from_fitted(fitted)

    return results
