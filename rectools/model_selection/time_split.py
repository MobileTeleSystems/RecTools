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

"""TimeRangeSplit."""

import typing as tp
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import sparse

from rectools import Columns
from rectools.dataset import Interactions
from rectools.utils import pairwise

DateRange = tp.Sequence[tp.Union[date, datetime]]


class TimeRangeSplit:
    def __init__(
        self,
        date_range: DateRange,
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        self.date_range = date_range
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

    def split(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        """
        Split interactions into folds.
        Parameters
        ----------
        df: pd.DataFrame
            User-item interactions.
            Obligatory columns: `Columns.User`, `Columns.Item`, `Columns.Datetime`.
        collect_fold_stats: bool, default False
            Add some stats to fold info,
            like size of train and test part, number of users and items.
        Returns
        -------
        iterator(array, array, dict)
            Yields tuples with train part row numbers, test part row numbers and fold info.
        """

        df = interactions.df

        idx_col = "__IDX"
        #         idx = np.arange(len(df))
        idx = pd.RangeIndex(0, len(df))

        series_datetime = df[Columns.Datetime]
        date_range = self._get_real_date_range(series_datetime, self.date_range)

        need_dfs = self.filter_cold_users or self.filter_cold_items or self.filter_already_seen or collect_fold_stats

        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}

            train_mask = series_datetime < start
            test_mask = (series_datetime >= start) & (series_datetime < end)

            if need_dfs:
                df_train = df.loc[train_mask, Columns.UserItem]
                df_train[idx_col] = idx[train_mask]
                df_test = df.loc[test_mask, Columns.UserItem]
                df_test[idx_col] = idx[test_mask]
            else:
                train_idx = idx[train_mask]
                test_idx = idx[test_mask]

            if self.filter_cold_users:
                new_users = np.setdiff1d(df_test[Columns.User].unique(), df_train[Columns.User].unique())
                df_test = df_test.loc[~df_test[Columns.User].isin(new_users)]

            if self.filter_cold_items:
                new_items = np.setdiff1d(df_test[Columns.Item].unique(), df_train[Columns.Item].unique())
                df_test = df_test.loc[~df_test[Columns.Item].isin(new_items)]

            if self.filter_already_seen:
                not_seen_mask = get_not_seen_mask(df_train, df_test)
                df_test = df_test.loc[not_seen_mask]

            if collect_fold_stats:
                fold_info["Train"] = len(df_train)
                fold_info["Train users"] = df_train[Columns.User].nunique()
                fold_info["Train items"] = df_train[Columns.Item].nunique()
                fold_info["Test"] = len(df_test)
                fold_info["Test users"] = df_test[Columns.User].nunique()
                fold_info["Test items"] = df_test[Columns.Item].nunique()

            if need_dfs:
                yield df_train[idx_col].values, df_test[idx_col].values, fold_info
            else:
                yield train_idx.values, test_idx.values, fold_info

    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Return real number of folds."""
        date_range = self._get_real_date_range(df[Columns.Datetime], self.date_range)
        return max(0, len(date_range) - 1)

    @staticmethod
    def _get_real_date_range(series_datetime: pd.Series, date_range: DateRange) -> pd.Series:
        return date_range[(date_range >= series_datetime.min()) & (date_range <= series_datetime.max())]


def fast_sorted_unique(arr: np.ndarray):
    """
    arr: np.ndarray
    Array with integer values and shape (n, 2)

    Return
    ------
    Array of shape (m, 2) with unique rows from arr, sorted by 1 col, then 2 col
    """
    coo = sparse.csr_matrix(
        (
            np.ones(len(arr), dtype=bool),
            (
                arr[:, 0],
                arr[:, 1]
            )
        ),
    ).tocoo(copy=False)
    res = np.array([coo.row, coo.col]).T
    return res


def isin_2d_int(ar1: np.ndarray, ar2: np.ndarray, invert: bool = False):
    # inspired by np.in1d and np.unique
    #     ar1, rev_idx = np.unique(ar1, return_inverse=True, axis=0)

    ar1_dtype, ar1_shape = ar1.dtype, ar1.shape
    ar1 = np.ascontiguousarray(ar1).view(
        np.dtype((np.void, ar1.dtype.itemsize * ar1.shape[1]))
    )
    ar1, rev_idx = np.unique(ar1, return_inverse=True)
    ar1 = ar1.view(ar1_dtype).reshape(-1, ar1_shape[1])

    ar2 = fast_sorted_unique(ar2)
    ar = np.concatenate((ar1, ar2))
    del ar1, ar2

    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]
    consolidated = ar.view(dtype).flatten()

    order = consolidated.argsort(kind='mergesort')
    del consolidated
    sar = ar[order]

    if invert:
        bool_ar = (sar[1:] != sar[:-1]).any(axis=1)
    else:
        bool_ar = (sar[1:] == sar[:-1]).all(axis=1)

    del sar

    flag = np.concatenate((bool_ar, [invert]))
    del bool_ar
    ret = np.empty(flag.shape[0], dtype=bool)
    ret[order] = flag
    res = ret[rev_idx]

    return res


def get_not_seen_mask(df_train, df_test):
    n_users = max(df_train[Columns.User].max(), df_test[Columns.User].max()) + 1
    n_items = max(df_train[Columns.Item].max(), df_test[Columns.Item].max()) + 1

    cls = sparse.csr_matrix if n_users < n_items else sparse.csc_matrix
    train_csr = cls(
        (
            np.ones(len(df_train), dtype=bool),
            (
                df_train[Columns.User],
                df_train[Columns.Item],
            )
        ),
        shape=(n_users, n_items)
    )
    test_csr = cls(
        (
            np.ones(len(df_test), dtype=bool),
            (
                df_test[Columns.User],
                df_test[Columns.Item],
            )
        ),
        shape=(n_users, n_items)
    )

    already_seen_coo = test_csr.multiply(train_csr).tocoo(copy=False)
    del train_csr, test_csr
    already_seen_arr = np.array([already_seen_coo.row, already_seen_coo.col]).T
    del already_seen_coo

    # We could use it for test and train immediately, but usually already_seen_arr is much smaller than test and this way is faster
    # TODO: use CSR instead of already_seen_arr??
    not_seen_mask = isin_2d_int(df_test[Columns.UserItem].values, already_seen_arr, invert=True)
    return not_seen_mask
