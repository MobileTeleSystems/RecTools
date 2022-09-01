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
from rectools.utils import pairwise, isin_2d_int

DateRange = tp.Sequence[tp.Union[date, datetime]]


class TimeRangeSplitter:
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


def get_not_seen_mask(
    train_users: np.ndarray,
    train_items: np.ndarray,
    test_users: np.ndarray,
    test_items: np.ndarray,
) -> np.ndarray:
    """
    Return mask for test interactions that is not in train interactions.

    Parameters
    ----------
    train_users : np.ndarray
        Integer array of users in train interactions (it's not a unique users!).
    train_items : np.ndarray
        Integer array of items in train interactions. Has same length as `train_users`.
    test_users : np.ndarray
        Integer array of users in test interactions (it's not a unique users!).
    test_items : np.ndarray
        Integer array of items in test interactions. Has same length as `test_users`.

    Returns
    -------
    np.ndarray
     Boolean mask of same length as `test_users` (`test_items`).
     ``True`` means interaction not present in train.
    """
    if train_users.size != train_items.size:
        raise ValueError("Lengths of `train_users` and `train_items` must be the same")
    if test_users.size != test_items.size:
        raise ValueError("Lengths of `test_users` and `test_items` must be the same")

    if train_users.size == 0:
        return np.ones(test_users.size, dtype=bool)
    if test_users.size == 0:
        return np.array([], dtype=bool)

    n_users = max(train_users.max(), test_users.max()) + 1
    n_items = max(train_items.max(), test_items.max()) + 1

    cls = sparse.csr_matrix if n_users < n_items else sparse.csc_matrix

    def make_matrix(users: np.ndarray, items: np.ndarray) -> sparse.spmatrix:
        return cls((np.ones(len(users), dtype=bool), (users, items)), shape=(n_users, n_items))

    train_mat = make_matrix(train_users, train_items)
    test_mat = make_matrix(test_users, test_items)

    already_seen_coo = test_mat.multiply(train_mat).tocoo(copy=False)
    del train_mat, test_mat
    already_seen_arr = np.vstack((already_seen_coo.row, already_seen_coo.col)).T.astype(test_users.dtype)
    del already_seen_coo

    test_ui = np.vstack((test_users, test_items)).T
    not_seen_mask = isin_2d_int(test_ui, already_seen_arr, invert=True)
    return not_seen_mask
