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

"""TimeRangeSplitter."""

import typing as tp
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy import sparse

from rectools import Columns
from rectools.dataset import Interactions
from rectools.utils import isin_2d_int, pairwise

DateRange = tp.Sequence[tp.Union[date, datetime]]


class TimeRangeSplitter:
    """
    Splitter for cross-validation by time.
    Generate train and test folds by time,
    it is also possible to exclude cold users and items
    and already seen items.

    Parameters
    ----------
    date_range: array-like(date|datetime)
        Ordered test fold borders.
        Left will be included, right will be excluded from fold.
        Interactions before first border will be used for train.
        Interaction after right border will not be used.
        Ca be easily generated with [`pd.date_range`]
        (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)
    filter_cold_users: bool, default ``True``
        If `True`, users that not in train will be excluded from test.
    filter_cold_items: bool, default ``True``
        If `True`, items that not in train will be excluded from test.
    filter_already_seen: bool, default ``True``
        If ``True``, pairs (user, item) that are in train will be excluded from test.

    Examples
    --------
    >>> from datetime import date
    >>> df = pd.DataFrame(
    ...     [
    ...         [1, 2, 1, "2021-09-01"],  # 0
    ...         [2, 1, 1, "2021-09-02"],  # 1
    ...         [2, 3, 1, "2021-09-03"],  # 2
    ...         [3, 2, 1, "2021-09-03"],  # 3
    ...         [3, 3, 1, "2021-09-04"],  # 4
    ...         [3, 4, 1, "2021-09-04"],  # 5
    ...         [1, 2, 1, "2021-09-05"],  # 6
    ...         [4, 2, 1, "2021-09-05"],  # 7
    ...         [4, 2, 1, "2021-09-06"],  # 8
    ...     ],
    ...     columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    ... ).astype({Columns.Datetime: "datetime64[ns]"})
    >>> interactions = Interactions(df)
    >>> date_range = pd.date_range(date(2021, 9, 4), date(2021, 9, 6))
    >>>
    >>> trs = TimeRangeSplitter(date_range, False, False, False)
    >>> for train_ids, test_ids, _ in trs.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3] [4 5]
    [0 1 2 3 4 5] [6 7]
    >>>
    >>> trs = TimeRangeSplitter(date_range, True, True, True)
    >>> for train_ids, test_ids, _ in trs.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3] [4]
    [0 1 2 3 4 5] []
    """

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
        interactions: Interactions
            User-item interactions.
        collect_fold_stats: bool, default False
            Add some stats to fold info,
            like size of train and test part, number of users and items.

        Returns
        -------
        iterator(array, array, dict)
            Yields tuples with train part row numbers, test part row numbers and fold info.
        """
        df = interactions.df
        idx = pd.RangeIndex(0, len(df))

        series_datetime = df[Columns.Datetime]
        date_range = self._get_real_date_range(series_datetime, self.date_range)

        need_ui = self.filter_cold_users or self.filter_cold_items or self.filter_already_seen or collect_fold_stats

        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}

            train_mask = series_datetime < start
            test_mask = (series_datetime >= start) & (series_datetime < end)

            train_idx = idx[train_mask].values
            test_idx = idx[test_mask].values

            if need_ui:
                train_users = df[Columns.User].values[train_mask]
                train_items = df[Columns.Item].values[train_mask]
                test_users = df[Columns.User].values[test_mask]
                test_items = df[Columns.Item].values[test_mask]

            unq_train_users = None
            unq_train_items = None

            if self.filter_cold_users:
                unq_train_users = pd.unique(train_users)
                mask = np.isin(test_users, unq_train_users)
                test_users = test_users[mask]
                test_items = test_items[mask]
                test_idx = test_idx[mask]

            if self.filter_cold_items:
                unq_train_items = pd.unique(train_items)
                mask = np.isin(test_items, unq_train_items)
                test_users = test_users[mask]
                test_items = test_items[mask]
                test_idx = test_idx[mask]

            if self.filter_already_seen:
                mask = get_not_seen_mask(train_users, train_items, test_users, test_items)
                test_users = test_users[mask]
                test_items = test_items[mask]
                test_idx = test_idx[mask]

            if collect_fold_stats:
                if unq_train_users is None:
                    unq_train_users = pd.unique(train_users)
                if unq_train_items is None:
                    unq_train_items = pd.unique(train_items)

                fold_info["Train"] = train_users.size
                fold_info["Train users"] = unq_train_users.size
                fold_info["Train items"] = unq_train_items.size
                fold_info["Test"] = test_users.size
                fold_info["Test users"] = pd.unique(test_users).size
                fold_info["Test items"] = pd.unique(test_items).size

            yield train_idx, test_idx, fold_info

    def get_n_splits(self, interactions: Interactions) -> int:
        """Return real number of folds."""
        date_range = self._get_real_date_range(interactions.df[Columns.Datetime], self.date_range)
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
