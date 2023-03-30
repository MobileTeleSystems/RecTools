#  Copyright 2023 MTS (Mobile Telesystems)
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

"""RandomSplitter."""

import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection.utils import get_not_seen_mask


class RandomSplitter:
    """
    Splitter for cross-validation by random.
    Generate train and test folds with fixed part ratio,
    it is also possible to exclude cold users and items
    and already seen items.

    Parameters
    ----------
    test_size : float
        Relative size of test part, must be between 0. and 1.
    n_splits : int, default 1
        Number of folds.
    random_state: int, default  None,
        Controls randomness of each fold. Pass an int to get reproducible result across multiple class instances.
    filter_cold_users: bool, default ``True``
        If `True`, users that not in train will be excluded from test.
    filter_cold_items: bool, default ``True``
        If `True`, items that not in train will be excluded from test.
    filter_already_seen: bool, default ``True``
        If `True`, pairs (user, item) that are in train will be excluded from test.

     Examples
    --------
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
    ...     ],
    ...     columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    ... ).astype({Columns.Datetime: "datetime64[ns]"})
    >>> interactions = Interactions(df)
    >>>
    >>> rs = RandomSplitter(test_size=0.25, random_state=42, n_splits=2, filter_cold_users=False,
    ...                     filter_cold_items=False, filter_already_seen=False)
    >>> for train_ids, test_ids, _ in rs.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3 5 6] [4 7]
    [0 1 2 3 4 7] [5 6]
    >>>
    >>> rs = RandomSplitter(test_size=0.25, random_state=42, n_splits=2, filter_cold_users=True,
    ...                     filter_cold_items=True, filter_already_seen=True)
    >>> for train_ids, test_ids, _ in rs.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3 5 6] [4]
    [0 1 2 3 4 7] []
    """

    def __init__(
        self,
        test_size: float,
        n_splits: int = 1,
        random_state: int = None,
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        if test_size <= 0.0 or test_size >= 1.0:
            raise ValueError("Value of test_size must be between 0 and 1")

        self.random_state = random_state
        self.n_splits = n_splits
        self.test_size = test_size
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
        test_part_size = int(self.test_size * len(df))

        need_ui = self.filter_cold_users or self.filter_cold_items or self.filter_already_seen or collect_fold_stats

        for i in range(self.n_splits):
            fold_info = {}

            test_mask = np.zeros_like(idx, dtype=bool)
            choose_idx = np.random.RandomState(self.random_state).choice(idx, test_part_size, replace=False)
            test_mask[choose_idx] = True
            train_mask = ~test_mask

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
