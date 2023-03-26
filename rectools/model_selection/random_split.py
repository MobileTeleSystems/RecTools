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
    train_size : float, default 0.75
        Relative size of train part, must be between 0. and 1.
    filter_cold_users: bool, default ``True``
        If `True`, users that not in train will be excluded from test.
    filter_cold_items: bool, default ``True``
        If `True`, items that not in train will be excluded from test.
    filter_already_seen: bool, default ``True``
        If ``True``, pairs (user, item) that are in train will be excluded from test.


    """

    def __init__(
        self,
        train_size: float,
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        if train_size < 0.0 or train_size > 1.0:
            raise ValueError("value of train_size must be between 0 and 1")

        self.train_size = train_size
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
        test_part_size = len(df) - int(self.train_size * len(df))

        need_ui = self.filter_cold_users or self.filter_cold_items or self.filter_already_seen or collect_fold_stats
        fold_info = {}

        test_mask = np.zeros_like(idx, dtype=bool)
        choose_idx = np.random.choice(idx, test_part_size, replace=False)
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
