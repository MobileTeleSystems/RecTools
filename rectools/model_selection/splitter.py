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

"""Splitter."""

import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection.utils import get_not_seen_mask


class Splitter:
    """
    Base class to construct data splitters. It cannot be used directly.
    New splitter can be defined by subclassing the `Splitter` class
    and implementing `_split_without_filter` method.
    Check specific class descriptions to get more information.
    """

    def __init__(
        self, filter_cold_users: bool = True, filter_cold_items: bool = True, filter_already_seen: bool = True
    ) -> None:
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

    def split(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        """
        Split interactions into folds and apply filtration to the result.

        Parameters
        ----------
        interactions : Interactions
            User-item interactions.
        collect_fold_stats : bool, default False
            Add some stats to split info,
            like size of train and test part, number of users and items.

        Returns
        -------
        iterator(array, array, dict)
            Yields tuples with train part row numbers, test part row numbers and split info.
        """
        for train_idx, test_idx, split_info in self._split_without_filter(interactions, collect_fold_stats):
            yield self.filter(interactions, collect_fold_stats, train_idx, test_idx, split_info)

    def _split_without_filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        """
        Split interactions into folds.

        Parameters
        ----------
        interactions : Interactions
            User-item interactions.
        collect_fold_stats : bool, default False
            Add some stats to split info,
            like size of train and test part, number of users and items.

        Returns
        -------
        iterator(array, array, dict)
            Yields tuples with train part row numbers, test part row numbers and split info.
        """
        raise NotImplementedError

    def filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        split_info: tp.Dict[str, tp.Any],
    ) -> tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]:
        """
        Filter train and test indexes from one fold based on `filter_cold_users`,
        `filter_cold_items`,`filter_already_seen` class fields.
        They are set to `True` by default.

        Parameters
        ----------
        interactions : Interactions
            User-item interactions.
        collect_fold_stats : bool, default False
            Add some stats to split info,
            like size of train and test part, number of users and items.
        train_idx : array
            Train part row numbers.
        test_idx : array
            Test part row numbers.
        split_info : dict
            Information about the split.

        Returns
        -------
        Tuple(array, array, dict)
            Returns tuple with filtered train part row numbers, test part row numbers and split info.
        """
        need_ui = self.filter_cold_users or self.filter_cold_items or self.filter_already_seen or collect_fold_stats

        if need_ui:
            df = interactions.df
            train_users = df[Columns.User].values[train_idx]
            train_items = df[Columns.Item].values[train_idx]
            test_users = df[Columns.User].values[test_idx]
            test_items = df[Columns.Item].values[test_idx]

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

            split_info["train"] = train_users.size
            split_info["train_users"] = unq_train_users.size
            split_info["train_items"] = unq_train_items.size
            split_info["test"] = test_users.size
            split_info["test_users"] = pd.unique(test_users).size
            split_info["test_items"] = pd.unique(test_items).size

        return train_idx, test_idx, split_info
