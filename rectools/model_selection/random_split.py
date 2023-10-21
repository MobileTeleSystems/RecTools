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
from typing import Optional

import numpy as np
import pandas as pd

from rectools.dataset import Interactions
from rectools.model_selection.splitter import Splitter


class RandomSplitter(Splitter):
    """
    Splitter for cross-validation by random.
    Generate train and test folds with fixed test part ratio
    without intersections between test folds.
    It is also possible to exclude cold users and items
    and already seen items.

    Parameters
    ----------
    test_size : float
        Relative size of test part, must be between 0. and 1.
    n_splits : int, default 1
        Number of folds.
    random_state : int, default  None,
        Controls randomness of each fold. Pass an int to get reproducible result across multiple `split` calls.
    filter_cold_users : bool, default ``True``
        If `True`, users that not in train will be excluded from test.
    filter_cold_items : bool, default ``True``
        If `True`, items that not in train will be excluded from test.
    filter_already_seen : bool, default ``True``
        If `True`, pairs (user, item) that are in train will be excluded from test.

    Examples
    --------
    >>> from rectools import Columns
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
    >>> kfs = KFoldSplitter(test_size=0.25, random_state=42, n_splits=2, filter_cold_users=False,
    ...                     filter_cold_items=False, filter_already_seen=False)
    >>> for train_ids, test_ids, _ in kfs.split(interactions):
    ...     print(train_ids, test_ids)
    [2 7 6 1 5 0] [3 4]
    [3 4 6 1 5 0] [2 7]
    >>>
    >>> kfs = KFoldSplitter(test_size=0.25, random_state=42, n_splits=2, filter_cold_users=True,
    ...                     filter_cold_items=True, filter_already_seen=True)
    >>> for train_ids, test_ids, _ in kfs.split(interactions):
    ...     print(train_ids, test_ids)
    [2 7 6 1 5 0] [3 4]
    [3 4 6 1 5 0] [2]
    """

    def __init__(
        self,
        test_size: float,
        n_splits: int = 1,
        random_state: Optional[int] = None,
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        if test_size <= 0.0 or test_size >= 1.0:
            raise ValueError("Value of test_size must be between 0 and 1")
        
        if test_size * n_splits > 1:
            raise ValueError(f"Impossible to create {n_splits} non-overlapping folds {test_size:.1%} each")

        super().__init__(filter_cold_users, filter_cold_items, filter_already_seen)
        self.random_state = random_state
        self.n_splits = n_splits
        self.test_size = test_size

    def _split_without_filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        rng = np.random.default_rng(self.random_state)
        df = interactions.df
        idx = pd.RangeIndex(0, len(df))

        test_part_size = int(round(self.test_size * len(df)))
        if test_part_size == 0:
            err_message = "Length of interactions ({} elements) with test_size={} leads to empty test part"
            raise ValueError(err_message.format(len(df), self.test_size))
        if test_part_size == len(df):
            err_message = (
                "Length of interactions ({} elements) with test_size={} leads to empty train part: "
                "all interactions are split to the test"
            )
            raise ValueError(err_message.format(len(df), self.test_size))

        if self.n_splits * test_part_size > len(df):
            err_message = f"Impossible to create {self.n_splits} non-overlapping folds with size {test_part_size} from {len(df)} interactions"
            raise ValueError(
                f"Impossible to create {self.n_splits} non-overlapping folds "
                f"with size {test_part_size} from {len(df)} interactions"
            )

        shuffled_idx = rng.permutation(idx)
        for i in range(self.n_splits):
            fold_info = {"fold_number": i}
            left = i * test_part_size
            right = (i + 1) * test_part_size
            test_idx = shuffled_idx[left:right]
            train_idx = np.concatenate((shuffled_idx[:left], shuffled_idx[right:]))

            yield train_idx, test_idx, fold_info
