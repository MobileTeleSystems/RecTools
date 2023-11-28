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

"""LastNSplitter."""

import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection.splitter import Splitter


class LastNSplitter(Splitter):
    """
    Splitter for cross-validation by leave-one-out / leave-k-out scheme (recent activity).
    Generate train and test putting last n interactions for each user in test
    and all of his previous interactions in train.
    Cross-validation is achieved with sliding window over each users interactions history.

    This technique may be used for sequential recommendation scenarios.
    It is common in research papers on sequential recommendations.
    But it doesn't fully prevent data leak from the future.

    It is also possible to exclude cold users and items and already seen items.

    Parameters
    ----------
    n : int
        Number of interactions for each user that will be included in test.
    n_splits : int, default 1
        Number of test folds.
    filter_cold_users : bool, default ``True``
        If `True`, users that not in train will be excluded from test.
    filter_cold_items : bool, default ``True``
        If `True`, items that not in train will be excluded from test.
    filter_already_seen : bool, default ``True``
        If ``True``, pairs (user, item) that are in train will be excluded from test.

    Examples
    --------
    >>> from rectools import Columns
    >>> df = pd.DataFrame(
    ...     [
    ...         [1, 1, 1, "2021-09-01"], # 0
    ...         [1, 2, 1, "2021-09-02"], # 1
    ...         [1, 1, 1, "2021-09-03"], # 2
    ...         [1, 2, 1, "2021-09-04"], # 3
    ...         [1, 2, 1, "2021-09-05"], # 4
    ...         [2, 1, 1, "2021-08-20"], # 5
    ...         [2, 2, 1, "2021-08-21"], # 6
    ...         [2, 2, 1, "2021-08-22"], # 7
    ...     ],
    ...     columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    ... ).astype({Columns.Datetime: "datetime64[ns]"})
    >>> interactions = Interactions(df)
    >>>
    >>> splitter = LastNSplitter(2, 2, False, False, False)
    >>> for train_ids, test_ids, _ in splitter.split(interactions):
    ...     print(train_ids, test_ids)
    [0] [1 2 5]
    [0 1 2 5] [3 4 6 7]
    >>>
    >>> splitter = LastNSplitter(2, 2, True, False, False)
    >>> for train_ids, test_ids, _ in splitter.split(interactions):
    ...     print(train_ids, test_ids)
    [0] [1 2]
    [0 1 2 5] [3 4 6 7]
    """

    def __init__(
        self,
        n: int,
        n_splits: int = 1,
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        super().__init__(filter_cold_users, filter_cold_items, filter_already_seen)
        self.n = n
        self.n_splits = n_splits

    def _split_without_filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        df = interactions.df
        idx = pd.RangeIndex(0, len(df))

        # last event - rank=1
        inv_ranks = df.groupby(Columns.User)[Columns.Datetime].rank(method="first", ascending=False)

        for i_split in range(self.n_splits)[::-1]:
            min_rank = i_split * self.n  # excluded
            max_rank = min_rank + self.n  # included

            test_mask = (inv_ranks > min_rank) & (inv_ranks <= max_rank)
            train_mask = inv_ranks > max_rank

            test_idx = idx[test_mask].values
            train_idx = idx[train_mask].values

            split_info = {"i_split": self.n_splits - i_split - 1}

            yield train_idx, test_idx, split_info
