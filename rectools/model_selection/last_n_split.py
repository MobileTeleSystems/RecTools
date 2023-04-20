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

from rectools.dataset import Interactions
from rectools.model_selection.splitter import Splitter


class LastNSplitter(Splitter):
    """
    Splitter for cross-validation by recent activity.
    Generate train and test putting last n interaction for
    each user in test and others in train.
    It is also possible to exclude cold users and items
    and already seen items.

    Parameters
    ----------
    n : int or iterable of ints
        Number of interactions for each user that will be included in test.
        If multiple arguments are passed, separate fold will be created for each of them.
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
    ...         [1, 1, 1, "2021-08-20"], # 2
    ...         [1, 2, 1, "2021-09-04"], # 3
    ...         [2, 1, 1, "2021-08-20"], # 4
    ...         [2, 2, 1, "2021-08-20"], # 5
    ...         [2, 3, 1, "2021-09-05"], # 6
    ...         [2, 2, 1, "2021-09-06"], # 7
    ...         [3, 1, 1, "2021-09-05"], # 8
    ...     ],
    ...     columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    ... ).astype({Columns.Datetime: "datetime64[ns]"})
    >>> interactions = Interactions(df)
    >>>
    >>> lns = LastNSplitter(2, False, False, False)
    >>> for train_ids, test_ids, _ in lns.split(interactions):
    ...     print(train_ids, test_ids)
    [0 2 4 5] [1 3 6 7 8]
    >>>
    >>> lns = LastNSplitter(2, True, True, True)
    >>> for train_ids, test_ids, _ in lns.split(interactions):
    ...     print(train_ids, test_ids)
    [0 2 4 5] [1 3]
    >>>
    >>> lns = LastNSplitter([1, 2], False, False, False)
    >>> for train_ids, test_ids, _ in lns.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 4 5 6] [3 7 8]
    [0 2 4 5] [1 3 6 7 8]
    """

    def __init__(
        self,
        n: tp.Union[int, tp.Iterable[int]],
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        super().__init__(filter_cold_users, filter_cold_items, filter_already_seen)
        if isinstance(n, int):
            self.n = [n]
        else:
            self.n = list(n)

    def _split_without_filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        df = interactions.df
        idx = pd.RangeIndex(0, len(df))
        index_df = pd.Series(idx, index=df.index)

        for n in self.n:
            if n <= 0:
                raise ValueError(f"N must be positive, got {n}")

            last_n_interactions = df.groupby("user_id")["datetime"].nlargest(n)
            test_idx_remapped = last_n_interactions.index.levels[1].to_numpy()
            train_mask = np.ones_like(idx, dtype=bool)
            train_mask[test_idx_remapped] = False
            train_idx_remapped = idx[train_mask]
            train_idx = index_df.loc[train_idx_remapped].values
            test_idx = index_df.loc[test_idx_remapped].values

            fold_info = {}
            if collect_fold_stats:
                fold_info["n"] = n

            yield train_idx, test_idx, fold_info
