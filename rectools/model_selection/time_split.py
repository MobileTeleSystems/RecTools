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

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection.splitter import Splitter
from rectools.utils import pairwise

DateRange = tp.Sequence[tp.Union[date, datetime]]


class TimeRangeSplitter(Splitter):
    """
    Splitter for cross-validation by time.
    Generate train and test folds by time,
    it is also possible to exclude cold users and items
    and already seen items.

    Parameters
    ----------
    date_range : array-like(date|datetime)
        Ordered test fold borders.
        Left will be included, right will be excluded from fold.
        Interactions before first border will be used for train.
        Interaction after right border will not be used.
        Ca be easily generated with [`pd.date_range`]
        (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)
    filter_cold_users : bool, default ``True``
        If `True`, users that not in train will be excluded from test.
    filter_cold_items : bool, default ``True``
        If `True`, items that not in train will be excluded from test.
    filter_already_seen : bool, default ``True``
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
    >>> splitter = TimeRangeSplitter(date_range, False, False, False)
    >>> for train_ids, test_ids, _ in splitter.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3] [4 5]
    [0 1 2 3 4 5] [6 7]
    >>>
    >>> splitter = TimeRangeSplitter(date_range, True, True, True)
    >>> for train_ids, test_ids, _ in splitter.split(interactions):
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
        super().__init__(filter_cold_users, filter_cold_items, filter_already_seen)
        self.date_range = date_range

    def _split_without_filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        df = interactions.df
        idx = pd.RangeIndex(0, len(df))

        series_datetime = df[Columns.Datetime]
        date_range = self._get_real_date_range(series_datetime, self.date_range)

        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}

            train_mask = series_datetime < start
            test_mask = (series_datetime >= start) & (series_datetime < end)

            train_idx = idx[train_mask].values
            test_idx = idx[test_mask].values

            yield train_idx, test_idx, fold_info

    def get_n_splits(self, interactions: Interactions) -> int:
        """Return real number of folds."""
        date_range = self._get_real_date_range(interactions.df[Columns.Datetime], self.date_range)
        return max(0, len(date_range) - 1)

    @staticmethod
    def _get_real_date_range(series_datetime: pd.Series, date_range: DateRange) -> pd.Series:
        return date_range[(date_range >= series_datetime.min()) & (date_range <= series_datetime.max())]
