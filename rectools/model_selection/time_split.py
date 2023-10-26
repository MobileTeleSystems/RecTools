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

import re
import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.dataset import Interactions
from rectools.model_selection.splitter import Splitter
from rectools.utils import pairwise


class TimeRangeSplitter(Splitter):
    r"""
    Splitter for cross-validation by leave-time-out scheme.
    Generate train and test putting all interactions for all users
    after fixed date-time in test and all interactions before this date-time in train.
    Cross-validation is achieved with sliding window over timeline of interactions.

    Size of the window is defined in days or hours.
    Test folds do not intersect and start one right after the other.
    This technique fully reproduces the real life scenario for recommender systems,
    preventing any data leak from the future.

    It is advised to remember daily and weekly patterns in time series,
    making each fold equal to full day or full week
    when such patterns are present in data.

    It is also possible to exclude cold users and items and already seen items.

    Parameters
    ----------
    test_size : str
        Size of test fold in format ``[1-9]\d*[DH]``, e.g. ``1D`` (1 day), ``4H`` (4 hours).
        Test folds are taken from the end of `interactions`.
        The last fold includes the whole time unit with the last interaction.
        E.g. if the last interaction was at 01:25 a.m. of Monday, then
        with `test_size = "1D"` the last fold will be the full Monday,
        and with `test_size = "1H"` the last fold will be between 01:00 a.m. and 02:00 a.m on Monday.
    n_splits : int
        Number of test folds.
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
    ...         [4, 4, 1, "2021-09-04"],  # 5
    ...         [1, 2, 1, "2021-09-05"],  # 6
    ...     ],
    ...     columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
    ... ).astype({Columns.Datetime: "datetime64[ns]"})
    >>> interactions = Interactions(df)
    >>>
    >>> splitter = TimeRangeSplitter("1D", 2, False, False, False)
    >>> for train_ids, test_ids, _ in splitter.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3] [4 5]
    [0 1 2 3 4 5] [6]
    >>>
    >>> splitter = TimeRangeSplitter("1D", 2, True, False, False)
    >>> for train_ids, test_ids, _ in splitter.split(interactions):
    ...     print(train_ids, test_ids)
    [0 1 2 3] [4]
    [0 1 2 3 4 5] [6]
    """

    def __init__(
        self,
        test_size: str,
        n_splits: int = 1,
        filter_cold_users: bool = True,
        filter_cold_items: bool = True,
        filter_already_seen: bool = True,
    ) -> None:
        super().__init__(filter_cold_users, filter_cold_items, filter_already_seen)
        m = re.fullmatch(r"([1-9]\d*)([DH])", test_size)
        if not m:
            raise ValueError(r"Test size must match to `[1-9]\d*[DH]`, e.g. 1D, 4H")
        self.test_size = test_size
        self.test_size_value = int(m.groups()[0])
        self.test_size_unit = m.groups()[1]
        self.n_splits = n_splits

    def get_test_fold_borders(self, interactions: Interactions) -> tp.List[tp.Tuple[pd.Timestamp, pd.Timestamp]]:
        """Return datetime borders of test folds based on given test fold sizes and last interaction."""
        last_dt = interactions.df[Columns.Datetime].max()
        last_dt_ceiled = last_dt.ceil(self.test_size_unit)

        if last_dt_ceiled == last_dt:  # dt is exactly on units border, like `2021-09-06 00:00:00` with unit = "D"
            last_dt_ceiled += pd.Timedelta(1, unit=self.test_size_unit)

        start_dt = last_dt_ceiled - pd.Timedelta(self.n_splits * self.test_size_value, unit=self.test_size_unit)
        date_range = pd.date_range(start=start_dt, periods=self.n_splits + 1, freq=self.test_size, tz=last_dt.tz)
        borders = list(pairwise(date_range))
        return borders

    def _split_without_filter(
        self,
        interactions: Interactions,
        collect_fold_stats: bool = False,
    ) -> tp.Iterator[tp.Tuple[np.ndarray, np.ndarray, tp.Dict[str, tp.Any]]]:
        idx = pd.RangeIndex(0, len(interactions.df))

        test_fold_borders = self.get_test_fold_borders(interactions)

        series_datetime = interactions.df[Columns.Datetime]

        for i_split, (start, end) in enumerate(test_fold_borders):
            train_mask = series_datetime < start
            test_mask = (series_datetime >= start) & (series_datetime < end)

            train_idx = idx[train_mask].values
            test_idx = idx[test_mask].values

            fold_info = {"i_split": i_split, "start": start, "end": end}

            yield train_idx, test_idx, fold_info
