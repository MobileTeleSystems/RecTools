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

from rectools import Columns
from rectools.utils import pairwise

DateRange = tp.Sequence[tp.Union[date, datetime]]


class TimeRangeSplit:
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
    ...         [
    ...             [1, 2, "2021-09-01"],  # 0
    ...             [2, 1, "2021-09-02"],  # 1
    ...             [2, 3, "2021-09-03"],  # 2
    ...             [3, 2, "2021-09-03"],  # 3
    ...             [3, 3, "2021-09-04"],  # 4
    ...             [3, 4, "2021-09-04"],  # 5
    ...             [1, 2, "2021-09-05"],  # 6
    ...             [4, 2, "2021-09-05"],  # 7
    ...             [4, 2, "2021-09-06"],  # 8
    ...         ],
    ...         columns=[Columns.User, Columns.Item, Columns.Datetime],
    ...     ).astype({Columns.Datetime: "datetime64[ns]"})
    >>> date_range = pd.date_range(date(2021, 9, 4), date(2021, 9, 6))
    >>>
    >>> trs = TimeRangeSplit(date_range, False, False, False)
    >>> for train_ids, test_ids, _ in trs.split(df):
    ...     print(train_ids, test_ids)
    [0 1 2 3] [4 5]
    [0 1 2 3 4 5] [6 7]
    >>>
    >>> trs = TimeRangeSplit(date_range, True, True, True)
    >>> for train_ids, test_ids, _ in trs.split(df):
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
        df: pd.DataFrame,
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
        required_columns = {Columns.User, Columns.Item, Columns.Datetime}
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(f"Missed columns {required_columns - actual_columns}")

        series_datetime = df[Columns.Datetime]
        train_datetime_mask = series_datetime.notnull()
        date_range = self._get_real_date_range(df[Columns.Datetime], self.date_range)

        df = df.loc[:]
        idx_col = "__IDX"
        df[idx_col] = np.arange(len(df))

        for start, end in pairwise(date_range):
            fold_info = {"Start date": start, "End date": end}
            train_mask = train_datetime_mask & (series_datetime < start)
            df_train = df.loc[train_mask]

            if collect_fold_stats:
                fold_info["Train"] = len(df_train)
                fold_info["Train users"] = df_train[Columns.User].nunique()
                fold_info["Train items"] = df_train[Columns.Item].nunique()

            test_mask = (series_datetime >= start) & (series_datetime < end)
            df_test = df.loc[test_mask]

            if self.filter_cold_users:
                new_users = np.setdiff1d(df_test[Columns.User].unique(), df_train[Columns.User].unique())
                df_test = df_test.loc[~df_test[Columns.User].isin(new_users)]

            if self.filter_cold_items:
                new_items = np.setdiff1d(df_test[Columns.Item].unique(), df_train[Columns.Item].unique())
                df_test = df_test.loc[~df_test[Columns.Item].isin(new_items)]

            if self.filter_already_seen:
                df_test.index.rename("_index", inplace=True)
                df_test = (
                    df_test.reset_index()
                    .merge(df_train[Columns.UserItem], on=Columns.UserItem, how="left", indicator=True)
                    .query("_merge == 'left_only'")
                    .drop(columns="_merge")
                    .set_index("_index")
                )

            if collect_fold_stats:
                fold_info["Test"] = len(df_test)
                fold_info["Test users"] = df_test[Columns.User].nunique()
                fold_info["Test items"] = df_test[Columns.Item].nunique()

            yield df_train[idx_col].values, df_test[idx_col].values, fold_info

    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Return real number of folds."""
        date_range = self._get_real_date_range(df[Columns.Datetime], self.date_range)
        return max(0, len(date_range) - 1)

    @staticmethod
    def _get_real_date_range(series_datetime: pd.Series, date_range: DateRange) -> pd.Series:
        return date_range[(date_range >= series_datetime.min()) & (date_range <= series_datetime.max())]
