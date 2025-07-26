#  Copyright 2022-2025 MTS (Mobile Telesystems)
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

"""Structure for saving user-item interactions."""

import typing as tp

import attr
import numpy as np
import pandas as pd
from scipy import sparse

from rectools import Columns

from .identifiers import IdMap


@attr.s(frozen=True, slots=True)
class Interactions:
    """
    Structure to store info about user-item interactions.

    Usually it's more convenient to use `from_raw` method instead of direct creating.

    Parameters
    ----------
        df : pd.DataFrame
            Table where every row contains user-item interaction and columns are:
                - `Columns.User` - internal user id (non-negative int values);
                - `Columns.Item` - internal item id (non-negative int values);
                - `Columns.Weight` - weight of interaction, float, use ``1`` if interactions have no weight;
                - `Columns.Datetime` - timestamp of interactions,
                  assign random value if you're not going to use it later.
            Extra columns can also be present.
    """

    df: pd.DataFrame = attr.ib()

    @staticmethod
    def _check_columns_present(df: pd.DataFrame) -> None:
        required_columns = {Columns.User, Columns.Item, Columns.Weight, Columns.Datetime}
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(f"Missed columns {required_columns - actual_columns}")

    @staticmethod
    def convert_weight_and_datetime_types(df: pd.DataFrame) -> None:
        """
        Convert weight column to float and datetime column to datetime64[ns] in-place.

        This method ensures that the specified weight column contains numeric values
        and that the datetime column can be converted to pandas' datetime64[ns] format.
        The conversion is done in-place, so the original DataFrame will be modified.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame that must contain the following columns:
                - `Columns.Weight` - interaction weight;
                - `Columns.Datetime` - interaction timestamp.
        """
        try:
            df[Columns.Weight] = df[Columns.Weight].astype(float)
        except ValueError:
            raise TypeError(f"Column '{Columns.Weight}' must be numeric")

        try:
            df[Columns.Datetime] = df[Columns.Datetime].astype("datetime64[ns]")
        except ValueError:
            raise TypeError(f"Column '{Columns.Datetime}' must be convertible to 'datetime64' type")

    @df.validator
    def _check_columns_present_validator(self, _: str, df: pd.DataFrame) -> None:
        self._check_columns_present(df)

    @df.validator
    def _check_ids(self, _: str, df: pd.DataFrame) -> None:
        for col in (Columns.User, Columns.Item):
            if not df[col].dtype.name.startswith(("int", "uint")):
                raise TypeError(f"Column '{col}' must be integer")
            if df[col].min() < 0:
                raise ValueError(f"Column '{col}' values must be >= 0")

    def __attrs_post_init__(self) -> None:
        """Convert datetime and weight columns to the right data types."""
        self.convert_weight_and_datetime_types(self.df)

    @staticmethod
    def _add_extra_cols(df: pd.DataFrame, interactions: pd.DataFrame) -> None:
        extra_cols = [col for col in interactions.columns if col not in df.columns]
        for extra_col in extra_cols:
            df[extra_col] = interactions[extra_col].values

    @classmethod
    def from_raw(
        cls, interactions: pd.DataFrame, user_id_map: IdMap, item_id_map: IdMap, keep_extra_cols: bool = False
    ) -> "Interactions":
        """
        Create `Interactions` from dataset with external ids and id mappings.

        Parameters
        ----------
        interactions : pd.DataFrame
            Table where every row contains user-item interaction and columns are:
                - `Columns.User` - user id;
                - `Columns.Item` - item id;
                - `Columns.Weight` - weight of interaction, float, use ``1`` if interactions have no weight;
                - `Columns.Datetime` - timestamp of interactions,
                  assign random value if you're not going to use it later.
        user_id_map : IdMap
            User identifiers mapping.
        item_id_map : IdMap
            Item identifiers mapping.
        keep_extra_cols: bool, default ``False``
            Flag to keep all columns from interactions besides the default ones.

        Returns
        -------
            Interactions
        """
        cls._check_columns_present(interactions)

        df = pd.DataFrame(
            {
                Columns.User: user_id_map.convert_to_internal(interactions[Columns.User]),
                Columns.Item: item_id_map.convert_to_internal(interactions[Columns.Item]),
            },
        )
        df[Columns.Weight] = interactions[Columns.Weight].values
        df[Columns.Datetime] = interactions[Columns.Datetime].values
        cls.convert_weight_and_datetime_types(df)
        if keep_extra_cols:
            cls._add_extra_cols(df, interactions)

        return cls(df)

    def get_user_item_matrix(self, include_weights: bool = True, dtype: tp.Type = np.float32) -> sparse.csr_matrix:
        """
        Form a user-item CSR matrix based on interactions data.

        Parameters
        ----------
        include_weights : bool, default ``True``
             Whether include interaction weights in matrix or not.
             If ``False``, all values in returned matrix will be equal to ``1``.

        Returns
        -------
        csr_matrix
        """
        if include_weights:
            values = self.df[Columns.Weight].values
        else:
            values = np.ones(len(self.df))

        csr = sparse.csr_matrix(
            (
                values.astype(dtype),
                (
                    self.df[Columns.User].values,
                    self.df[Columns.Item].values,
                ),
            ),
        )
        return csr

    def to_external(
        self,
        user_id_map: IdMap,
        item_id_map: IdMap,
        include_weight: bool = True,
        include_datetime: bool = True,
        include_extra_cols: tp.Union[bool, tp.List[str]] = True,
    ) -> pd.DataFrame:
        """
        Convert itself to `pd.DataFrame` with replacing internal user and item ids to external ones.

        Parameters
        ----------
        user_id_map : IdMap
            User id map that has to be used for converting internal user ids to external ones.
        item_id_map : IdMap
            Item id map that has to be used for converting internal item ids to external ones.
        include_weight : bool, default ``True``
            Whether to include weight column into resulting table or not
        include_datetime : bool, default ``True``
            Whether to include datetime column into resulting table or not.
        include_extra_cols: bool or List[str], default ``True``
            If bool, indicates whether to include all extra columns into resulting table or not.
            If list of strings, indicates which extra columns to include into resulting table.

        Returns
        -------
        pd.DataFrame
        """
        res = pd.DataFrame(
            {
                Columns.User: user_id_map.convert_to_external(self.df[Columns.User].values),
                Columns.Item: item_id_map.convert_to_external(self.df[Columns.Item].values),
            }
        )
        cols_to_add = []

        if include_weight:
            cols_to_add.append(Columns.Weight)
        if include_datetime:
            cols_to_add.append(Columns.Datetime)

        extra_cols = []
        if isinstance(include_extra_cols, list):
            extra_cols = [col for col in include_extra_cols if col in self.df and col not in Columns.Interactions]
        elif include_extra_cols:
            extra_cols = [col for col in self.df if col not in Columns.Interactions]
        cols_to_add.extend(extra_cols)

        for col in cols_to_add:
            res[col] = self.df[col]
        return res
