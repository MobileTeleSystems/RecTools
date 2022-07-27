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

"""Structure for saving user-item interactions."""

import attr
import numpy as np
import pandas as pd
from scipy import sparse

from rectools import Columns

from .identifiers import IdMap


@attr.s(frozen=True, slots=True)
class Interactions:
    """
    Structure to storage info about user-item interactions.

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
    """

    df: pd.DataFrame = attr.ib()

    @staticmethod
    def _check_columns_present(df: pd.DataFrame) -> None:
        required_columns = {Columns.User, Columns.Item, Columns.Weight, Columns.Datetime}
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(f"Missed columns {required_columns - actual_columns}")

    @staticmethod
    def _convert_weight_and_datetime_types(df: pd.DataFrame) -> None:
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
        self._convert_weight_and_datetime_types(self.df)

    @classmethod
    def from_raw(
        cls,
        interactions: pd.DataFrame,
        user_id_map: IdMap,
        item_id_map: IdMap,
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

        Returns
        -------
            Interactions
        """
        cls._check_columns_present(interactions)

        df = pd.DataFrame(
            {
                Columns.User: interactions[Columns.User].map(user_id_map.to_internal),
                Columns.Item: interactions[Columns.Item].map(item_id_map.to_internal),
            },
        )
        df[Columns.Weight] = interactions[Columns.Weight]
        df[Columns.Datetime] = interactions[Columns.Datetime]
        cls._convert_weight_and_datetime_types(df)

        return cls(df)

    def get_user_item_matrix(self, include_weights: bool = True) -> sparse.csr_matrix:
        """
        Form an user-item CSR matrix based on `interactions` attribute.

        It is used `Interactions.get_user_item_matrix`, see its documentations for details.

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
                values.astype(np.float32),
                (
                    self.df[Columns.User].values,
                    self.df[Columns.Item].values,
                ),
            ),
        )
        return csr
