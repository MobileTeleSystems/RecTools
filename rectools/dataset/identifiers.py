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

"""Mapping between external and internal ids."""

import typing as tp

import attr
import numpy as np
import pandas as pd

from rectools import ExternalId, ExternalIds, InternalId, InternalIds
from rectools.utils import get_from_series_by_index


@attr.s(frozen=True, slots=True)
class IdMap:
    """Mapping between external and internal object ids.

    External ids may be any unique hashable values, internal - always integers from ``0`` to ``n_objects-1``.

    Usually you do not need to create this object directly, use `from_values` class method instead.

    Parameters
    ----------
    to_internal: pd.Series
        Mapping external -> internal ids.
    """

    to_internal: pd.Series = attr.ib()

    @to_internal.validator
    def _check_internal_correct(self, _: str, value: pd.Series) -> None:
        """Check that internal ids are correct."""
        if (np.sort(value.values) != np.arange(len(value))).any():
            raise ValueError("Internal ids must be integers from 0 to n_objects-1")

    @to_internal.validator
    def _check_external_unique(self, _: str, value: pd.Series) -> None:  # noqa: D102
        """Check that external ids are unique."""
        if np.unique(value.index.values).size != len(value):
            raise ValueError("External ids must be unique")

    @classmethod
    def from_values(cls, values: ExternalIds) -> "IdMap":
        """
        Create IdMap from list of external ids.

        Parameters
        ----------
        values: iterable(hashable) :
            List of external ids (may be not unique).

        Returns
        -------
        IdMap
        """
        unq_values = np.unique(np.asarray(values))
        ids = np.arange(len(unq_values))
        return cls(pd.Series(ids, index=unq_values))

    @classmethod
    def from_dict(cls, mapping: tp.Dict[ExternalId, InternalId]) -> "IdMap":
        """
        Create IdMap from dict of external id -> internal id mappings.
        Could be used if mappings were previously defined somewhere else.

        Parameters
        ----------
        mapping: dict(hashable, int) :
            Dict of mappings from external ids to internal ids.

        Returns
        -------
        IdMap
        """
        return cls(pd.Series(mapping))

    @property
    def to_external(self) -> pd.Series:
        """Map internal->external."""
        return pd.Series(self.to_internal.index.values, index=self.to_internal.values)

    @property
    def internal_ids(self) -> np.ndarray:
        """Array of internal ids."""
        return self.to_internal.values

    @property
    def external_ids(self) -> np.ndarray:
        """Array of external ids."""
        return self.to_internal.index.values

    def get_sorted_internal(self) -> np.ndarray:
        """Return array of sorted internal ids."""
        return np.sort(self.to_internal.values)

    def get_external_sorted_by_internal(self) -> np.ndarray:
        """Return array of external ids sorted by internal ids."""
        return self.to_internal.index.values[np.argsort(self.to_internal.values)]

    def convert_to_internal(self, external: ExternalIds, strict: bool = True) -> np.ndarray:
        """
        Convert any sequence of external ids to array of internal ids (map external -> internal).

        Parameters
        ----------
        external : sequence(hashable)
            Sequence of external ids to convert.
        strict : bool, default ``True``
             Defines behaviour when some of given external ids do not exist in mapping.
                - If ``True``, `KeyError` will be raised;
                - If ``False``, nonexistent ids will be skipped.

        Returns
        -------
        np.ndarray
            Array of internal ids.

        Raises
        ------
        KeyError
            If some of given external ids do not exist in mapping and `strict` flag is ``True``.
        """
        internal = get_from_series_by_index(self.to_internal, external, strict)
        return internal

    def convert_to_external(self, internal: InternalIds, strict: bool = True) -> np.ndarray:
        """
        Convert any sequence of internal ids to array of external ids (map internal -> external).

        Parameters
        ----------
        internal : sequence(int)
            Sequence of internal ids to convert.
        strict : bool, default ``True``
             Defines behaviour when some of given internal ids do not exist in mapping.
                - If ``True``, `KeyError` will be raised;
                - If ``False``, nonexistent ids will be skipped.

        Returns
        -------
        np.ndarray
            Array of external ids.

        Raises
        ------
        KeyError
            If some of given internal ids do not exist in mapping and `strict` flag is True.
        """
        external = get_from_series_by_index(self.to_external, internal, strict)
        return external
