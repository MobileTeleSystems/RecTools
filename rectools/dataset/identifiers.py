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
import warnings

import attr
import numpy as np
import pandas as pd

from rectools import ExternalId, ExternalIds, InternalId, InternalIds
from rectools.utils import fast_isin, get_from_series_by_index


@attr.s(frozen=True, slots=True)
class IdMap:
    """Mapping between external and internal object ids.

    External ids may be any unique hashable values, internal - always integers from ``0`` to ``n_objects-1``.

    Usually you do not need to create this object directly, use `from_values` class method instead.

    When creating directly you have to pass unique external ids.

    Parameters
    ----------
    external_ids: np.ndarray
        Array of *unique* external ids.
    """

    external_ids: np.ndarray = attr.ib()

    @classmethod
    def from_values(cls, values: ExternalIds) -> "IdMap":
        """
        Create IdMap from list of external ids (possibly not unique).

        Parameters
        ----------
        values: iterable(hashable) :
            List of all external ids (may be not unique).

        Returns
        -------
        IdMap
        """
        unq_values = pd.unique(values)
        return cls(unq_values)

    @classmethod
    def from_dict(cls, mapping: tp.Dict[ExternalId, InternalId]) -> "IdMap":
        """
        Create IdMap from dict of external id -> internal id mappings.
        Could be used if mappings were previously defined somewhere else.

        Parameters
        ----------
        mapping: dict(hashable, int) :
            Dict of mappings from external ids to internal ids.
            Internal ids must be integers from 0 to n_objects-1.

        Returns
        -------
        IdMap
        """
        external_ids = list(mapping.keys())
        internal_ids = np.array([mapping[e] for e in external_ids])
        order = np.argsort(internal_ids)
        internal_ids_sorted = internal_ids[order]

        with warnings.catch_warnings():
            # When comparing numeric vs. non-numeric array returns scalar, will change in the future
            warnings.simplefilter("ignore", FutureWarning)
            internals_incorrect = internal_ids_sorted != np.arange(internal_ids_sorted.size)

        if internals_incorrect is True or internals_incorrect.any():
            raise ValueError("Internal ids must be integers from 0 to n_objects-1")

        res = np.array(external_ids)[order]
        return cls(res)

    @property
    def size(self) -> int:
        """Return number of ids in map."""
        return self.external_ids.size

    @property
    def to_internal(self) -> pd.Series:
        """Map internal->external."""
        return pd.Series(np.arange(self.size), index=self.external_ids)

    @property
    def to_external(self) -> pd.Series:
        """Map internal->external."""
        return pd.Series(self.external_ids, index=pd.RangeIndex(0, self.size))

    @property
    def internal_ids(self) -> np.ndarray:
        """Array of internal ids."""
        return np.arange(self.size)

    def get_sorted_internal(self) -> np.ndarray:
        """Return array of sorted internal ids."""
        return self.internal_ids

    def get_external_sorted_by_internal(self) -> np.ndarray:
        """Return array of external ids sorted by internal ids."""
        return self.external_ids

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

    def add_ids(self, values: ExternalIds, raise_if_already_present: bool = False) -> "IdMap":
        """
        Add new external ids to current IdMap and return new IdMap.
        Mapping for old ids does not change.
        New ids are added to the end of list of external ids.

        Parameters
        ----------
        values : iterable(hashable)
            List of new external ids (may be not unique).
        raise_if_already_present : bool, default ``False``
            If True and some of given ids are already present in the map
            ValueError will be raised.

        Returns
        -------
        IdMap

        Raises
        ------
        ValueError
            If some of given ids are already present in the map and `raise_if_already_present` flag is ``True``.
        """
        unq = pd.unique(values)
        new_ids = unq[fast_isin(unq, self.external_ids, invert=True)]
        if raise_if_already_present and new_ids.size < unq.size:
            raise ValueError("Some of new ids are already present in map")
        full_external_ids = np.concatenate((self.external_ids, new_ids))
        return self.__class__(full_external_ids)
