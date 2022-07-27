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

# pylint: disable=c-extension-no-member
"""Approximate Nearest Neighbours accelerators"""
from __future__ import annotations

import itertools
import typing as tp
from tempfile import NamedTemporaryFile

import nmslib
import numpy as np

from rectools import ExternalId, ExternalIds, InternalId, InternalIds
from rectools.dataset import IdMap

T = tp.TypeVar("T", bound="BaseNmslibRecommender")


class BaseNmslibRecommender:
    """
    Class implements base constructor parameters, pickling protocol
    and sort-truncate logic for `UserToItemAnnRecommender` and `ItemToItemAnnRecommender`.

    Parameters
    ----------
    item_vectors : ndarray
        Ndarray of item latent features of size ``(N, K)``,
        where `N` is the number if items and `K` is the
        number of features.
    item_id_map : dict(hashable, int) | rectools.datasets.IdMap
        Mappings from external item ids to internal item
        ids used by recommender. Values must be positive integers.
    index_top_k : int, default 0
        Number of items to return per knn query (in addition to `top_n` passed to
        `get_item_list_for_user`, `get_item_list_for_user_batch`,
        `get_item_list_for_item` or `get_item_list_for_item_batch`).
        In this case nmslib index query.
        This might be important in order to account for filters.
        See `self.index.knnQueryBatch` in `_compute_sorted_similar'
    index_init_params : optional(dict(str, str)), default None
        NMSLIB initialization parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    index_query_time_params: optional(dict(str, str)), default None
        NMSLIB query time parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    create_index_params: optional(dict(str, str)), default None
        NMSLIB index creation parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    index : FloatIndex, optional
        Optonal instance of FloatIndex. Exists for outside initialization.

    See Also
    --------
    UserToItemAnnRecommender
    ItemToItemAnnRecommender
    """

    def __init__(
        self,
        item_vectors: np.ndarray,
        item_id_map: tp.Union[IdMap, tp.Dict[ExternalId, InternalId]],
        index_top_k: int = 0,
        index_init_params: tp.Optional[tp.Dict[str, str]] = None,
        index_query_time_params: tp.Optional[tp.Dict[str, int]] = None,
        create_index_params: tp.Optional[tp.Dict[str, int]] = None,
        index: tp.Optional[nmslib.FloatIndex] = None,
    ) -> None:
        self.item_vectors = item_vectors
        if isinstance(item_id_map, dict):
            self.item_id_map = IdMap.from_dict(item_id_map)
        else:
            self.item_id_map = item_id_map
        self.index_top_k = index_top_k
        if index_init_params is None:
            self.index_init_params = {"method": "hnsw", "space": "cosinesimil"}
        else:
            self.index_init_params = index_init_params
        if index_query_time_params is None:
            self.index_query_time_params = {"efSearch": 100}
        else:
            self.index_query_time_params = index_query_time_params
        if create_index_params is None:
            self.create_index_params = {"M": 100, "efConstruction": 100, "post": 0}
        else:
            self.create_index_params = create_index_params
        self.index = nmslib.init(**self.index_init_params) if index is None else index

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        with NamedTemporaryFile() as file:
            self.index.saveIndex(filename=file.name)
            file.seek(0)
            index = file.read()
        serialize_dict = self.__dict__.copy()
        serialize_dict["index"] = index
        return serialize_dict

    def __setstate__(self, d: tp.Dict[str, tp.Any]) -> None:
        index = nmslib.init(**d["index_init_params"])

        with NamedTemporaryFile() as file:
            file.write(d["index"])
            file.flush()
            index.loadIndex(file.name)

        nmslib.setQueryTimeParams(index, d["index_query_time_params"])
        d["index"] = index
        self.__dict__.update(d)

    def fit(self: T, verbose: bool = False) -> T:
        """
        Create and fit `nmslib` index.

        Parameters
        ----------
        verbose : bool
            Verbosity switch, see `NMSLIB` documentation.

        Returns
        -------
        BaseNmslibRecommender
            Returns self.
        """
        self._build_index(verbose=verbose)
        return self

    def _build_index(self, verbose: bool) -> None:
        self.index.addDataPointBatch(self.item_vectors)

        self.index.createIndex(self.create_index_params, print_progress=verbose)
        nmslib.setQueryTimeParams(self.index, self.index_query_time_params)

    @staticmethod
    def _truncate_item_list(
        top_n: int,
        item_arrays: tp.Sequence[InternalIds],
        available_items: tp.Optional[tp.Sequence[InternalIds]] = None,
        self_indices: tp.Optional[InternalIds] = None,
    ) -> tp.Sequence[InternalIds]:
        """
        Take sequence of items-candidates, intersect them with whitelists of
        allowed items and return filtered and truncated sequences of items.

        Parameters
        ----------
        item_arrays : sequence
            Two dimensional array of item indices. Each element in the outer sequence
            represents a sequence of items for one user id.
        available_items : sequence
            Two dimensional array of item indices. Each element in the outer sequence
            represents a sequence of allowed items for one user id.
        self_indices : sequence
            A sequence of item (self) indices to filter. Used in item to item.

        Returns
        -------
        sequence(sequence(int))
            Two-dimensional array of filtered top-n items.
        """
        out = []
        if available_items is not None:
            for item_array, available_list in zip(item_arrays, available_items):
                available_set: tp.Set[int] = (
                    set(available_list) if self_indices is None else set(available_list).difference(set(self_indices))
                )
                truncated_item_list = list(itertools.islice((rec for rec in item_array if rec in available_set), top_n))
                out.append(truncated_item_list)
            return out

        for idx, item_array in enumerate(item_arrays):
            set_self_indices = {self_indices[idx]} if self_indices is not None else {}
            truncated_item_list = list(
                itertools.islice((rec for rec in item_array if rec not in set_self_indices), top_n)
            )
            out.append(truncated_item_list)

        return out

    def _map_to_external_id(self, item_arrays: tp.Sequence[InternalIds]) -> tp.Sequence[ExternalIds]:
        return [self.item_id_map.convert_to_external(item_array) for item_array in item_arrays]

    def _compute_sorted_similar(self, input_vectors: np.ndarray, top_n: int) -> tp.Sequence[InternalIds]:
        res = self.index.knnQueryBatch(input_vectors, k=top_n + self.index_top_k)
        res = np.vstack([out[0] for out in res])
        return res


class UserToItemAnnRecommender(BaseNmslibRecommender):
    """
    Class implements user to item ANN recommender.

    Parameters
    ----------
    user_vectors : ndarray
        Ndarray of user latent features of size ``(M, K)``,
        where `M` is the number of items and `K` is the
        number of features.
    item_vectors : ndarray
        Ndarray of item latent features of size ``(N, K)``,
        where `N` is the number of items and `K` is the
        number of features.
    user_id_map : dict(hashable, int) | rectools.dataset.IdMap
        Mappings from external user ids to internal user
        ids used by recommender. Values must be positive integers.
    item_id_map : dict(hashable, int) | rectools.dataset.IdMap
        Mappings from external item ids to internal item
        ids used by recommender. Values must be positive integers.
    index_top_k : int, default 0
        Number of items to return per knn query (in addition to `top_n` passed to
        `get_item_list_for_user`, `get_item_list_for_user_batch`,
        `get_item_list_for_item` or `get_item_list_for_item_batch`).
        This might be important in order to account for filters.
        See `self.index.knnQueryBatch` in `_compute_sorted_similar'
    index_init_params : optional(dict(str, str)), default None
        NMSLIB initialization parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    index_query_time_params: optional(dict(str, int)), default None
        NMSLIB query time parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    create_index_params: optional(dict(str, int)), default None
        NMSLIB index creation parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    index : FloatIndex, optional
        Optonal instance of `FloatIndex`. Exists for outside initialization.

    Methods
    -------
    get_item_list_for_user
        Part of public API. Given user id and item ids, calculates
        recommendations via index query.
    get_item_list_for_user_batch
        Part of public API. Does what get_item_list_for_user,
        except it takes a batch of user ids and a batch of item sets.

    See Also
    --------
    ItemToItemAnnRecommender
    """

    def __init__(
        self,
        user_vectors: np.ndarray,
        item_vectors: np.ndarray,
        user_id_map: tp.Union[IdMap, tp.Dict[ExternalId, InternalId]],
        item_id_map: tp.Union[IdMap, tp.Dict[ExternalId, InternalId]],
        index_top_k: int = 0,
        index_init_params: tp.Optional[tp.Dict[str, str]] = None,
        index_query_time_params: tp.Optional[tp.Dict[str, int]] = None,
        create_index_params: tp.Optional[tp.Dict[str, int]] = None,
        index: tp.Optional[nmslib.FloatIndex] = None,
    ) -> None:
        super().__init__(
            item_vectors=item_vectors,
            item_id_map=item_id_map,
            index_top_k=index_top_k,
            index_init_params=index_init_params,
            index_query_time_params=index_query_time_params,
            create_index_params=create_index_params,
            index=index,
        )
        self.user_vectors = user_vectors
        if isinstance(user_id_map, dict):
            self.user_id_map = IdMap.from_dict(user_id_map)
        else:
            self.user_id_map = user_id_map

        if self.user_vectors.shape[1] != self.item_vectors.shape[1]:
            raise ValueError(
                f"""Vectors shape mismatch:
                                 user vectors dim={self.user_vectors.shape[1]} !=
                                 item vectors dim={self.item_vectors.shape[1]}"""
            )

    def get_item_list_for_user(
        self, user_id: ExternalId, top_n: int, item_ids: tp.Optional[ExternalIds] = None
    ) -> ExternalIds:
        """
        Calculate top n recommendations for a given user id.

        Parameters
        ----------
        user_id : hashable
            User id used by external systems.
        top_n : int
            How many items to return.
        item_ids : optional(sequence(hashable)), default None
            A set of item ids from which to recommend.
            In case of None this function recommends without constraints.

        Returns
        -------
        sequence(hashable)
            Sorted sequence of external ids
        """
        user_id_ = self.user_id_map.convert_to_internal([user_id])
        item_ids_ = None
        if item_ids is not None:
            item_ids_ = [self.item_id_map.convert_to_internal(item_ids)]
        return self._get_item_list_from_index(user_id_, top_n, item_ids_)[0]

    def _get_item_list_from_index(
        self, user_ids: InternalIds, top_n: int, item_ids: tp.Optional[tp.Sequence[InternalIds]] = None
    ) -> tp.Sequence[ExternalIds]:
        user_vectors = self.user_vectors[user_ids, :]
        available_items = item_ids

        ids = self._compute_sorted_similar(input_vectors=user_vectors, top_n=top_n)
        return self._map_to_external_id(
            self._truncate_item_list(top_n, item_arrays=ids, available_items=available_items)
        )

    def get_item_list_for_user_batch(
        self,
        user_ids: ExternalIds,
        top_n: int,
        item_ids: tp.Optional[tp.Sequence[ExternalIds]] = None,
    ) -> tp.Sequence[ExternalIds]:
        """
        Calculate top-n recommendations for given user ids and item lists.
        Item lists define which items are allowed to be recommended.

        Parameters
        ----------
        user_ids : sequence(hashable)
            List of user ids used by external systems.
        top_n : int
            How many items to return.
        item_ids : optional(sequence(sequence(hashable))), default None
            List of lists of allowed items for each user id from user_ids in that exact order.
            In case of None this function recommends without constraints.

        Returns
        -------
        sequence(sequence(hashable))
            Sequence of sorted sequences of external ids.
        """
        user_ids_ = self.user_id_map.convert_to_internal(user_ids)
        item_ids_: tp.Optional[tp.Sequence[InternalIds]] = None
        if item_ids is not None:
            item_ids_ = [self.item_id_map.convert_to_internal(user_item_set) for user_item_set in item_ids]
        return self._get_item_list_from_index(user_ids_, top_n, item_ids_)


class ItemToItemAnnRecommender(BaseNmslibRecommender):
    """
    Class implements item-to-item ANN recommender.

    Parameters
    ----------
    item_vectors : ndarray
        Ndarray of item latent features of size ``(N, K)``,
        where `N` is the number of items and `K` is the
        number of features.
    item_id_map : dict(hashable, int) | rectools.datasets.IdMap
        Mappings from external item ids to internal item
        ids used by recommender. Values must be positive integers.
    index_top_k : int, default 0
        Number of items to return per knn query (in addition to `top_n` passed to
        `get_item_list_for_user`, `get_item_list_for_user_batch`,
        `get_item_list_for_item` or `get_item_list_for_item_batch`).
        In this case nmslib index query.
        This might be important in order to account for filters.
        See `self.index.knnQueryBatch` in `_compute_sorted_similar'
    index_init_params : optional(dict(str, str)) | rectools.dataset.IdMap
        NMSLIB initialization parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    index_query_time_params : optional(dict(str, int)) | rectools.dataset.IdMap
        NMSLIB query time parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    create_index_params : optional(dict(str, int)) | rectools.dataset.IdMap
        NMSLIB index creation parameters. See nmslib documentation.
        In case of None defaults to reasonable parameters.
    index : FloatIndex, optional
        Optonal instance of FloatIndex. Exists for outside initialization.

    Methods
    -------
    get_item_list_for_item
        Part of public API. Given item id and available item ids, calculates
        recommendations via index query.
    get_item_list_for_item_batch
        Part of public API. Does exactly what `get_item_list_for_item`,
        but for a batch of item ids and available item ids.

    See Also
    --------
    UserToItemAnnRecommender
    """

    def _get_item_list_from_index(
        self, item_ids: InternalIds, top_n: int, item_available_ids: tp.Optional[tp.Sequence[InternalIds]]
    ) -> tp.Sequence[ExternalIds]:
        item_vectors = self.item_vectors[item_ids, :]
        available_items = item_available_ids

        ids = self._compute_sorted_similar(input_vectors=item_vectors, top_n=top_n)
        return self._map_to_external_id(
            self._truncate_item_list(
                top_n=top_n, item_arrays=ids, available_items=available_items, self_indices=item_ids
            )
        )

    def get_item_list_for_item(
        self, item_id: ExternalId, top_n: int, item_available_ids: tp.Optional[ExternalIds] = None
    ) -> ExternalIds:
        """
        Calculate top-n recommendations for a given item id and item list.
        Item list defines which items are allowed to be recommended.

        Parameters
        ----------
        item_id : hashable
            Item id used by external systems.
        top_n : int
            How many items to return.
        item_available_ids : optional(sequence(hashable)), default None
            List of allowed items.
            In case of None this function recommends without constraints

        Returns
        -------
        sequence(hashable)
            Sorted sequence of external ids.
        """
        item_id_ = self.item_id_map.convert_to_internal([item_id])
        item_available_ids_: tp.Optional[tp.Sequence[InternalIds]] = None
        if item_available_ids is not None:
            item_available_ids_ = [self.item_id_map.convert_to_internal(item_available_ids)]

        return self._get_item_list_from_index(item_id_, top_n, item_available_ids_)[0]

    def get_item_list_for_item_batch(
        self,
        item_ids: ExternalIds,
        top_n: int,
        item_available_ids: tp.Optional[tp.Sequence[ExternalIds]] = None,
    ) -> tp.Sequence[ExternalIds]:
        """
        Calculate top-n recommendations for given item ids and item lists.
        Item lists define which items are allowed to be recommended.

        Parameters
        ----------
        item_ids : sequence(hashable)
            List of user ids used by external systems.
        top_n : int
            How many items to return.
        item_available_ids : optional(sequence(sequence(hashable))), default None
            List of lists of allowed items for each item id from item_ids in that exact order.
            In case of None this function recommends without constraints.

        Returns
        -------
        sequence(sequence(hashable))
            Sequence of sorted sequences of external ids.
        """
        item_ids_ = self.item_id_map.convert_to_internal(item_ids)
        item_available_ids_: tp.Optional[tp.Sequence[InternalIds]] = None
        if item_available_ids is not None:
            item_available_ids_ = [
                self.item_id_map.convert_to_internal(item_item_set) for item_item_set in item_available_ids
            ]
        return self._get_item_list_from_index(item_ids_, top_n, item_available_ids_)
