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

"""Base classes for vector models."""

import typing as tp

import attr
import numpy as np

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.models.base import ModelBase, Scores

from .rank import Distance, ImplicitRanker


@attr.s(auto_attribs=True)
class Factors:
    """Embeddings and biases"""

    embeddings: np.ndarray
    biases: tp.Optional[np.ndarray] = None


class VectorModel(ModelBase):
    """Base class for models that represents users and items as vectors"""

    u2i_dist: Distance = NotImplemented
    i2i_dist: Distance = NotImplemented
    n_threads: int = 0  # TODO: decide how to pass it correctly for all models

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        if filter_viewed:
            user_items = dataset.get_user_item_matrix(include_weights=False)
        else:
            user_items = None

        user_vectors, item_vectors = self._get_u2i_vectors(dataset)

        ranker = ImplicitRanker(self.u2i_dist, user_vectors, item_vectors)
        ui_csr_for_filter = user_items[user_ids] if filter_viewed else None
        return ranker.rank(
            subject_ids=user_ids,
            k=k,
            filter_pairs_csr=ui_csr_for_filter,
            sorted_object_whitelist=sorted_item_ids_to_recommend,
            num_threads=self.n_threads,
        )

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        item_vectors_1, item_vectors_2 = self._get_i2i_vectors(dataset)

        ranker = ImplicitRanker(self.i2i_dist, item_vectors_1, item_vectors_2)

        return ranker.rank(
            subject_ids=target_ids,
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,
            num_threads=self.n_threads,
        )

    def _process_biases_to_vectors(
        self,
        distance: Distance,
        subject_embeddings: np.ndarray,
        subject_biases: np.ndarray,
        object_embeddings: np.ndarray,
        object_biases: np.ndarray,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        # TODO: make it possible to control if add biases or not (even if they are present)
        if distance == Distance.DOT:
            subject_vectors = np.hstack(
                (subject_biases[:, np.newaxis], np.ones((subject_biases.size, 1)), subject_embeddings)
            )
            object_vectors = np.hstack(
                (np.ones((object_biases.size, 1)), object_biases[:, np.newaxis], object_embeddings)
            )
        elif distance in (Distance.COSINE, Distance.EUCLIDEAN):
            subject_vectors = np.hstack((subject_biases[:, np.newaxis], subject_embeddings))
            object_vectors = np.hstack((object_biases[:, np.newaxis], object_embeddings))
        else:
            raise ValueError(f"Unexpected distance `{distance}`")
        return subject_vectors, object_vectors

    def _get_u2i_vectors(self, dataset: Dataset) -> tp.Tuple[np.ndarray, np.ndarray]:
        user_factors = self._get_users_factors(dataset)
        item_factors = self._get_items_factors(dataset)

        user_vectors = user_factors.embeddings
        item_vectors = item_factors.embeddings
        user_biases = user_factors.biases
        item_biases = item_factors.biases

        if user_biases is not None and item_biases is not None:
            user_vectors, item_vectors = self._process_biases_to_vectors(
                self.u2i_dist, user_vectors, user_biases, item_vectors, item_biases
            )

        return user_vectors, item_vectors

    def _get_i2i_vectors(self, dataset: Dataset) -> tp.Tuple[np.ndarray, np.ndarray]:
        item_factors = self._get_items_factors(dataset)
        item_vectors = item_factors.embeddings
        item_biases = item_factors.biases
        item_vectors_1 = item_vectors_2 = item_vectors

        if item_biases is not None:
            item_vectors_1, item_vectors_2 = self._process_biases_to_vectors(
                self.i2i_dist, item_vectors, item_biases, item_vectors, item_biases
            )

        return item_vectors_1, item_vectors_2

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        raise NotImplementedError()

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        raise NotImplementedError()
