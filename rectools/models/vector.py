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
from enum import Enum

import attr
import implicit.cpu
import numpy as np
from implicit.cpu.matrix_factorization_base import _filter_items_from_sparse_matrix as filter_items_from_sparse_matrix
from scipy import sparse

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.models.base import ModelBase, Scores


class Distance(Enum):
    """Distance metric"""

    DOT = 1  # Bigger value means closer vectors
    COSINE = 2  # Bigger value means closer vectors
    EUCLIDEAN = 3  # Smaller value means closer vectors


@attr.s(auto_attribs=True)
class Factors:
    """Embeddings and biases"""

    embeddings: np.ndarray
    biases: tp.Optional[np.ndarray] = None


class ImplicitRanker:
    """
    Ranker for vector-based models which uses implicit library matrix factorization topk method

    Parameters
    ----------
    distance : Distance
        Distance metric.
    subjects_factors : np.ndarray
        Array of subject embeddings, shape (n_subjects, n_factors).
    objects_factors : np.ndarray
        Array with embeddings of all objects, shape (n_objects, n_factors).
    """

    def __init__(self, distance: Distance, subjects_factors: np.ndarray, objects_factors: np.ndarray) -> None:
        self.distance = distance
        self.subjects_factors = subjects_factors.astype(np.float32)
        self.objects_factors = objects_factors.astype(np.float32)

        self.subjects_norms: np.ndarray
        if distance == Distance.COSINE:
            self.subjects_norms = self._calc_norms(self.subjects_factors, avoid_zeros=True)

        self.subjects_dots: np.ndarray
        if distance == Distance.EUCLIDEAN:
            self.subjects_dots = self._calc_dots(self.subjects_factors)

    def _get_neginf_score(self) -> float:
        # Adding 1 to avoid float calculation errors (we're comparing `scores <= neginf_score`)
        return -np.finfo(np.float32).max + 1

    @staticmethod
    def _calc_dots(factors: np.ndarray) -> np.ndarray:
        return (factors**2).sum(axis=1)

    @staticmethod
    def _calc_norms(factors: np.ndarray, avoid_zeros: bool = False) -> np.ndarray:
        norms = np.linalg.norm(factors, axis=1)
        # Used for COSINE distance
        # If one or both vectors are zero, assume they're orthogonal, need to avoid 0 in denominator
        if avoid_zeros:
            norms[norms == 0] = 1e-10
        return norms

    def _get_mask_for_correct_scores(self, scores: np.ndarray) -> tp.List[bool]:
        """Filter scores from implicit library that are not relevant. Implicit library assigns `neginf` score
        to items that are meant to be filtered (e.g. blacklist items or already seen items)
        """
        num_masked = 0
        min_score = self._get_neginf_score()
        for el in np.flip(scores):
            if el <= min_score:
                num_masked += 1
            else:
                break
        return [True] * (len(scores) - num_masked) + [False] * num_masked

    def _process_implicit_scores(
        self, subject_ids: InternalIds, ids: np.ndarray, scores: np.ndarray
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:

        all_target_ids = []
        all_reco_ids: tp.List[np.ndarray] = []
        all_scores: tp.List[np.ndarray] = []

        for subject_id, object_ids, object_scores in zip(subject_ids, ids, scores):
            correct_mask = self._get_mask_for_correct_scores(object_scores)
            relevant_scores = object_scores[correct_mask]
            relevant_ids = object_ids[correct_mask]

            if self.distance == Distance.COSINE:
                subject_norm = self.subjects_norms[subject_id]
                relevant_scores /= subject_norm

            if self.distance == Distance.EUCLIDEAN:
                # Restore Euclidean distances from scores
                d2 = self.subjects_dots[subject_id] - relevant_scores
                # Theoretically d2 >= 0, but can be <0 because of rounding errors
                relevant_scores = np.sqrt(np.maximum(d2, 0))

            all_target_ids.extend([subject_id for _ in range(len(relevant_ids))])
            all_reco_ids.append(relevant_ids)
            all_scores.append(relevant_scores)

        return all_target_ids, np.concatenate(all_reco_ids), np.concatenate(all_scores)

    def rank(
        self,
        subject_ids: InternalIds,
        k: int,
        filter_pairs_csr: tp.Optional[sparse.csr_matrix] = None,  # subect-object interactions, relevant for u2i case
        sorted_object_whitelist: tp.Optional[np.ndarray] = None,
        num_threads: int = 0,
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        """Proceed inference using implicit library matrix factorization topk cpu method"""
        if sorted_object_whitelist is not None:
            object_factors = self.objects_factors[sorted_object_whitelist]

            if filter_pairs_csr is not None:
                #  filter ui_csr_for_filter matrix to contain only whitelist objects
                filter_query_items = filter_items_from_sparse_matrix(sorted_object_whitelist, filter_pairs_csr)
            else:
                filter_query_items = None

        else:
            # keep all objects and full ui_csr_for_filter
            object_factors = self.objects_factors
            filter_query_items = filter_pairs_csr

        subject_factors = self.subjects_factors[subject_ids]

        object_norms = None  # for DOT and EUCLIDEAN distance
        if self.distance == Distance.COSINE:
            object_norms = self._calc_norms(object_factors, avoid_zeros=True)

        if self.distance == Distance.EUCLIDEAN:
            # Transform factors to get top-k by Euclidean distance using Dot metric
            # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
            subject_factors = np.hstack((-np.ones((subject_factors.shape[0], 1)), 2 * subject_factors))
            object_factors = np.hstack(((object_factors**2).sum(axis=1).reshape(-1, 1), object_factors))

        real_k = min(k, object_factors.shape[0])

        ids, scores = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
            items=object_factors,
            query=subject_factors,
            k=real_k,
            item_norms=object_norms,  # query norms for COSINE distance are applied afterwards
            filter_query_items=filter_query_items,  # queries x objects csr matrix for getting neginf scores
            filter_items=None,  # rectools doesn't support blacklist for now
            num_threads=num_threads,
        )

        if sorted_object_whitelist is not None:
            ids = sorted_object_whitelist[ids]

        # filter neginf from implicit scores and apply transformations to scores (for COSINE and EUCLIDEAN distances)
        all_target_ids, all_reco_ids, all_scores = self._process_implicit_scores(subject_ids, ids, scores)

        return all_target_ids, all_reco_ids, all_scores


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
