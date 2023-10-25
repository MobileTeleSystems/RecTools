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
from tqdm.auto import tqdm

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.models.base import ModelBase, Scores
from rectools.models.utils import get_viewed_item_ids, recommend_from_scores


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
    Ranker for DOT and COSINE similarity distance which uses implicit library matrix factorization topk method

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
        if distance not in (Distance.DOT, Distance.COSINE):
            raise ValueError(f"ImplicitRanker is not suitable for {distance} distance")
        self.distance = distance
        self.subjects_factors = subjects_factors.astype(np.float32)
        self.objects_factors = objects_factors.astype(np.float32)

        self.subjects_norms: np.ndarray
        self.objects_norms: np.ndarray
        if distance == Distance.COSINE:
            self.subjects_norms = np.linalg.norm(self.subjects_factors, axis=1)
            self.objects_norms = np.linalg.norm(self.objects_factors, axis=1)

    def _get_neginf_score(self) -> float:
        return -np.finfo(np.float32).max

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
        self, subject_ids: np.ndarray, ids: np.ndarray, scores: np.ndarray
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
                subject_norm = 1e-10 if subject_norm == 0 else subject_norm
                relevant_scores /= subject_norm

            all_target_ids.extend([subject_id for _ in range(len(relevant_ids))])
            all_reco_ids.append(relevant_ids)
            all_scores.append(relevant_scores)

        return all_target_ids, np.concatenate(all_reco_ids), np.concatenate(all_scores)

    def calc_batch_scores_via_implicit_matrix_topk(
        self,
        subject_ids: np.ndarray,
        k: int,
        ui_csr_for_filter: tp.Optional[sparse.csr_matrix],  # only relevant for u2i recos
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],  # whitelist
        num_threads: int = 0,
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        """Proceed inference using implicit library matrix factorization topk cpu method"""
        if sorted_item_ids_to_recommend is not None:
            object_factors_whitelist = self.objects_factors[sorted_item_ids_to_recommend]

            if ui_csr_for_filter is not None:
                #  filter ui_csr_for_filter matrix to contain only whitelist objects
                filter_query_items = filter_items_from_sparse_matrix(sorted_item_ids_to_recommend, ui_csr_for_filter)
            else:
                filter_query_items = None

        else:
            # keep all objects and full ui_csr_for_filter
            object_factors_whitelist = self.objects_factors
            filter_query_items = ui_csr_for_filter

        subject_factors = self.subjects_factors[subject_ids]

        object_norms = None  # for DOT distance
        if self.distance == Distance.COSINE:
            object_norms = self.objects_norms
            if sorted_item_ids_to_recommend is not None:
                object_norms = object_norms[sorted_item_ids_to_recommend]

        if object_norms is not None:  # prevent zero division
            object_norms[object_norms == 0] = 1e-10

        real_k = min(k, object_factors_whitelist.shape[0])

        ids, scores = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
            items=object_factors_whitelist,
            query=subject_factors,
            k=real_k,
            item_norms=object_norms,  # query norms for COSINE distance are applied afterwards
            filter_query_items=filter_query_items,  # queries x objects csr matrix for getting neginf scores
            filter_items=None,  # rectools doesn't support blacklist for now
            num_threads=num_threads,
        )

        if sorted_item_ids_to_recommend is not None:
            ids = sorted_item_ids_to_recommend[ids]

        # filter neginf from implicit scores and apply norms for correct COSINE distance
        all_target_ids, all_reco_ids, all_scores = self._process_implicit_scores(subject_ids, ids, scores)

        return all_target_ids, all_reco_ids, all_scores


class ScoreCalculator:
    """
    Calculate proximity scores between one subject (e.g. user) and all objects (e.g. items)
    according to given distance metric.

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
        self.subjects_factors = subjects_factors
        self.objects_factors = objects_factors

        self.subjects_norms: np.ndarray
        self.objects_norms: np.ndarray
        self.zero_objects_mask: np.ndarray = np.zeros(objects_factors.shape[0], dtype=bool)
        self.has_zero_object = False  # For optimization only
        if distance == Distance.COSINE:
            self.subjects_norms = np.linalg.norm(subjects_factors, axis=1)
            self.objects_norms = np.linalg.norm(objects_factors, axis=1)
            self.zero_objects_mask[self.objects_norms == 0] = True
            self.has_zero_object = bool(self.zero_objects_mask.any())

        self.subjects_dots: np.ndarray
        self.objects_dots: np.ndarray
        if distance == Distance.EUCLIDEAN:
            self.subjects_dots = (subjects_factors**2).sum(axis=1)
            self.objects_dots = (objects_factors**2).sum(axis=1)

    def calc(self, subject_id: int) -> np.ndarray:
        """
        Calculate proximity scores between one subject and all objects according to given distance metric.

        Parameters
        ----------
        subject_id : int
            Subject index.

        Returns
        -------
        np.ndarray
            Array of scores, shape (n_objects,).
        """
        subject_factors = self.subjects_factors[subject_id]
        if self.distance == Distance.DOT:
            scores = self.objects_factors @ subject_factors
        elif self.distance == Distance.EUCLIDEAN:
            subject_dot = self.subjects_dots[subject_id]
            dot = self.objects_factors @ subject_factors
            d2 = self.objects_dots + subject_dot - 2 * dot
            scores = np.sqrt(np.maximum(d2, 0))  # Theoretically d2 >= 0, but can be <0 because of rounding errors
        elif self.distance == Distance.COSINE:
            subject_norm = self.subjects_norms[subject_id]
            if subject_norm == 0:
                scores = np.zeros_like(self.objects_norms)
            else:
                scores = (self.objects_factors @ subject_factors) / (self.objects_norms * subject_norm)
                if self.has_zero_object:
                    scores[self.zero_objects_mask] = 0  # If one or both vectors are zero, assume they're orthogonal
        else:
            raise ValueError(f"Unexpected distance `{self.distance}`")

        return scores


class VectorModel(ModelBase):
    """Base class for models that represents users and items as vectors"""

    u2i_dist: Distance = NotImplemented
    i2i_dist: Distance = NotImplemented
    n_threads: int = 0  # TODO: decide how to pass it correctly for all models
    use_implicit: bool = True  # TODO: remove. For bedug only

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

        if self.use_implicit and self.u2i_dist in (Distance.COSINE, Distance.DOT):

            ranker = ImplicitRanker(self.u2i_dist, user_vectors, item_vectors)
            ui_csr_for_filter = user_items[user_ids] if filter_viewed else None
            return ranker.calc_batch_scores_via_implicit_matrix_topk(
                subject_ids=user_ids,
                k=k,
                ui_csr_for_filter=ui_csr_for_filter,
                sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
                num_threads=self.n_threads,
            )

        scores_calculator = ScoreCalculator(self.u2i_dist, user_vectors, item_vectors)
        all_target_ids = []
        all_reco_ids: tp.List[np.ndarray] = []
        all_scores: tp.List[np.ndarray] = []
        for target_id in tqdm(user_ids, disable=self.verbose == 0):
            scores = scores_calculator.calc(target_id)
            reco_ids, reco_scores = recommend_from_scores(
                scores=scores,
                k=k,
                sorted_blacklist=get_viewed_item_ids(user_items, target_id) if filter_viewed else None,
                sorted_whitelist=sorted_item_ids_to_recommend,
                ascending=self.u2i_dist == Distance.EUCLIDEAN,
            )
            all_target_ids.extend([target_id] * len(reco_ids))
            all_reco_ids.append(reco_ids)
            all_scores.append(reco_scores)

        return all_target_ids, np.concatenate(all_reco_ids), np.concatenate(all_scores)

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        item_vectors_1, item_vectors_2 = self._get_i2i_vectors(dataset)

        if self.use_implicit and self.i2i_dist in (Distance.COSINE, Distance.DOT):
            ranker = ImplicitRanker(self.i2i_dist, item_vectors_1, item_vectors_2)

            return ranker.calc_batch_scores_via_implicit_matrix_topk(
                subject_ids=target_ids,
                k=k,
                ui_csr_for_filter=None,
                sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
                num_threads=self.n_threads,
            )

        scores_calculator = ScoreCalculator(self.i2i_dist, item_vectors_1, item_vectors_2)
        all_target_ids = []
        all_reco_ids: tp.List[np.ndarray] = []
        all_scores: tp.List[np.ndarray] = []
        for target_id in tqdm(target_ids, disable=self.verbose == 0):
            scores = scores_calculator.calc(target_id)
            reco_ids, reco_scores = recommend_from_scores(
                scores=scores,
                k=k,
                sorted_blacklist=None,
                sorted_whitelist=sorted_item_ids_to_recommend,
                ascending=self.i2i_dist == Distance.EUCLIDEAN,
            )
            all_target_ids.extend([target_id] * len(reco_ids))
            all_reco_ids.append(reco_ids)
            all_scores.append(reco_scores)

        return all_target_ids, np.concatenate(all_reco_ids), np.concatenate(all_scores)

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
