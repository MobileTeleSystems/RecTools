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

    def _get_neginf_score(self) -> float:
        dummy_factors = np.array([[1, 2]], dtype=np.float32)
        neginf = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
            items=dummy_factors,  # overall whitelist item factors
            query=dummy_factors,  # query_factors
            k=1,
            filter_items=np.array([0]),
        )[1][0][0]
        return neginf

    
    def _get_mask_for_correct_scores(self, scores: np.ndarray, min_score: float = 3e-38):
        num_masked = 0
        for el in np.flip(scores):
            if el == 0 or el <= min_score:
                num_masked +=1
            else:
                break
        return [True for _ in range(len(scores) - num_masked)] + [False for _ in range(num_masked)]

    def _process_implicit_scores(
        self, subject_ids: np.ndarray, ids: np.ndarray, scores: np.ndarray, apply_norm: bool
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        # neginf = self._get_neginf_score()
        # max_pos = scores.shape[1]

        all_target_ids = []
        all_reco_ids: tp.List[np.ndarray] = []
        all_scores: tp.List[np.ndarray] = []

        for i in range(scores.shape[0]):
            correct_mask = self._get_mask_for_correct_scores(scores[i])
            relevant_scores = scores[i][correct_mask]
            relevant_ids = ids[i][correct_mask]
            #neginf_start_pos = max_pos - np.searchsorted(np.flip(scores[i]), np.array([neginf]), side="right")[0]
            #relevant_scores = scores[i][:neginf_start_pos]
            #relevant_ids = ids[i][:neginf_start_pos]

            if apply_norm:
                subject_norm = self.subjects_norms[subject_ids[i]]
                subject_norm = 1e-10 if subject_norm == 0 else subject_norm
                relevant_scores /= subject_norm

            all_target_ids.extend([subject_ids[i] for _ in range(len(relevant_ids))])
            all_reco_ids.append(relevant_ids)
            all_scores.append(relevant_scores)

        return all_target_ids, np.concatenate(all_reco_ids), np.concatenate(all_scores)

    def calc_batch_scores_via_implicit_matrix_topk(
        self,
        subject_ids: np.ndarray,
        k: int,
        user_items_csr_for_filter_viewed: tp.Optional[sparse.csr_matrix],  # only relevant for u2i recos
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],  # whitelist
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        """Proceed inference using implicit library matrix factorization topk cpu method"""
        if self.distance == Distance.EUCLIDEAN:
            raise ValueError("Implicit Matrix topk scoring method is not suitable for Euclidean distance")

        if sorted_item_ids_to_recommend is not None:
            object_factors_whitelist = self.objects_factors[sorted_item_ids_to_recommend]
            if user_items_csr_for_filter_viewed is not None:
                filter_query_items = implicit.cpu.matrix_factorization_base._filter_items_from_sparse_matrix(
                    sorted_item_ids_to_recommend, user_items_csr_for_filter_viewed
                )
            else:
                filter_query_items = user_items_csr_for_filter_viewed

        else:
            object_factors_whitelist = self.objects_factors
            filter_query_items = user_items_csr_for_filter_viewed

        subject_factors = self.subjects_factors[subject_ids]

        object_norms = None
        if self.distance == Distance.COSINE:
            object_norms = self.objects_norms
            if sorted_item_ids_to_recommend is not None:
                object_norms = object_norms[sorted_item_ids_to_recommend]
            if object_norms is not None:
                object_norms[object_norms == 0] = 1e-10

        ids, scores = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
            items=object_factors_whitelist,  # overall whitelist item factors
            query=subject_factors,  # query_factors
            k=k,
            item_norms=object_norms,  # for COSINE we pass norms. query norms is applied after
            # filter_query_items = user_items_csr for filter_viewed=True. these items get neginf score
            filter_query_items=filter_query_items,
            # can't be both 'items' and 'filter_items'. items to score as neginf. rectools doesn't support
            filter_items=None,
            # num_threads=self.num_threads, TODO: pass num_threads from implicit model
        )

        if sorted_item_ids_to_recommend is not None:
            ids = sorted_item_ids_to_recommend[ids]

        # filter neginf from implicit results and apply norms from correct cosine scores
        all_target_ids, all_reco_ids, all_scores = self._process_implicit_scores(
            subject_ids, ids, scores, apply_norm=self.distance == Distance.COSINE
        )

        return all_target_ids, all_reco_ids, all_scores


class VectorModel(ModelBase):
    """Base class for models that represents users and items as vectors"""

    u2i_dist: Distance = NotImplemented
    i2i_dist: Distance = NotImplemented
    use_implicit: bool = True  # TODO: remove

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

        scores_calculator = self._get_u2i_calculator(dataset)

        if self.use_implicit and self.u2i_dist in (Distance.COSINE, Distance.DOT):
            user_items_csr_for_filter_viewed = user_items[user_ids] if filter_viewed else None
            return scores_calculator.calc_batch_scores_via_implicit_matrix_topk(
                subject_ids=user_ids,
                k=k,
                user_items_csr_for_filter_viewed=user_items_csr_for_filter_viewed,
                sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
            )

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
        scores_calculator = self._get_i2i_calculator(dataset)

        if self.use_implicit and self.i2i_dist in (Distance.COSINE, Distance.DOT):
            return scores_calculator.calc_batch_scores_via_implicit_matrix_topk(
                subject_ids=target_ids,
                k=k,
                user_items_csr_for_filter_viewed=None,
                sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
            )

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

    def _get_u2i_calculator(self, dataset: Dataset) -> ScoreCalculator:
        user_factors = self._get_users_factors(dataset)
        item_factors = self._get_items_factors(dataset)

        user_vectors = user_factors.embeddings
        item_vectors = item_factors.embeddings
        user_biases = user_factors.biases
        item_biases = item_factors.biases

        # TODO: make it possible to control if add biases or not (even if they present)
        if user_biases is not None and item_biases is not None:
            if self.u2i_dist == Distance.DOT:
                user_vectors = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_vectors))
                item_vectors = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_vectors))
            elif self.u2i_dist in (Distance.COSINE, Distance.EUCLIDEAN):
                user_vectors = np.hstack((user_biases[:, np.newaxis], user_vectors))
                item_vectors = np.hstack((item_biases[:, np.newaxis], item_vectors))
            else:
                raise ValueError(f"Unexpected distance `{self.u2i_dist}`")

        return ScoreCalculator(self.u2i_dist, user_vectors, item_vectors)

    def _get_i2i_calculator(self, dataset: Dataset) -> ScoreCalculator:
        item_factors = self._get_items_factors(dataset)
        item_vectors = item_factors.embeddings
        item_biases = item_factors.biases
        item_vectors_1 = item_vectors_2 = item_vectors

        if item_biases is not None:  # TODO: make it possible to control if add biases or not (even if they present)
            if self.i2i_dist == Distance.DOT:
                item_vectors_1 = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_vectors))
                item_vectors_2 = np.hstack((item_biases[:, np.newaxis], np.ones((item_biases.size, 1)), item_vectors))
            elif self.i2i_dist in (Distance.COSINE, Distance.EUCLIDEAN):
                item_vectors_1 = item_vectors_2 = np.hstack((item_biases[:, np.newaxis], item_vectors))
            else:
                raise ValueError(f"Unexpected distance `{self.u2i_dist}`")
        return ScoreCalculator(self.i2i_dist, item_vectors_1, item_vectors_2)

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        raise NotImplementedError()

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        raise NotImplementedError()
