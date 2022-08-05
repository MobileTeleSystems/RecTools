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
import numpy as np
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


class VectorModel(ModelBase):
    """Base class for models that represents users and items as vectors"""

    u2i_dist: Distance = NotImplemented
    i2i_dist: Distance = NotImplemented

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

        scores_calculator = self._get_u2i_calculator(dataset)

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
