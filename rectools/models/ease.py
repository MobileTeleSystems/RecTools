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

"""EASE model."""

import typing as tp

import numpy as np
from scipy import sparse
from tqdm import tqdm

from rectools import InternalIds
from rectools.dataset import Dataset

from .base import ModelBase, Scores
from .vector import Distance, ImplicitRanker
from .utils import get_viewed_item_ids, recommend_from_scores


class EASEModel(ModelBase):
    """
    Embarrassingly Shallow Autoencoders for Sparse Data model.

    See https://arxiv.org/abs/1905.03375.

    Parameters
    ----------
    weight : np.matrix
        The Item-Item weight-matrix.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    """

    u2i_dist = Distance.DOT

    def __init__(
        self,
        regularization: float = 500.0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.regularization = regularization
        self.weight: np.ndarray

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        gram_matrix = ui_csr.T @ ui_csr
        gram_matrix += self.regularization * sparse.identity(gram_matrix.shape[0]).astype(np.float32)
        gram_matrix = gram_matrix.todense()

        gram_matrix_inv = np.linalg.inv(gram_matrix)

        self.weight = np.array(gram_matrix / (-np.diag(gram_matrix_inv)))
        np.fill_diagonal(self.weight, 0.0)

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        user_items = dataset.get_user_item_matrix(include_weights=True)

        ranker = ImplicitRanker(
            distance=self.u2i_dist,
            subjects_factors=user_items,
            objects_factors=self.weight,
        )
        ui_csr_for_filter = user_items[user_ids] if filter_viewed else None

        all_user_ids, all_reco_ids, all_scores = ranker.rank(
            subject_ids=user_ids,
            k=k,
            filter_pairs_csr=ui_csr_for_filter,
            sorted_object_whitelist=sorted_item_ids_to_recommend,
            num_threads=0,
        )

        return all_user_ids, all_reco_ids, all_scores

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        similarity = self.weight
        if sorted_item_ids_to_recommend is not None:
            similarity = similarity[:, sorted_item_ids_to_recommend]

        all_target_ids = []
        all_reco_ids: tp.List[np.ndarray] = []
        all_scores: tp.List[np.ndarray] = []
        for target_id in tqdm(target_ids, disable=self.verbose == 0):
            reco_ids, reco_scores = self._recommend_for_item(
                similarity=similarity,
                target_id=target_id,
                k=k,
            )
            all_target_ids.extend([target_id] * len(reco_ids))
            all_reco_ids.append(reco_ids)
            all_scores.append(reco_scores)

        all_reco_ids_arr = np.concatenate(all_reco_ids)

        if sorted_item_ids_to_recommend is not None:
            all_reco_ids_arr = sorted_item_ids_to_recommend[all_reco_ids_arr]

        return all_target_ids, all_reco_ids_arr, np.concatenate(all_scores)

    @staticmethod
    def _recommend_for_item(
        similarity: np.ndarray,
        target_id: int,
        k: int,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        similar_item_scores = similarity[target_id]
        reco_ids, reco_scores = recommend_from_scores(scores=similar_item_scores, k=k)
        return reco_ids, reco_scores
