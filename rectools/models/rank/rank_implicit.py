#  Copyright 2025 MTS (Mobile Telesystems)
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

"""Implicit ranker model."""

import typing as tp
import warnings

import implicit.cpu
import implicit.gpu
import numpy as np
from implicit.cpu.matrix_factorization_base import _filter_items_from_sparse_matrix as filter_items_from_sparse_matrix
from implicit.gpu import HAS_CUDA
from scipy import sparse

from rectools import InternalIds
from rectools.models.base import Scores
from rectools.models.rank.rank import Distance
from rectools.models.utils import convert_arr_to_implicit_gpu_matrix
from rectools.types import InternalIdsArray


class ImplicitRanker:
    """
    Ranker model which uses implicit library matrix factorization topk method.

    This ranker is suitable for the following cases of scores calculation:
    1. subject_embeddings.dot(objects_embeddings)
    2. subject_interactions.dot(item-item-similarities)

    Parameters
    ----------
    distance : Distance
        Distance metric.
    subjects_factors : np.ndarray | sparse.csr_matrix
        Array of subjects embeddings, shape (n_subjects, n_factors).
        For item-item similarity models subjects vectors from ui_csr are viewed as factors.
    objects_factors : np.ndarray
        Array with embeddings of all objects, shape (n_objects, n_factors).
        For item-item similarity models item similarity vectors are viewed as factors.
    num_threads : int, default 0
            Will be used as `num_threads` parameter for `implicit.cpu.topk.topk`. Omitted if use_gpu is True
    use_gpu : bool, default False
        If True `implicit.gpu.KnnQuery().topk` will be used instead of classic cpu version.
    """

    def __init__(
        self,
        distance: Distance,
        subjects_factors: tp.Union[np.ndarray, sparse.csr_matrix],
        objects_factors: np.ndarray,
        num_threads: int = 0,
        use_gpu: bool = False,
    ) -> None:
        if isinstance(subjects_factors, sparse.csr_matrix) and distance != Distance.DOT:
            raise ValueError("To use `sparse.csr_matrix` distance must be `Distance.DOT`")

        self.distance = distance
        self.subjects_factors: np.ndarray = subjects_factors.astype(np.float32)
        self.objects_factors: np.ndarray = objects_factors.astype(np.float32)
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.subjects_norms: np.ndarray
        if distance == Distance.COSINE:
            self.subjects_norms = self._calc_norms(self.subjects_factors, avoid_zeros=True)

        self.subjects_dots: np.ndarray
        if distance == Distance.EUCLIDEAN:
            self.subjects_dots = self._calc_dots(self.subjects_factors)

    def _get_neginf_score(self) -> float:
        # neginf_score computed according to implicit gpu FLT_FILTER_DISTANCE
        # https://github.com/benfred/implicit/blob/main/implicit/gpu/knn.cu#L36
        # we're comparing `scores <= neginf_score`
        return float(
            np.asarray(
                np.asarray(-np.finfo(np.float32).max, dtype=np.float32).view(np.uint32) - 1,
                dtype=np.uint32,
            ).view(np.float32)
        )

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

    def _rank_on_gpu(
        self,
        object_factors: np.ndarray,
        subject_factors: tp.Union[np.ndarray, sparse.csr_matrix],
        k: int,
        object_norms: tp.Optional[np.ndarray],
        filter_query_items: tp.Optional[tp.Union[sparse.csr_matrix, sparse.csr_array]],
    ) -> tp.Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        object_factors = convert_arr_to_implicit_gpu_matrix(object_factors)

        if isinstance(subject_factors, sparse.spmatrix):
            warnings.warn("Sparse subject factors converted to Dense matrix")
            subject_factors = subject_factors.todense()

        subject_factors = convert_arr_to_implicit_gpu_matrix(subject_factors)

        if object_norms is not None:
            if len(np.shape(object_norms)) == 1:
                object_norms = np.expand_dims(object_norms, axis=0)
            object_norms = convert_arr_to_implicit_gpu_matrix(object_norms)

        if filter_query_items is not None:
            if filter_query_items.count_nonzero() > 0:
                filter_query_items = implicit.gpu.COOMatrix(filter_query_items.tocoo())
            else:  # can't create `implicit.gpu.COOMatrix` for all zeroes
                filter_query_items = None

        ids, scores = implicit.gpu.KnnQuery().topk(  # pylint: disable=c-extension-no-member
            items=object_factors,
            m=subject_factors,
            k=k,
            item_norms=object_norms,
            query_filter=filter_query_items,
            item_filter=None,
        )

        scores = scores.astype(np.float64)
        return ids, scores

    def rank(  # pylint: disable=too-many-branches
        self,
        subject_ids: InternalIds,
        k: tp.Optional[int] = None,
        filter_pairs_csr: tp.Optional[sparse.csr_matrix] = None,
        sorted_object_whitelist: tp.Optional[InternalIdsArray] = None,
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        """Rank objects to proceed inference using implicit library topk cpu method.

        Parameters
        ----------
        subject_ids : csr_matrix
            Array of ids to recommend for.
        k : int, optional, default ``None``
            Derived number of recommendations for every subject id.
        filter_pairs_csr : sparse.csr_matrix, optional, default ``None``
            Subject-object interactions that should be filtered from recommendations.
            This is relevant for u2i case.
        sorted_object_whitelist : sparse.csr_matrix, optional, default ``None``
            Whitelist of object ids.
            If given, only these items will be used for recommendations.
            Otherwise all items from dataset will be used.

        Returns
        -------
        (InternalIds, InternalIds, Scores)
            Array of subject ids, array of recommended items, sorted by score descending and array of scores.
        """
        if filter_pairs_csr is not None and filter_pairs_csr.shape[0] != len(subject_ids):
            explanation = "Number of rows in `filter_pairs_csr` must be equal to `len(sublect_ids)`"
            raise ValueError(explanation)

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

        if k is None:
            k = object_factors.shape[0]

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

        use_gpu = self.use_gpu
        if use_gpu and not HAS_CUDA:
            warnings.warn("Forced rank() on CPU")
            use_gpu = False

        if use_gpu:  # pragma: no cover
            ids, scores = self._rank_on_gpu(
                object_factors=object_factors,
                subject_factors=subject_factors,
                k=real_k,
                object_norms=object_norms,
                filter_query_items=filter_query_items,
            )
        else:
            ids, scores = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
                items=object_factors,
                query=subject_factors,
                k=real_k,
                item_norms=object_norms,  # query norms for COSINE distance are applied afterwards
                filter_query_items=filter_query_items,  # queries x objects csr matrix for getting neginf scores
                filter_items=None,  # rectools doesn't support blacklist for now
                num_threads=self.num_threads,
            )

        if sorted_object_whitelist is not None:
            ids = sorted_object_whitelist[ids]

        # filter neginf from implicit scores and apply transformations to scores (for COSINE and EUCLIDEAN distances)
        all_target_ids, all_reco_ids, all_scores = self._process_implicit_scores(subject_ids, ids, scores)

        return all_target_ids, all_reco_ids, all_scores
