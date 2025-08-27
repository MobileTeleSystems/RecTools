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

"""Torch ranker model."""

import typing as tp

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

from rectools import InternalIds
from rectools.models.base import Scores
from rectools.models.rank.rank import Distance
from rectools.types import InternalIdsArray


class TorchRanker:
    """
    Ranker model based on torch.

    This ranker is suitable for the following cases of scores calculation:
    1. subject_embeddings.dot(objects_embeddings)
    2. subject_interactions.dot(item-item-similarities)

    Parameters
    ----------
    distance : Distance
        Distance metric.
    device: torch.device | str
        Device to calculate on.
    batch_size: int, default 128
        Batch size for scores calculation.
    subjects_factors : np.ndarray | sparse.csr_matrix | torch.Tensor
        Array of subjects embeddings, shape (n_subjects, n_factors).
        For item-item similarity models subjects vectors from ui_csr are viewed as factors.
    objects_factors : np.ndarray | torch.Tensor
        Array with embeddings of all objects, shape (n_objects, n_factors).
        For item-item similarity models item similarity vectors are viewed as factors.
    dtype: torch.dtype, optional, default `torch.float32`
        dtype to convert non-torch tensors to.
        Conversion is skipped if provided dtype is ``None``.
    """

    epsilon_cosine_dist: torch.Tensor = torch.tensor([1e-8])

    def __init__(
        self,
        distance: Distance,
        device: tp.Union[torch.device, str],
        subjects_factors: tp.Union[np.ndarray, sparse.csr_matrix, torch.Tensor],
        objects_factors: tp.Union[np.ndarray, torch.Tensor],
        batch_size: int = 128,
        dtype: tp.Optional[torch.dtype] = torch.float32,
    ):
        self.dtype = dtype
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.distance = distance
        self._scorer, self._higher_is_better = self._get_scorer(distance)

        self.subjects_factors = self._normalize_tensor(subjects_factors)
        self.objects_factors = self._normalize_tensor(objects_factors)

    def rank(
        self,
        subject_ids: InternalIds,
        k: tp.Optional[int] = None,
        filter_pairs_csr: tp.Optional[sparse.csr_matrix] = None,
        sorted_object_whitelist: tp.Optional[InternalIdsArray] = None,
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        """Rank objects to proceed inference using implicit library topk cpu method.

        Parameters
        ----------
        subject_ids : InternalIds
            Array of ids to recommend for.
        k : int, optional, default ``None``
            Derived number of recommendations for every subject id.
            Return all recs if None.
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
        # pylint: disable=too-many-locals
        if filter_pairs_csr is not None and filter_pairs_csr.shape[0] != len(subject_ids):
            explanation = "Number of rows in `filter_pairs_csr` must be equal to `len(sublect_ids)`"
            raise ValueError(explanation)

        if sorted_object_whitelist is None:
            sorted_object_whitelist = np.arange(self.objects_factors.shape[0])

        subject_ids = np.asarray(subject_ids)

        if k is None:
            k = len(sorted_object_whitelist)

        user_embs = self.subjects_factors[subject_ids]
        item_embs = self.objects_factors[sorted_object_whitelist]

        user_embs_dataset = TensorDataset(torch.arange(user_embs.shape[0]), user_embs)
        dataloader = DataLoader(user_embs_dataset, batch_size=self.batch_size, shuffle=False)
        mask_values = float("-inf")
        all_top_scores_list = []
        all_top_inds_list = []
        all_target_inds_list = []
        with torch.no_grad():
            for (
                cur_user_emb_inds,
                cur_user_embs,
            ) in dataloader:
                scores = self._scorer(
                    cur_user_embs.to(self.device),
                    item_embs.to(self.device),
                )

                if filter_pairs_csr is not None:
                    # Convert cur_user_emb_inds to numpy to avoid
                    # AttributeError: 'torch.dtype' object has no attribute 'kind'
                    cur_user_filter_pairs_csr = filter_pairs_csr[cur_user_emb_inds.cpu().numpy()]
                    whitelisted_filter_matrix = cur_user_filter_pairs_csr.toarray()[:, sorted_object_whitelist]
                    mask = torch.from_numpy(whitelisted_filter_matrix).to(scores.device) != 0
                    scores = torch.masked_fill(scores, mask, mask_values)

                top_scores, top_inds = torch.topk(
                    scores,
                    k=min(k, scores.shape[1]),
                    dim=1,
                    sorted=True,
                    largest=self._higher_is_better,
                )
                all_top_scores_list.append(top_scores.cpu().numpy())
                all_top_inds_list.append(top_inds.cpu().numpy())
                all_target_inds_list.append(cur_user_emb_inds.cpu().numpy())

        all_top_scores = np.concatenate(all_top_scores_list, axis=0)
        all_top_inds = np.concatenate(all_top_inds_list, axis=0)
        all_target_inds = np.concatenate(all_target_inds_list, axis=0)

        # flatten and convert inds back to input ids
        all_scores = all_top_scores.flatten()
        all_target_ids = subject_ids[all_target_inds].repeat(all_top_inds.shape[1])
        all_reco_ids = sorted_object_whitelist[all_top_inds].flatten()

        # filter masked items if they appeared at top
        if filter_pairs_csr is not None:
            mask = all_scores > mask_values
            all_scores = all_scores[mask]
            all_target_ids = all_target_ids[mask]
            all_reco_ids = all_reco_ids[mask]

        return (
            all_target_ids,
            all_reco_ids,
            all_scores,
        )

    def _get_scorer(
        self, distance: Distance
    ) -> tp.Tuple[tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], bool]:
        """Return scorer and higher_is_better flag"""
        if distance == Distance.DOT:
            return self._dot_score, True

        if distance == Distance.COSINE:
            return self._cosine_score, True

        if distance == Distance.EUCLIDEAN:
            return self._euclid_score, False

        raise NotImplementedError(f"distance {distance} is not supported")  # pragma: no cover

    def _euclid_score(self, user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        return torch.cdist(user_embs.unsqueeze(0), item_embs.unsqueeze(0)).squeeze(0)

    def _cosine_score(self, user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        user_embs = user_embs / torch.max(
            torch.norm(user_embs, p=2, dim=1).unsqueeze(dim=1), self.epsilon_cosine_dist.to(user_embs)
        )
        item_embs = item_embs / torch.max(
            torch.norm(item_embs, p=2, dim=1).unsqueeze(dim=1), self.epsilon_cosine_dist.to(user_embs)
        )

        return user_embs @ item_embs.T

    def _dot_score(self, user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        return user_embs @ item_embs.T

    def _normalize_tensor(
        self,
        tensor: tp.Union[np.ndarray, sparse.csr_matrix, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(tensor, sparse.csr_matrix):
            tensor = tensor.toarray()

        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)

        if self.dtype is not None:
            tensor = tensor.to(self.dtype)

        return tensor
