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

import typing as tp

import numpy as np
import torch
from scipy import sparse

from rectools.models.base import InternalRecoTriplet
from rectools.models.rank import Distance, TorchRanker
from rectools.types import InternalIdsArray


class SimilarityModuleBase(torch.nn.Module):
    """Base class for similarity module."""

    def _get_full_catalog_logits(self, session_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_pos_neg_logits(
        self, session_embs: torch.Tensor, item_embs: torch.Tensor, candidate_item_ids: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(
        self,
        session_embs: torch.Tensor,
        item_embs: torch.Tensor,
        candidate_item_ids: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to get logits."""
        raise NotImplementedError()

    def _recommend_u2i(
        self,
        user_embs: torch.Tensor,
        item_embs: torch.Tensor,
        user_ids: InternalIdsArray,
        k: int,
        sorted_item_ids_to_recommend: InternalIdsArray,
        ui_csr_for_filter: tp.Optional[sparse.csr_matrix],
    ) -> InternalRecoTriplet:
        """Recommend to users."""
        raise NotImplementedError()


class DistanceSimilarityModule(SimilarityModuleBase):
    """Distance similarity module."""

    dist_available: tp.List[str] = [Distance.DOT, Distance.COSINE]
    epsilon_cosine_dist: torch.Tensor = torch.tensor([1e-8])

    def __init__(self, distance: str = "dot") -> None:
        super().__init__()
        if distance not in self.dist_available:
            raise ValueError("`dist` can only be either `dot` or `cosine`.")

        self.distance = Distance(distance)

    def _get_full_catalog_logits(self, session_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        logits = session_embs @ item_embs.T
        return logits

    def _get_pos_neg_logits(
        self, session_embs: torch.Tensor, item_embs: torch.Tensor, candidate_item_ids: torch.Tensor
    ) -> torch.Tensor:
        # [batch_size, session_max_len, len(candidate_item_ids), n_factors]
        pos_neg_embs = item_embs[candidate_item_ids]
        # [batch_size, session_max_len,len(item_ids)]
        logits = (pos_neg_embs @ session_embs.unsqueeze(-1)).squeeze(-1)
        return logits

    def _get_embeddings_norm(self, embeddings: torch.Tensor) -> torch.Tensor:
        embedding_norm = torch.norm(embeddings, p=2, dim=1).unsqueeze(dim=1)
        embeddings = embeddings / torch.max(embedding_norm, self.epsilon_cosine_dist.to(embeddings))
        return embeddings

    def forward(
        self,
        session_embs: torch.Tensor,
        item_embs: torch.Tensor,
        candidate_item_ids: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to get logits."""
        if self.distance == Distance.COSINE:
            session_embs = self._get_embeddings_norm(session_embs)
            item_embs = self._get_embeddings_norm(item_embs)

        if candidate_item_ids is None:
            return self._get_full_catalog_logits(session_embs, item_embs)
        return self._get_pos_neg_logits(session_embs, item_embs, candidate_item_ids)

    def _recommend_u2i(
        self,
        user_embs: torch.Tensor,
        item_embs: torch.Tensor,
        user_ids: InternalIdsArray,
        k: int,
        sorted_item_ids_to_recommend: InternalIdsArray,
        ui_csr_for_filter: tp.Optional[sparse.csr_matrix],
    ) -> InternalRecoTriplet:
        """Recommend to users."""
        ranker = TorchRanker(
            distance=self.distance,
            device=item_embs.device,
            subjects_factors=user_embs[user_ids],
            objects_factors=item_embs,
        )
        user_ids_indices, all_reco_ids, all_scores = ranker.rank(
            subject_ids=np.arange(len(user_ids)),  # n_rec_users
            k=k,
            filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + n_item_extra_tokens]
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model_internal
        )
        all_user_ids = user_ids[user_ids_indices]
        return all_user_ids, all_reco_ids, all_scores
