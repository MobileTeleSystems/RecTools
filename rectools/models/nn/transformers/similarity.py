import typing as tp

import torch
import torch.nn as nn

from rectools.models.rank import Distance


class SimilarityModuleBase(nn.Module):

    def __init__(self, loss_type: str, *args: tp.Any, **kwargs: tp.Any) -> None:
        self.loss_type = loss_type

    def forward(self, session_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SimilarityModuleDistance(SimilarityModuleBase):

    dist_available: tp.List[Distance] = [Distance.DOT, Distance.COSINE]
    epsilon_cosine_dist: float = 1e-8

    def __init__(self, loss_type: str, dist: Distance = Distance.DOT) -> None:
        if dist not in self.dist_available:
            raise ValueError("`dist` can only be either `Distance.DOT` or `Distance.COSINE`.")

        self.dist = dist
        self.loss_type = loss_type

    def _get_embeddings_norm(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings / (torch.norm(embeddings, p=2, dim=1).unsqueeze(dim=1) + self.epsilon_cosine_dist)
        return embeddings

    def _calc_custom_score(self, session_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:  # TODO
        raise ValueError(f"loss {self.loss} is not supported in `DistanceSimilarity`")  # pragma: no cover

    def forward(self, session_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        if self.dist == Distance.COSINE:
            session_embs = self._get_embeddings_norm(session_embs)
            item_embs = self._get_embeddings_norm(item_embs)

        if self.loss_type == "softmax":
            scores = session_embs @ item_embs.T
        elif self.loss_type in ["BCE", "gBCE"]:
            scores = (item_embs @ session_embs.unsqueeze(-1)).squeeze(-1)
        else:  # TODO: think about it
            scores = self._calc_custom_score(session_embs, item_embs)  # pragma: no cover
        return scores
