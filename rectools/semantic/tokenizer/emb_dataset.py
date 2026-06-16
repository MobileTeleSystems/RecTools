import typing as tp

import numpy as np
from torch.utils.data import Dataset


class EmbDataset(Dataset):
    """Dataset of item IDs paired with their embeddings.

    Parameters
    ----------
    item_ids : list of int
        Item identifiers.
    embeddings : np.ndarray
        Embedding matrix of shape ``(len(item_ids), embed_dim)``.
    """

    def __init__(
        self,
        item_ids: tp.List[int],
        embeddings: np.ndarray,
    ) -> None:
        if len(item_ids) != len(embeddings):
            raise ValueError(f"item_ids length ({len(item_ids)}) != embeddings length ({len(embeddings)})")
        self.item_ids: tp.List[int] = list(item_ids)
        self.embeddings: np.ndarray = np.asarray(embeddings)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, idx: tp.Union[int, slice]) -> tp.Dict[str, tp.Any]:
        if isinstance(idx, slice):
            return {
                "item_id": self.item_ids[idx],
                "embed": self.embeddings[idx],
            }
        return {
            "item_id": self.item_ids[idx],
            "embed": self.embeddings[idx],
        }

    def __len__(self) -> int:
        return len(self.embeddings)
