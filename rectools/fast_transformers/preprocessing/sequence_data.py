#  Copyright 2026 MTS (Mobile Telesystems)
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

"""Vectorized sequence building for transformer recommender training.

All operations use pure PyTorch tensor ops, avoiding pandas/numpy overhead.
On GPU this gives ~30x speedup over pandas-based preprocessing on ML-20M.
"""

import logging
import typing as tp

import torch
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


def build_sequences(  # pylint: disable=too-many-locals
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    timestamps: torch.Tensor,
    max_len: int,
    min_interactions: int = 2,
    device: tp.Optional[str] = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build left-padded input/target sequence pairs from interaction data.

    Groups interactions by user, sorts by timestamp, and produces
    ``(x, y)`` pairs where ``y[i, j] = x[i, j+1]`` (next-item prediction).
    Item IDs are remapped to contiguous internal indices ``1..N``
    (0 is reserved for padding).

    Parameters
    ----------
    user_ids : LongTensor (N,)
        User ID for each interaction.
    item_ids : LongTensor (N,)
        Item ID for each interaction.
    timestamps : LongTensor (N,)
        Timestamp for each interaction (any monotonic int64 values).
    max_len : int
        Maximum sequence length.
    min_interactions : int, default 2
        Minimum interactions per user to be included.
    device : str, optional
        Device for computation.  Defaults to the device of ``user_ids``
        (pass ``"cuda"`` explicitly for GPU acceleration).

    Returns
    -------
    x : LongTensor (U, max_len)
        Left-padded input sequences (0 = padding).
    y : LongTensor (U, max_len)
        Left-padded target sequences.
    unique_items : LongTensor
        External item IDs that appear in the data.
    result_users : LongTensor
        External user IDs that passed the ``min_interactions`` filter.

    Examples
    --------
    >>> users = torch.tensor([0, 0, 0, 1, 1, 1])
    >>> items = torch.tensor([10, 20, 30, 40, 50, 60])
    >>> times = torch.tensor([1, 2, 3, 1, 2, 3])
    >>> x, y, uniq_items, uniq_users = build_sequences(users, items, times, max_len=4)
    >>> x.shape[1]
    4
    """
    if device is None:
        device = str(user_ids.device)
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)
    timestamps = timestamps.to(device)

    unique_items, item_inv = torch.unique(item_ids, return_inverse=True)
    internal_items = item_inv + 1

    unique_users, user_inv = torch.unique(user_ids, return_inverse=True)

    order1 = torch.argsort(timestamps, stable=True)
    order2 = torch.argsort(user_inv[order1], stable=True)
    order = order1[order2]

    sorted_user_inv = user_inv[order]
    sorted_items = internal_items[order]

    changes = torch.where(sorted_user_inv[1:] != sorted_user_inv[:-1])[0] + 1
    starts = torch.cat([torch.tensor([0], device=device), changes])
    ends = torch.cat([changes, torch.tensor([len(sorted_user_inv)], device=device)])
    lengths = ends - starts

    mask = lengths >= min_interactions
    starts = starts[mask]
    ends = ends[mask]
    lengths = lengths[mask]
    n_users = len(starts)

    capped_lens = torch.clamp(lengths, max=max_len + 1)

    effective_lens = torch.clamp(capped_lens - 1, min=0)
    total_elements = effective_lens.sum().item()

    x = torch.zeros(n_users, max_len, dtype=torch.long, device=device)
    y = torch.zeros(n_users, max_len, dtype=torch.long, device=device)

    if total_elements > 0:
        user_indices = torch.repeat_interleave(torch.arange(n_users, device=device), effective_lens)
        cumsum = effective_lens.cumsum(0)
        offsets = torch.arange(total_elements, device=device) - torch.repeat_interleave(
            cumsum - effective_lens, effective_lens
        )

        x_src = torch.repeat_interleave(ends - capped_lens, effective_lens) + offsets
        y_src = x_src + 1
        col_indices = max_len - torch.repeat_interleave(effective_lens, effective_lens) + offsets

        x[user_indices, col_indices] = sorted_items[x_src]
        y[user_indices, col_indices] = sorted_items[y_src]

    valid_user_indices = torch.where(mask)[0]
    result_users = unique_users[valid_user_indices] if len(valid_user_indices) < len(unique_users) else unique_users

    return x, y, unique_items, result_users


def align_embeddings(
    pretrained: torch.Tensor,
    unique_items: torch.Tensor,
    n_items: int,
) -> torch.Tensor:
    """Reorder a pretrained embedding matrix to match internal item ID order.

    Internal IDs are contiguous ``1..n_items`` as produced by
    :func:`build_sequences`.  Index 0 is padding (zeros).

    Parameters
    ----------
    pretrained : Tensor (V, D) or (V, K, D)
        Pretrained embeddings indexed by external item ID.
    unique_items : LongTensor
        External item IDs returned by :func:`build_sequences`.
    n_items : int
        Number of unique items.

    Returns
    -------
    Tensor (n_items + 1, D) or (n_items + 1, K, D)
        Aligned embeddings with padding row at index 0.
    """
    device = pretrained.device
    dtype = pretrained.dtype
    idx = unique_items.long().to(device)
    valid = (idx >= 0) & (idx < pretrained.shape[0])

    n_missing = int((~valid).sum().item())
    if n_missing > 0:
        logger.warning(
            "align_embeddings: %d out of %d items have no pretrained embedding "
            "(index out of range). They will be zero-filled (same as padding).",
            n_missing,
            len(idx),
        )

    if pretrained.ndim == 2:
        aligned = torch.zeros(n_items + 1, pretrained.shape[1], dtype=dtype, device=device)
    else:
        aligned = torch.zeros(n_items + 1, pretrained.shape[1], pretrained.shape[2], dtype=dtype, device=device)

    aligned[1:][valid] = pretrained[idx[valid]]
    return aligned


class SequenceBatchDataset(TorchDataset):
    """Lightweight Dataset wrapping prebuilt (x, y) sequence tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform: tp.Optional[tp.Callable] = None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tp.Dict[str, torch.Tensor]:
        batch = {"x": self.x[idx], "y": self.y[idx]}
        if self.transform:
            batch = self.transform(batch)
        return batch
