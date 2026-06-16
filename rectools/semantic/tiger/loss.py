import typing as tp

import torch
from torch.nn.functional import cross_entropy


def compute_tiger_loss(logits: tp.List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for TIGER single-target prediction.

    Parameters
    ----------
    logits : list of Tensor, each [B, codebook_size_d]
        One tensor per codebook level.
    labels : LongTensor [B, sid_len]
        Raw SID codes (0-based, no offset) for the target item.
    """
    total_loss = torch.tensor(0.0)
    sid_len = labels.shape[-1]

    for d in range(sid_len):
        total_loss = total_loss + cross_entropy(
            logits[d],  # [B, codebook_size_d]
            labels[:, d],  # [B]
            ignore_index=-100,
        )

    return total_loss / sid_len
