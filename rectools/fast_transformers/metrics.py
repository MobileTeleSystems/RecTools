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

"""GPU-friendly ranking metrics for leave-one-out evaluation.

All functions operate on PyTorch tensors and stay on the original device
(CPU or CUDA), avoiding numpy/pandas roundtrips.  Results are numerically
identical to the corresponding RecTools metrics with default settings:

- :class:`rectools.metrics.HitRate` (k=K)
- :class:`rectools.metrics.NDCG` (k=K, log_base=2, divide_by_achievable=False)
- :class:`rectools.metrics.MRR` (k=K)

These functions assume **leave-one-out** evaluation: each user has exactly
one ground-truth target item.
"""

import math
import typing as tp

import torch


@torch.no_grad()
def hitrate_at_k(
    topk_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Hit Rate @ K (leave-one-out).

    Parameters
    ----------
    topk_ids : LongTensor (B, K)
        Top-K predicted item IDs per user.
    targets : LongTensor (B,)
        Ground-truth item ID per user.

    Returns
    -------
    Tensor (scalar)
        Mean hit rate across users.
    """
    hits = (topk_ids == targets.unsqueeze(1)).any(dim=1)
    return hits.float().mean()


@torch.no_grad()
def ndcg_at_k(
    topk_ids: torch.Tensor,
    targets: torch.Tensor,
    log_base: int = 2,
) -> torch.Tensor:
    """NDCG @ K (leave-one-out, divide_by_achievable=False).

    Matches :class:`rectools.metrics.NDCG` with default parameters.
    IDCG is computed as the maximum possible DCG when all K positions are
    relevant (constant across users), which is the RecTools default.

    Parameters
    ----------
    topk_ids : LongTensor (B, K)
        Top-K predicted item IDs per user.
    targets : LongTensor (B,)
        Ground-truth item ID per user.
    log_base : int, default 2
        Logarithm base for the discount factor.

    Returns
    -------
    Tensor (scalar)
        Mean NDCG across users.
    """
    k = topk_ids.shape[1]
    hits = (topk_ids == targets.unsqueeze(1)).float()  # (B, K)
    ranks = torch.arange(1, k + 1, device=topk_ids.device, dtype=torch.float)
    discounts = 1.0 / torch.log(ranks + 1) * (1.0 / math.log(log_base))
    dcg = (hits * discounts.unsqueeze(0)).sum(dim=1)  # (B,)
    idcg = discounts.sum()
    return (dcg / idcg).mean()


@torch.no_grad()
def mrr_at_k(
    topk_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """MRR @ K (leave-one-out).

    Parameters
    ----------
    topk_ids : LongTensor (B, K)
        Top-K predicted item IDs per user.
    targets : LongTensor (B,)
        Ground-truth item ID per user.

    Returns
    -------
    Tensor (scalar)
        Mean reciprocal rank across users.
    """
    hits = topk_ids == targets.unsqueeze(1)  # (B, K)
    # For each user find the rank of the first hit (1-based), 0 if no hit
    has_hit = hits.any(dim=1)
    # argmax returns the first True index
    first_hit_rank = hits.float().argmax(dim=1) + 1  # (B,)
    rr = torch.zeros_like(first_hit_rank, dtype=torch.float)
    rr[has_hit] = 1.0 / first_hit_rank[has_hit].float()
    return rr.mean()


@torch.no_grad()
def compute_metrics(
    topk_ids: torch.Tensor,
    targets: torch.Tensor,
    ks: tp.Optional[tp.List[int]] = None,
    log_base: int = 2,
) -> tp.Dict[str, float]:
    """Compute HR, NDCG, MRR at multiple K values.

    Parameters
    ----------
    topk_ids : LongTensor (B, K_max)
        Top-K_max predicted item IDs per user.
    targets : LongTensor (B,)
        Ground-truth item ID per user.
    ks : list of int, optional
        K values to evaluate. Defaults to ``[K_max]``.
    log_base : int, default 2
        Logarithm base for NDCG discount.

    Returns
    -------
    dict
        Keys like ``"HR@10"``, ``"NDCG@10"``, ``"MRR@10"``.
    """
    k_max = topk_ids.shape[1]
    if ks is None:
        ks = [k_max]
    results: tp.Dict[str, float] = {}
    for k in ks:
        if k > k_max:
            raise ValueError(f"k={k} exceeds topk_ids width {k_max}")
        top = topk_ids[:, :k]
        results[f"HR@{k}"] = hitrate_at_k(top, targets).item()
        results[f"NDCG@{k}"] = ndcg_at_k(top, targets, log_base=log_base).item()
        results[f"MRR@{k}"] = mrr_at_k(top, targets).item()
    return results
