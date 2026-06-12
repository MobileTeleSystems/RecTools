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

"""Tests for GPU-friendly ranking metrics.

Tests verify:
  a) correctness on hand-crafted examples
  b) exact match with RecTools metrics (HitRate, NDCG, MRR)
"""

import numpy as np
import pandas as pd
import pytest
import torch

from rectools import Columns
from rectools.fast_transformers.metrics import compute_metrics, hitrate_at_k, mrr_at_k, ndcg_at_k
from rectools.metrics import MRR, NDCG, HitRate

# ---------------------------------------------------------------------------
# Helpers to bridge tensor metrics <-> RecTools DataFrame metrics
# ---------------------------------------------------------------------------


def _build_rectools_inputs(
    topk_ids: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert tensors to RecTools reco / interactions DataFrames."""
    B, K = topk_ids.shape
    users, items, ranks = [], [], []
    for u in range(B):
        for r in range(K):
            users.append(u)
            items.append(topk_ids[u, r].item())
            ranks.append(r + 1)
    reco = pd.DataFrame(
        {
            Columns.User: users,
            Columns.Item: items,
            Columns.Rank: ranks,
        }
    )
    interactions = pd.DataFrame(
        {
            Columns.User: list(range(B)),
            Columns.Item: targets.tolist(),
        }
    )
    return reco, interactions


# ---------------------------------------------------------------------------
# HitRate
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_all_hits(self) -> None:
        topk = torch.tensor([[5, 2, 3], [1, 7, 9]])
        targets = torch.tensor([5, 7])
        assert hitrate_at_k(topk, targets).item() == pytest.approx(1.0)

    def test_no_hits(self) -> None:
        topk = torch.tensor([[5, 2, 3], [1, 7, 9]])
        targets = torch.tensor([99, 88])
        assert hitrate_at_k(topk, targets).item() == pytest.approx(0.0)

    def test_partial_hits(self) -> None:
        topk = torch.tensor([[5, 2, 3], [1, 7, 9]])
        targets = torch.tensor([5, 88])
        assert hitrate_at_k(topk, targets).item() == pytest.approx(0.5)

    def test_hit_at_last_position(self) -> None:
        topk = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([3])
        assert hitrate_at_k(topk, targets).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NDCG
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        """Target at rank 1 => DCG = 1/log2(2) = 1.0, NDCG = 1/IDCG * 1.0."""
        topk = torch.tensor([[5]])
        targets = torch.tensor([5])
        # k=1: IDCG = 1/log2(2) = 1.0, DCG = 1.0, NDCG = 1.0
        assert ndcg_at_k(topk, targets).item() == pytest.approx(1.0)

    def test_no_hit(self) -> None:
        topk = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([99])
        assert ndcg_at_k(topk, targets).item() == pytest.approx(0.0)

    def test_hit_at_position_2(self) -> None:
        """Target at rank 2 out of k=3."""
        topk = torch.tensor([[1, 5, 3]])
        targets = torch.tensor([5])
        # DCG = 1/log2(3), IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        dcg = 1.0 / np.log2(3)
        idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3) + 1.0 / np.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(topk, targets).item() == pytest.approx(expected, abs=1e-6)

    def test_log_base_10(self) -> None:
        topk = torch.tensor([[5, 1]])
        targets = torch.tensor([5])
        dcg = 1.0 / np.log10(2)
        idcg = 1.0 / np.log10(2) + 1.0 / np.log10(3)
        expected = dcg / idcg
        assert ndcg_at_k(topk, targets, log_base=10).item() == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_hit_at_rank_1(self) -> None:
        topk = torch.tensor([[5, 2, 3]])
        targets = torch.tensor([5])
        assert mrr_at_k(topk, targets).item() == pytest.approx(1.0)

    def test_hit_at_rank_3(self) -> None:
        topk = torch.tensor([[1, 2, 5]])
        targets = torch.tensor([5])
        assert mrr_at_k(topk, targets).item() == pytest.approx(1.0 / 3)

    def test_no_hit(self) -> None:
        topk = torch.tensor([[1, 2, 3]])
        targets = torch.tensor([99])
        assert mrr_at_k(topk, targets).item() == pytest.approx(0.0)

    def test_multiple_users(self) -> None:
        topk = torch.tensor([[5, 2, 3], [1, 2, 7]])
        targets = torch.tensor([5, 7])
        # user 0: 1/1, user 1: 1/3
        expected = (1.0 + 1.0 / 3) / 2
        assert mrr_at_k(topk, targets).item() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_default_k(self) -> None:
        topk = torch.tensor([[5, 2], [1, 7]])
        targets = torch.tensor([5, 99])
        result = compute_metrics(topk, targets)
        assert "HR@2" in result
        assert "NDCG@2" in result
        assert "MRR@2" in result

    def test_multiple_ks(self) -> None:
        topk = torch.tensor([[5, 2, 3, 4], [1, 7, 9, 8]])
        targets = torch.tensor([5, 9])
        result = compute_metrics(topk, targets, ks=[1, 2, 4])
        assert "HR@1" in result and "HR@2" in result and "HR@4" in result

    def test_k_exceeds_width_raises(self) -> None:
        topk = torch.tensor([[5, 2]])
        targets = torch.tensor([5])
        with pytest.raises(ValueError, match="exceeds"):
            compute_metrics(topk, targets, ks=[5])


# ---------------------------------------------------------------------------
# Cross-validation with RecTools metrics
# ---------------------------------------------------------------------------


class TestMatchRecTools:
    """Verify that our GPU metrics produce identical results to RecTools."""

    @pytest.fixture()
    def scenario_mixed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """4 users, k=5. Mix of hits at various ranks and misses."""
        topk = torch.tensor(
            [
                [10, 20, 30, 40, 50],  # target=30, hit at rank 3
                [11, 21, 31, 41, 51],  # target=99, no hit
                [12, 22, 32, 42, 52],  # target=12, hit at rank 1
                [13, 23, 33, 43, 53],  # target=53, hit at rank 5
            ]
        )
        targets = torch.tensor([30, 99, 12, 53])
        return topk, targets

    @pytest.fixture()
    def scenario_all_hit(self) -> tuple[torch.Tensor, torch.Tensor]:
        topk = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        targets = torch.tensor([2, 4, 9])
        return topk, targets

    @pytest.fixture()
    def scenario_no_hit(self) -> tuple[torch.Tensor, torch.Tensor]:
        topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
        targets = torch.tensor([99, 88])
        return topk, targets

    @pytest.mark.parametrize("fixture_name", ["scenario_mixed", "scenario_all_hit", "scenario_no_hit"])
    def test_hitrate_matches_rectools(self, fixture_name: str, request: pytest.FixtureRequest) -> None:
        topk, targets = request.getfixturevalue(fixture_name)
        k = topk.shape[1]
        reco, interactions = _build_rectools_inputs(topk, targets)

        ours = hitrate_at_k(topk, targets).item()
        theirs = HitRate(k=k).calc(reco, interactions)
        assert ours == pytest.approx(theirs, abs=1e-7), f"HR@{k}: ours={ours}, rectools={theirs}"

    @pytest.mark.parametrize("fixture_name", ["scenario_mixed", "scenario_all_hit", "scenario_no_hit"])
    def test_ndcg_matches_rectools(self, fixture_name: str, request: pytest.FixtureRequest) -> None:
        topk, targets = request.getfixturevalue(fixture_name)
        k = topk.shape[1]
        reco, interactions = _build_rectools_inputs(topk, targets)

        ours = ndcg_at_k(topk, targets).item()
        theirs = NDCG(k=k).calc(reco, interactions)
        assert ours == pytest.approx(theirs, abs=1e-7), f"NDCG@{k}: ours={ours}, rectools={theirs}"

    @pytest.mark.parametrize("fixture_name", ["scenario_mixed", "scenario_all_hit", "scenario_no_hit"])
    def test_mrr_matches_rectools(self, fixture_name: str, request: pytest.FixtureRequest) -> None:
        topk, targets = request.getfixturevalue(fixture_name)
        k = topk.shape[1]
        reco, interactions = _build_rectools_inputs(topk, targets)

        ours = mrr_at_k(topk, targets).item()
        theirs = MRR(k=k).calc(reco, interactions)
        assert ours == pytest.approx(theirs, abs=1e-7), f"MRR@{k}: ours={ours}, rectools={theirs}"

    @pytest.mark.parametrize("fixture_name", ["scenario_mixed", "scenario_all_hit", "scenario_no_hit"])
    def test_all_ks_match_rectools(self, fixture_name: str, request: pytest.FixtureRequest) -> None:
        """Test at multiple K values to make sure slicing is correct."""
        topk, targets = request.getfixturevalue(fixture_name)
        k_max = topk.shape[1]
        ks = list(range(1, k_max + 1))

        reco, interactions = _build_rectools_inputs(topk, targets)

        ours = compute_metrics(topk, targets, ks=ks)
        for k in ks:
            rt_hr = HitRate(k=k).calc(reco, interactions)
            rt_ndcg = NDCG(k=k).calc(reco, interactions)
            rt_mrr = MRR(k=k).calc(reco, interactions)
            assert ours[f"HR@{k}"] == pytest.approx(rt_hr, abs=1e-7), f"HR@{k}"
            assert ours[f"NDCG@{k}"] == pytest.approx(rt_ndcg, abs=1e-7), f"NDCG@{k}"
            assert ours[f"MRR@{k}"] == pytest.approx(rt_mrr, abs=1e-7), f"MRR@{k}"

    def test_random_large_batch(self) -> None:
        """Randomized test with 500 users, k=20."""
        torch.manual_seed(42)
        B, K = 500, 20
        n_items = 1000
        topk = torch.randint(1, n_items, (B, K))
        targets = torch.randint(1, n_items, (B,))
        # Ensure some hits by placing target at random positions
        for i in range(0, B, 3):
            pos = int(torch.randint(0, K, (1,)).item())
            topk[i, pos] = targets[i]

        reco, interactions = _build_rectools_inputs(topk, targets)

        for k in [1, 5, 10, 20]:
            our_hr = hitrate_at_k(topk[:, :k], targets).item()
            our_ndcg = ndcg_at_k(topk[:, :k], targets).item()
            our_mrr = mrr_at_k(topk[:, :k], targets).item()

            rt_hr = HitRate(k=k).calc(reco, interactions)
            rt_ndcg = NDCG(k=k).calc(reco, interactions)
            rt_mrr = MRR(k=k).calc(reco, interactions)

            assert our_hr == pytest.approx(rt_hr, abs=1e-6), f"HR@{k}"
            assert our_ndcg == pytest.approx(rt_ndcg, abs=1e-6), f"NDCG@{k}"
            assert our_mrr == pytest.approx(rt_mrr, abs=1e-6), f"MRR@{k}"
