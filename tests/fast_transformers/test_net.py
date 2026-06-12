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

"""Tests for FlatSASRecNet network."""

# pylint: disable=redefined-outer-name

import pytest
import torch

from rectools.fast_transformers.net import FlatSASRecNet


@pytest.fixture()
def net() -> FlatSASRecNet:
    return FlatSASRecNet(n_items=30, n_factors=16, n_blocks=1, n_heads=2, session_max_len=8, dropout=0.0)


class TestFlatSASRecNet:
    def test_full_catalog_logits_shape(self, net: FlatSASRecNet) -> None:
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
        }
        logits = net(batch)
        assert logits.shape == (2, 5, 30)  # (B, L, n_items)

    def test_candidate_logits_shape(self, net: FlatSASRecNet) -> None:
        batch = {
            "x": torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]),
            "y": torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 7, 8]]),
            "negatives": torch.randint(1, 30, (2, 5, 3)),
        }
        logits = net(batch)
        assert logits.shape == (2, 5, 4)  # (B, L, 1 + n_neg)

    def test_encode_last_shape(self, net: FlatSASRecNet) -> None:
        x = torch.tensor([[0, 0, 1, 2, 3]])
        emb = net.encode_last(x)
        assert emb.shape == (1, 16)

    def test_determinism(self, net: FlatSASRecNet) -> None:
        """Same input produces identical output across two forward passes."""
        net.eval()
        x_a = torch.tensor([[0, 0, 0, 5, 10]])
        x_b = torch.tensor([[0, 0, 0, 5, 10]])
        with torch.no_grad():
            e_a = net.encode_last(x_a)
            e_b = net.encode_last(x_b)
        torch.testing.assert_close(e_a, e_b)
