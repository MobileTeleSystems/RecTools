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

"""Tests for UniSRecNet network."""

# pylint: disable=redefined-outer-name

import pytest
import torch

from rectools.fast_transformers.unisrec.net import UniSRecNet


@pytest.fixture()
def pretrained_emb() -> torch.Tensor:
    """Fake pretrained embeddings: (31, 64) — 30 items + 1 padding."""
    torch.manual_seed(0)
    emb = torch.randn(31, 64)
    emb[0] = 0.0  # padding
    return emb


@pytest.fixture()
def net(pretrained_emb: torch.Tensor) -> UniSRecNet:
    return UniSRecNet(
        n_items=30,
        pretrained_embeddings=pretrained_emb,
        n_factors=16,
        projection_hidden=32,
        n_blocks=1,
        n_heads=2,
        session_max_len=8,
        dropout=0.0,
        adaptor_dropout=0.0,
    )


class TestUniSRecNetShapes:
    def test_forward_shape(self, net: UniSRecNet) -> None:
        x = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
        h = net(x)
        assert h.shape == (2, 5, 16)

    def test_encode_last_shape(self, net: UniSRecNet) -> None:
        x = torch.tensor([[0, 0, 1, 2, 3]])
        emb = net.encode_last(x)
        assert emb.shape == (1, 16)

    def test_project_all_shape(self, net: UniSRecNet) -> None:
        proj = net.project_all()
        assert proj.shape == (31, 16)  # n_items + 1 (with padding)


class TestUniSRecNetAdaptor:
    def test_pca_no_ffn(self, pretrained_emb: torch.Tensor) -> None:
        net = UniSRecNet(
            n_items=30,
            pretrained_embeddings=pretrained_emb,
            n_factors=16,
            n_blocks=1,
            n_heads=2,
            session_max_len=8,
            adaptor_type="pca",
            use_adaptor_ffn=False,
        )
        proj = net.project_all()
        assert proj.shape == (31, 16)
        assert net.head is None

    def test_multi_variant(self) -> None:
        torch.manual_seed(0)
        emb = torch.randn(31, 3, 64)  # 3 variants
        emb[0] = 0.0
        net = UniSRecNet(
            n_items=30,
            pretrained_embeddings=emb,
            n_factors=16,
            projection_hidden=32,
            n_blocks=1,
            n_heads=2,
            session_max_len=8,
        )
        assert net.n_variants == 3
        x = torch.tensor([[0, 0, 1, 2, 3]])
        h = net(x)
        assert h.shape == (1, 5, 16)


class TestPaddingInvariance:
    def test_determinism_and_padding_masking(self, net: UniSRecNet) -> None:
        """Same input produces identical output; padding positions are zeroed."""
        net.eval()
        x_a = torch.tensor([[0, 0, 0, 5, 10]])
        x_b = torch.tensor([[0, 0, 0, 5, 10]])
        with torch.no_grad():
            e_a = net.encode_last(x_a)
            e_b = net.encode_last(x_b)
            h_a = net(x_a)
        torch.testing.assert_close(e_a, e_b)
        # Padding positions (first 3 columns) should be zeroed in full output
        torch.testing.assert_close(h_a[:, :3, :], torch.zeros_like(h_a[:, :3, :]))
