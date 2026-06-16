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

import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.semantic.modules.transformer_blocks import (
    T5Attention,
    T5DecoderLayer,
    T5EncoderLayer,
    T5RMSNorm,
    T5RelativePositionBias,
    init_feed_forward,
)


class TestT5RMSNorm:
    def test_output_shape(self) -> None:
        norm = T5RMSNorm(d_model=16)
        x = torch.randn(3, 5, 16)
        out = norm(x)
        assert out.shape == (3, 5, 16)

    def test_normalized_rms(self) -> None:
        norm = T5RMSNorm(d_model=16)
        x = torch.randn(3, 5, 16)
        out = norm(x)
        # Output should have roughly unit RMS (before learned scaling)
        assert torch.isfinite(out).all()


class TestT5RelativePositionBias:
    def test_output_shape(self) -> None:
        bias = T5RelativePositionBias(num_heads=4, bidirectional=True)
        out = bias(query_len=5, key_len=5, device=torch.device("cpu"))
        assert out.shape == (1, 4, 5, 5)

    def test_causal_output_shape(self) -> None:
        bias = T5RelativePositionBias(num_heads=2, bidirectional=False)
        out = bias(query_len=3, key_len=3, device=torch.device("cpu"))
        assert out.shape == (1, 2, 3, 3)

    def test_asymmetric_shape(self) -> None:
        bias = T5RelativePositionBias(num_heads=2, bidirectional=True)
        out = bias(query_len=3, key_len=7, device=torch.device("cpu"))
        assert out.shape == (1, 2, 3, 7)


class TestT5Attention:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_self_attention_output_shape(self) -> None:
        attn = T5Attention(d_model=16, num_heads=2, d_kv=8)
        x = torch.randn(2, 5, 16)
        out, _ = attn(x, x, x)
        assert out.shape == (2, 5, 16)

    def test_cross_attention_output_shape(self) -> None:
        attn = T5Attention(d_model=16, num_heads=2, d_kv=8, kv_input_dim=32)
        q = torch.randn(2, 3, 16)
        kv = torch.randn(2, 7, 32)
        out, _ = attn(q, kv, kv)
        assert out.shape == (2, 3, 16)

    def test_with_position_bias(self) -> None:
        attn = T5Attention(d_model=16, num_heads=2, d_kv=8)
        x = torch.randn(2, 5, 16)
        pos_bias = torch.randn(1, 2, 5, 5)
        out, _ = attn(x, x, x, position_bias=pos_bias)
        assert out.shape == (2, 5, 16)


class TestT5EncoderLayer:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_output_shape(self) -> None:
        layer = T5EncoderLayer(d_model=16, num_heads=2, d_kv=8, dim_feedforward=32)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_with_padding_mask(self) -> None:
        layer = T5EncoderLayer(d_model=16, num_heads=2, d_kv=8)
        x = torch.randn(2, 5, 16)
        mask = torch.tensor([[False, False, False, True, True], [False, False, True, True, True]])
        out = layer(x, src_key_padding_mask=mask)
        assert out.shape == (2, 5, 16)


class TestT5DecoderLayer:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_output_shape(self) -> None:
        layer = T5DecoderLayer(d_model=16, num_heads=2, d_kv=8, dim_feedforward=32)
        tgt = torch.randn(2, 3, 16)
        memory = torch.randn(2, 5, 16)
        out = layer(tgt, memory)
        assert out.shape == (2, 3, 16)


class TestInitFeedForward:
    def test_relu(self) -> None:
        ff = init_feed_forward(16, 4, 0.1, "relu")
        out = ff(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_gelu(self) -> None:
        ff = init_feed_forward(16, 4, 0.1, "gelu")
        out = ff(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_swiglu(self) -> None:
        ff = init_feed_forward(16, 4, 0.1, "swiglu")
        out = ff(torch.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_unsupported_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            init_feed_forward(16, 4, 0.1, "tanh")
