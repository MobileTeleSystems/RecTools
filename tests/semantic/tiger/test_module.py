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

from rectools.semantic.tiger.module import TIGERNet


class TestTIGERNetStandard:
    """Tests for TIGERNet with standard nn.Transformer (d_kv=None)."""

    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    @pytest.fixture
    def model(self) -> TIGERNet:
        return TIGERNet(
            codebook_sizes=[4, 4],
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            dropout_rate=0.0,
            max_length=10,
        )

    def test_forward_shapes(self, model: TIGERNet) -> None:
        batch_size = 3
        enc_len = 4  # 2 items * 2 codewords
        dec_len = 2  # sid_len = 2
        enc_input = torch.randint(2, 10, (batch_size, enc_len))
        dec_input = torch.randint(0, 10, (batch_size, dec_len))
        logits = model(enc_input, dec_input)
        assert len(logits) == 2
        assert logits[0].shape == (batch_size, 4)
        assert logits[1].shape == (batch_size, 4)

    def test_generate_greedy_shape(self, model: TIGERNet) -> None:
        model.eval()
        enc_input = torch.randint(2, 10, (2, 4))
        codes = model.generate_greedy(enc_input)
        assert codes.shape == (2, 2)  # (batch_size, sid_len)

    def test_generate_beam_shape(self, model: TIGERNet) -> None:
        model.eval()
        enc_input = torch.randint(2, 10, (2, 4))
        codes, scores = model.generate(enc_input, beam_size=3)
        assert codes.shape == (2, 3, 2)  # (batch, beam, sid_len)
        assert scores.shape == (2, 3)

    def test_generate_with_padding_mask(self, model: TIGERNet) -> None:
        model.eval()
        enc_input = torch.randint(2, 10, (2, 6))
        enc_input[1, 4:] = TIGERNet.PAD_TOKEN_ID
        mask = enc_input == TIGERNet.PAD_TOKEN_ID
        codes, _scores = model.generate(enc_input, enc_padding_mask=mask, beam_size=2)
        assert codes.shape == (2, 2, 2)

    def test_sid_to_tokens_roundtrip(self, model: TIGERNet) -> None:
        sid = torch.tensor([[0, 1], [2, 3]])
        tokens = model.sid_to_tokens(sid)
        recovered = model.tokens_to_sid(tokens)
        assert torch.equal(sid, recovered)

    def test_codes_in_valid_range(self, model: TIGERNet) -> None:
        model.eval()
        enc_input = torch.randint(2, 10, (3, 4))
        codes = model.generate_greedy(enc_input)
        for d in range(model.sid_len):
            assert (codes[:, d] >= 0).all()
            assert (codes[:, d] < model.codebook_sizes[d]).all()


class TestTIGERNetT5:
    """Tests for TIGERNet with T5-style layers (d_kv set)."""

    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    @pytest.fixture
    def model(self) -> TIGERNet:
        return TIGERNet(
            codebook_sizes=[4, 4],
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            d_kv=8,
            dropout_rate=0.0,
            max_length=10,
        )

    def test_forward_shapes(self, model: TIGERNet) -> None:
        enc_input = torch.randint(2, 10, (2, 4))
        dec_input = torch.randint(0, 10, (2, 2))
        logits = model(enc_input, dec_input)
        assert len(logits) == 2
        assert logits[0].shape == (2, 4)

    def test_generate_greedy(self, model: TIGERNet) -> None:
        model.eval()
        enc_input = torch.randint(2, 10, (2, 4))
        codes = model.generate_greedy(enc_input)
        assert codes.shape == (2, 2)

    def test_generate_beam(self, model: TIGERNet) -> None:
        model.eval()
        enc_input = torch.randint(2, 10, (2, 4))
        codes, scores = model.generate(enc_input, beam_size=3)
        assert codes.shape == (2, 3, 2)
        assert scores.shape == (2, 3)

    def test_no_positional_embeddings(self, model: TIGERNet) -> None:
        assert model.enc_pos_emb is None
        assert model.dec_pos_emb is None
