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

import torch
from pytorch_lightning import seed_everything

from rectools.semantic.tokenizer.rqvae import RQVAE, WhiteningAdapter


class TestWhiteningAdapter:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_output_shape(self) -> None:
        adapter = WhiteningAdapter(adapter_type="identity", emb_dim=16, proj_dim=8)
        inputs = torch.randn(5, 16)
        adapter.init_from_embeds(inputs)
        output = adapter(inputs)
        assert output.shape == (5, 8)

    def test_ffn_adapter(self) -> None:
        adapter = WhiteningAdapter(adapter_type="ffn", emb_dim=16, proj_dim=8, hidden_units=4)
        inputs = torch.randn(10, 16)
        adapter.init_from_embeds(inputs)
        output = adapter(inputs)
        assert output.shape == (10, 8)

    def test_freeze_adapter(self) -> None:
        adapter = WhiteningAdapter(emb_dim=16, proj_dim=8)
        inputs = torch.randn(10, 16)
        adapter.init_from_embeds(inputs)
        assert not adapter.weight.requires_grad
        assert not adapter.bias.requires_grad


class TestRQVAE:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_forward_shapes(self) -> None:
        model = RQVAE(
            input_dim=16,
            codebook_dim=8,
            hidden_dims=[8],
            codebook_sizes=[4, 4],
            adapter_proj_dim=8,
        )
        data = torch.randn(20, 16)
        model.init_codebooks(data)
        sem_ids, loss = model(data)
        assert len(sem_ids) == 20
        assert all(len(sid) == 2 for sid in sem_ids)
        assert loss.dim() == 0

    def test_loss_is_finite(self) -> None:
        model = RQVAE(
            input_dim=16,
            codebook_dim=8,
            hidden_dims=[8],
            codebook_sizes=[4, 4],
            adapter_proj_dim=8,
        )
        data = torch.randn(20, 16)
        model.init_codebooks(data)
        _, loss = model(data)
        assert torch.isfinite(loss)

    def test_init_codebooks_sets_flag(self) -> None:
        model = RQVAE(
            input_dim=16,
            codebook_dim=8,
            hidden_dims=[8],
            codebook_sizes=[4, 4],
            adapter_proj_dim=8,
        )
        data = torch.randn(50, 16)
        model.init_codebooks(data)
        for layer in model.codebooks:
            assert layer.kmeans_initted_

    def test_sem_ids_are_tuples_of_ints(self) -> None:
        model = RQVAE(
            input_dim=16,
            codebook_dim=8,
            hidden_dims=[8],
            codebook_sizes=[4, 4, 4],
            adapter_proj_dim=8,
        )
        data = torch.randn(20, 16)
        model.init_codebooks(data)
        sem_ids, _ = model(data)
        for sid in sem_ids:
            assert isinstance(sid, tuple)
            assert all(isinstance(c, int) for c in sid)
            assert len(sid) == 3
