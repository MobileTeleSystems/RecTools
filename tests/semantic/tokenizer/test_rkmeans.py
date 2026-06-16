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

from rectools.semantic.tokenizer.rkmeans import Codebook, RKmeans


class TestCodebook:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_output_shapes(self) -> None:
        cb = Codebook(code_dim=8, n_codes=16)
        inputs = torch.randn(4, 8)
        codes, quantized, residuals = cb(inputs)
        assert codes.shape == (4,)
        assert quantized.shape == (4, 8)
        assert residuals.shape == (4, 8)

    def test_codes_in_range(self) -> None:
        cb = Codebook(code_dim=8, n_codes=16)
        inputs = torch.randn(10, 8)
        codes, _, _ = cb(inputs)
        assert (codes >= 0).all()
        assert (codes < 16).all()

    def test_init_from_centroids(self) -> None:
        cb = Codebook(code_dim=4, n_codes=3)
        centroids = torch.randn(3, 4)
        cb.init_from_centroids(centroids)
        assert cb.kmeans_initted_
        assert torch.allclose(cb.code_embs.weight.data, centroids)


class TestRKmeans:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_forward_shapes(self) -> None:
        model = RKmeans(input_dim=16, codebook_sizes=[8, 8])
        inputs = torch.randn(10, 16)
        sem_ids, loss = model(inputs)
        assert len(sem_ids) == 10
        assert all(len(sid) == 2 for sid in sem_ids)
        assert loss.dim() == 0

    def test_loss_is_positive(self) -> None:
        model = RKmeans(input_dim=16, codebook_sizes=[8, 8])
        inputs = torch.randn(10, 16)
        _, loss = model(inputs)
        assert loss.item() > 0.0

    def test_init_codebooks(self) -> None:
        model = RKmeans(input_dim=16, codebook_sizes=[4, 4])
        data = torch.randn(100, 16)
        model.init_codebooks(data)
        for layer in model.codebooks:
            assert layer.kmeans_initted_

    def test_sem_ids_are_tuples(self) -> None:
        model = RKmeans(input_dim=8, codebook_sizes=[4, 4, 4])
        inputs = torch.randn(5, 8)
        sem_ids, _ = model(inputs)
        for sid in sem_ids:
            assert isinstance(sid, tuple)
            assert len(sid) == 3
