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

import tempfile

import numpy as np
from pytorch_lightning import seed_everything

from rectools.semantic.tokenizer.model import SIDTokenizer


class TestSIDTokenizerFit:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_fit_rkmeans(self) -> None:
        rng = np.random.RandomState(42)
        item_ids = list(range(100))
        embeddings = rng.randn(100, 16).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tok = SIDTokenizer(
                input_dim=16,
                codebook_sizes=[4, 4],
                quantizer="rkmeans",
                device="cpu",
            )
            tok.fit(
                item_ids=item_ids,
                embeddings=embeddings,
                max_epochs=3,
                patience=2,
                batch_size=50,
                save_dir=tmpdir,
            )
            assert len(tok.id2sid) > 0

    def test_fit_rqvae(self) -> None:
        rng = np.random.RandomState(42)
        item_ids = list(range(100))
        embeddings = rng.randn(100, 16).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tok = SIDTokenizer(
                input_dim=16,
                codebook_sizes=[4, 4],
                codebook_dim=8,
                hidden_dims=[8],
                quantizer="rqvae",
                adapter_proj_dim=8,
                device="cpu",
            )
            tok.fit(
                item_ids=item_ids,
                embeddings=embeddings,
                max_epochs=3,
                patience=2,
                batch_size=50,
                save_dir=tmpdir,
            )
            assert len(tok.id2sid) > 0
