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

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.semantic.tiger.lightning import TIGERLightning
from rectools.semantic.tiger.module import TIGERNet
from rectools.semantic.tokenizer.emb_dataset import EmbDataset
from rectools.semantic.tokenizer.model import SIDTokenizer


@pytest.fixture
def trained_tokenizer() -> SIDTokenizer:
    seed_everything(42, workers=True)
    rng = np.random.RandomState(42)
    item_ids = list(range(1, 21))
    embeddings = rng.randn(20, 16).astype(np.float32)
    dataset = EmbDataset(item_ids, embeddings)
    tok = SIDTokenizer(
        codebook_dim=16,
        hidden_dims=None,
        codebook_sizes=[4, 4],
        input_dim=16,
        device="cpu",
        quantizer="rkmeans",
    )
    tok.init_codebooks(dataset)
    batch = dataset[:]
    tok(batch)
    return tok


class TestTIGERLightning:  # pylint: disable=redefined-outer-name
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    @pytest.fixture
    def lightning_model(self, trained_tokenizer: SIDTokenizer) -> TIGERLightning:
        net = TIGERNet(
            codebook_sizes=[4, 4],
            hidden_units=16,
            num_blocks=1,
            num_heads=2,
            dropout_rate=0.0,
            max_length=10,
        )
        return TIGERLightning(
            model=net,
            tokenizer=trained_tokenizer,
            beam_size=4,
            top_k=3,
            num_items=20,
        )

    def test_training_step(self, lightning_model: TIGERLightning) -> None:
        batch = {
            "input_ids": torch.randint(2, 10, (4, 6)),
            "dec_input": torch.randint(0, 10, (4, 2)),
            "labels": torch.randint(0, 4, (4, 2)),
        }
        loss = lightning_model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_configure_optimizers_adamw(self, lightning_model: TIGERLightning) -> None:
        result = lightning_model.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_configure_optimizers_with_schedule(self, trained_tokenizer: SIDTokenizer) -> None:
        net = TIGERNet(codebook_sizes=[4, 4], hidden_units=16, num_blocks=1, num_heads=2)
        model = TIGERLightning(
            model=net,
            tokenizer=trained_tokenizer,
            lr_schedule="cosine",
            warmup_steps=10,
            max_iters=100,
        )
        result = model.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_configure_optimizers_unknown_raises(self, trained_tokenizer: SIDTokenizer) -> None:
        net = TIGERNet(codebook_sizes=[4, 4], hidden_units=16, num_blocks=1, num_heads=2)
        model = TIGERLightning(
            model=net,
            tokenizer=trained_tokenizer,
            optimizer_name="unknown",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.configure_optimizers()
