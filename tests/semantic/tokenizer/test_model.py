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

import os
import tempfile

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.semantic.tokenizer.emb_dataset import EmbDataset
from rectools.semantic.tokenizer.model import SIDTokenizer


class TestSIDTokenizerRKmeans:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    @pytest.fixture
    def dataset(self) -> EmbDataset:
        rng = np.random.RandomState(42)
        item_ids = list(range(100))
        embeddings = rng.randn(100, 16).astype(np.float32)
        return EmbDataset(item_ids, embeddings)

    @pytest.fixture
    def tokenizer(self, dataset: EmbDataset) -> SIDTokenizer:
        tok = SIDTokenizer(
            codebook_dim=16,
            hidden_dims=None,
            codebook_sizes=[8, 8],
            input_dim=16,
            device="cpu",
            quantizer="rkmeans",
        )
        tok.init_codebooks(dataset)
        # Run a forward pass to populate id2sid
        batch = dataset[:]
        tok(batch)
        return tok

    def test_init(self) -> None:
        tok = SIDTokenizer(
            codebook_dim=16,
            hidden_dims=None,
            codebook_sizes=[8, 8],
            input_dim=16,
            quantizer="rkmeans",
        )
        assert tok.quantizer_name == "rkmeans"
        assert tok.codebook_sizes == [8, 8]

    def test_forward_returns_loss(self, dataset: EmbDataset) -> None:
        tok = SIDTokenizer(
            codebook_dim=16,
            hidden_dims=None,
            codebook_sizes=[8, 8],
            input_dim=16,
            device="cpu",
            quantizer="rkmeans",
        )
        tok.init_codebooks(dataset)
        batch = dataset[0:10]
        loss = tok(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_populates_id2sid(self, dataset: EmbDataset) -> None:
        tok = SIDTokenizer(
            codebook_dim=16,
            hidden_dims=None,
            codebook_sizes=[8, 8],
            input_dim=16,
            device="cpu",
            quantizer="rkmeans",
        )
        tok.init_codebooks(dataset)
        batch = dataset[0:10]
        tok(batch)
        assert len(tok.id2sid) == 10

    def test_tokenize_single(self, tokenizer: SIDTokenizer) -> None:
        sid = tokenizer.tokenize(0)
        assert isinstance(sid, tuple)
        assert len(sid) == 2

    def test_tokenize_multiple(self, tokenizer: SIDTokenizer) -> None:
        sids = tokenizer.tokenize([0, 1, 2])
        assert isinstance(sids, list)
        assert len(sids) == 3
        for sid in sids:
            assert isinstance(sid, tuple)
            assert len(sid) == 2

    def test_tokenize_unknown_item_raises(self, tokenizer: SIDTokenizer) -> None:
        with pytest.raises(ValueError, match="out of vocabulary"):
            tokenizer.tokenize(9999)

    def test_decode_single(self, tokenizer: SIDTokenizer) -> None:
        sid = tokenizer.tokenize(5)
        item_id = tokenizer.decode(sid)
        # The decoded item should exist in the vocabulary
        assert item_id is not None

    def test_decode_multiple(self, tokenizer: SIDTokenizer) -> None:
        sids = tokenizer.tokenize([0, 1, 2])
        decoded = tokenizer.decode(sids)
        assert isinstance(decoded, list)
        assert len(decoded) == 3

    def test_decode_unknown_sid(self, tokenizer: SIDTokenizer) -> None:
        unknown_sid = (999, 999)
        result = tokenizer.decode(unknown_sid)
        assert result is None

    def test_decode_with_default_value(self, tokenizer: SIDTokenizer) -> None:
        unknown_sid = (999, 999)
        result = tokenizer.decode(unknown_sid, default_value=-1)
        assert result == -1

    def test_decode_collision_is_reproducible_with_rng(self, tokenizer: SIDTokenizer) -> None:
        tokenizer.id2sid = {
            10: (1, 1),
            20: (1, 1),
            30: (2, 2),
        }
        tokenizer.sid2id.clear()

        first_rng = np.random.RandomState(42)
        second_rng = np.random.RandomState(42)

        first = tokenizer.decode([(1, 1), (1, 1), (1, 1)], rng=first_rng)
        second = tokenizer.decode([(1, 1), (1, 1), (1, 1)], rng=second_rng)

        assert first == second

    def test_len(self, tokenizer: SIDTokenizer) -> None:
        assert len(tokenizer) > 0
        assert len(tokenizer) <= 100

    def test_extend(self, tokenizer: SIDTokenizer) -> None:
        rng = np.random.RandomState(99)
        new_ids = list(range(100, 120))
        new_embeddings = rng.randn(20, 16).astype(np.float32)
        new_dataset = EmbDataset(new_ids, new_embeddings)
        tokenizer.extend(new_dataset)
        # New items should be in id2sid
        for item_id in new_ids:
            assert item_id in tokenizer.id2sid

    def test_extend_skips_existing(self, tokenizer: SIDTokenizer) -> None:
        old_sid = tokenizer.id2sid[0]
        rng = np.random.RandomState(99)
        # Include existing item 0
        ids = [0, 200]
        embs = rng.randn(2, 16).astype(np.float32)
        ds = EmbDataset(ids, embs)
        tokenizer.extend(ds)
        # Item 0 should still have the same SID
        assert tokenizer.id2sid[0] == old_sid
        assert 200 in tokenizer.id2sid

    def test_save_and_load(self, tokenizer: SIDTokenizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tokenizer.pt")
            tokenizer.save(path)
            loaded = SIDTokenizer.load(path, map_location="cpu")

            assert loaded.codebook_sizes == tokenizer.codebook_sizes
            assert loaded.input_dim == tokenizer.input_dim
            assert loaded.codebook_dim == tokenizer.codebook_dim
            assert set(loaded.id2sid.keys()) == set(tokenizer.id2sid.keys())

            # Check that tokenization produces same results
            sid_orig = tokenizer.tokenize(5)
            sid_loaded = loaded.tokenize(5)
            assert sid_orig == sid_loaded


class TestSIDTokenizerRQVAE:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    @pytest.fixture
    def dataset(self) -> EmbDataset:
        rng = np.random.RandomState(42)
        item_ids = list(range(100))
        embeddings = rng.randn(100, 16).astype(np.float32)
        return EmbDataset(item_ids, embeddings)

    def test_init_rqvae(self) -> None:
        tok = SIDTokenizer(
            codebook_dim=8,
            hidden_dims=[8],
            codebook_sizes=[4, 4],
            input_dim=16,
            quantizer="rqvae",
        )
        assert tok.quantizer_name == "rqvae"

    def test_forward_rqvae(self, dataset: EmbDataset) -> None:
        tok = SIDTokenizer(
            codebook_dim=8,
            hidden_dims=[8],
            codebook_sizes=[4, 4],
            input_dim=16,
            device="cpu",
            quantizer="rqvae",
            adapter_proj_dim=8,
        )
        tok.init_codebooks(dataset)
        batch = dataset[0:10]
        loss = tok(batch)
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
        assert len(tok.id2sid) == 10
