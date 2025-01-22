#  Copyright 2024 MTS (Mobile Telesystems)
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

import typing as tp

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.models import BERT4RecModel
from rectools.models.nn.bert4rec import BERT4RecDataPreparator
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.models.nn.transformer_base import (
    LearnableInversePositionalEncoding,
    PreLNTransformerLayers,
    SessionEncoderLightningModule,
)
from tests.models.data import DATASET
from tests.models.utils import assert_default_config_and_default_model_params_are_the_same

# TODO: add tests with BCE and GBCE (they can be broken for the model when softmax is ok => we need happy path test)


class TestBERT4RecModelConfiguration:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    def test_from_config(self) -> None:
        config = {
            "n_blocks": 2,
            "n_heads": 4,
            "n_factors": 64,
            "use_pos_emb": False,
            "use_causal_attn": False,
            "use_key_padding_mask": True,
            "dropout_rate": 0.5,
            "session_max_len": 10,
            "dataloader_num_workers": 5,
            "batch_size": 1024,
            "loss": "softmax",
            "n_negatives": 10,
            "gbce_t": 0.5,
            "lr": 0.001,
            "epochs": 10,
            "verbose": 1,
            "deterministic": True,
            "recommend_device": "auto",
            "recommend_n_threads": 0,
            "recommend_use_gpu_ranking": True,
            "train_min_user_interactions": 5,
            "item_net_block_types": (IdEmbeddingsItemNet,),
            "pos_encoding_type": LearnableInversePositionalEncoding,
            "transformer_layers_type": PreLNTransformerLayers,
            "data_preparator_type": BERT4RecDataPreparator,
            "lightning_module_type": SessionEncoderLightningModule,
            "mask_prob": 0.15,
        }
        model = BERT4RecModel.from_config(config)
        assert model.n_blocks == 2
        assert model.n_heads == 4
        assert model.n_factors == 64
        assert model.use_pos_emb is False
        assert model.use_causal_attn is False
        assert model.use_key_padding_mask is True
        assert model.dropout_rate == 0.5
        assert model.session_max_len == 10
        assert model.dataloader_num_workers == 5
        assert model.batch_size == 1024
        assert model.loss == "softmax"
        assert model.n_negatives == 10
        assert model.gbce_t == 0.5
        assert model.lr == 0.001
        assert model.epochs == 10
        assert model.verbose == 1
        assert model.deterministic is True
        assert model.recommend_device == "auto"
        assert model.recommend_n_threads == 0
        assert model.recommend_use_gpu_ranking is True
        assert model.train_min_user_interactions == 5
        assert model._trainer is not None  # pylint: disable = protected-access
        assert model.item_net_block_types == (IdEmbeddingsItemNet,)
        assert model.pos_encoding_type == LearnableInversePositionalEncoding
        assert model.transformer_layers_type == PreLNTransformerLayers
        assert model.data_preparator_type == BERT4RecDataPreparator
        assert model.lightning_module_type == SessionEncoderLightningModule
        assert model.mask_prob == 0.15

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config(self, simple_types: bool) -> None:
        model = BERT4RecModel(
            n_blocks=2,
            n_heads=4,
            n_factors=64,
            use_pos_emb=False,
            use_causal_attn=False,
            use_key_padding_mask=True,
            dropout_rate=0.5,
            session_max_len=10,
            dataloader_num_workers=5,
            batch_size=1024,
            loss="softmax",
            n_negatives=10,
            gbce_t=0.5,
            lr=0.001,
            epochs=10,
            verbose=1,
            deterministic=True,
            recommend_device="auto",
            recommend_n_threads=0,
            recommend_use_gpu_ranking=True,
            train_min_user_interactions=5,
            item_net_block_types=(IdEmbeddingsItemNet,),
            pos_encoding_type=LearnableInversePositionalEncoding,
            transformer_layers_type=PreLNTransformerLayers,
            data_preparator_type=BERT4RecDataPreparator,
            lightning_module_type=SessionEncoderLightningModule,
            mask_prob=0.15,
        )
        config = model.get_config(simple_types=simple_types)
        expected = {
            "cls": "BERT4RecModel" if simple_types else BERT4RecModel,
            "n_blocks": 2,
            "n_heads": 4,
            "n_factors": 64,
            "use_pos_emb": False,
            "use_causal_attn": False,
            "use_key_padding_mask": True,
            "dropout_rate": 0.5,
            "session_max_len": 10,
            "dataloader_num_workers": 5,
            "batch_size": 1024,
            "loss": "softmax",
            "n_negatives": 10,
            "gbce_t": 0.5,
            "lr": 0.001,
            "epochs": 10,
            "verbose": 1,
            "deterministic": True,
            "recommend_device": "auto",
            "recommend_n_threads": 0,
            "recommend_use_gpu_ranking": True,
            "train_min_user_interactions": 5,
            "item_net_block_types": (
                ["rectools.models.nn.item_net.IdEmbeddingsItemNet"] if simple_types else (IdEmbeddingsItemNet,)
            ),
            "pos_encoding_type": (
                "rectools.models.nn.transformer_net_blocks.LearnableInversePositionalEncoding"
                if simple_types
                else LearnableInversePositionalEncoding
            ),
            "transformer_layers_type": (
                "rectools.models.nn.transformer_net_blocks.PreLNTransformerLayers"
                if simple_types
                else PreLNTransformerLayers
            ),
            "data_preparator_type": (
                "rectools.models.nn.bert4rec.BERT4RecDataPreparator" if simple_types else BERT4RecDataPreparator
            ),
            "lightning_module_type": (
                "rectools.models.nn.transformer_base.SessionEncoderLightningModule"
                if simple_types
                else SessionEncoderLightningModule
            ),
            "mask_prob": 0.15,
        }
        assert config == expected

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(self, simple_types: bool) -> None:
        initial_config = {
            "n_blocks": 1,
            "n_heads": 1,
            "n_factors": 10,
            "use_pos_emb": False,
            "use_causal_attn": False,
            "use_key_padding_mask": True,
            "dropout_rate": 0.5,
            "session_max_len": 5,
            "dataloader_num_workers": 1,
            "batch_size": 100,
            "loss": "softmax",
            "n_negatives": 4,
            "gbce_t": 0.5,
            "lr": 0.001,
            "epochs": 1,
            "verbose": 0,
            "deterministic": True,
            "recommend_device": "auto",
            "recommend_n_threads": 0,
            "recommend_use_gpu_ranking": True,
            "train_min_user_interactions": 2,
            "item_net_block_types": (IdEmbeddingsItemNet,),
            "pos_encoding_type": LearnableInversePositionalEncoding,
            "transformer_layers_type": PreLNTransformerLayers,
            "data_preparator_type": BERT4RecDataPreparator,
            "lightning_module_type": SessionEncoderLightningModule,
            "mask_prob": 0.15,
        }

        dataset = DATASET
        model = BERT4RecModel

        def get_reco(model: BERT4RecModel) -> pd.DataFrame:
            return model.fit(dataset).recommend(users=np.array([10, 20]), dataset=dataset, k=2, filter_viewed=False)

        model_1 = model.from_config(initial_config)
        reco_1 = get_reco(model_1)
        config_1 = model_1.get_config(simple_types=simple_types)

        self._seed_everything()
        model_2 = model.from_config(config_1)
        reco_2 = get_reco(model_2)
        config_2 = model_2.get_config(simple_types=simple_types)

        assert config_1 == config_2
        pd.testing.assert_frame_equal(reco_1, reco_2)

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        default_config: tp.Dict[str, int] = {}
        model = BERT4RecModel()
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
