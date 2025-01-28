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

from .utils import leave_one_out_mask


class TestBERT4RecModelConfiguration:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def initial_config(self) -> tp.Dict[str, tp.Any]:
        config = {
            "n_blocks": 2,
            "n_heads": 4,
            "n_factors": 64,
            "use_pos_emb": False,
            "use_causal_attn": False,
            "use_key_padding_mask": True,
            "dropout_rate": 0.5,
            "session_max_len": 10,
            "dataloader_num_workers": 0,
            "batch_size": 1024,
            "loss": "softmax",
            "n_negatives": 10,
            "gbce_t": 0.5,
            "lr": 0.001,
            "epochs": 10,
            "verbose": 1,
            "deterministic": True,
            "recommend_accelerator": "auto",
            "recommend_devices": 1,
            "recommend_batch_size": 256,
            "recommend_n_threads": 0,
            "recommend_use_gpu_ranking": True,
            "train_min_user_interactions": 2,
            "item_net_block_types": (IdEmbeddingsItemNet,),
            "pos_encoding_type": LearnableInversePositionalEncoding,
            "transformer_layers_type": PreLNTransformerLayers,
            "data_preparator_type": BERT4RecDataPreparator,
            "lightning_module_type": SessionEncoderLightningModule,
            "mask_prob": 0.15,
            "get_val_mask_func": leave_one_out_mask,
        }
        return config

    def test_from_config(self, initial_config: tp.Dict[str, tp.Any]) -> None:
        model = BERT4RecModel.from_config(initial_config)

        for key, config_value in initial_config.items():
            assert getattr(model, key) == config_value

        assert model._trainer is not None  # pylint: disable = protected-access

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config(self, simple_types: bool, initial_config: tp.Dict[str, tp.Any]) -> None:
        model = BERT4RecModel(**initial_config)
        config = model.get_config(simple_types=simple_types)

        expected = initial_config.copy()
        expected["cls"] = BERT4RecModel

        if simple_types:
            simple_types_params = {
                "cls": "BERT4RecModel",
                "item_net_block_types": ["rectools.models.nn.item_net.IdEmbeddingsItemNet"],
                "pos_encoding_type": "rectools.models.nn.transformer_net_blocks.LearnableInversePositionalEncoding",
                "transformer_layers_type": "rectools.models.nn.transformer_net_blocks.PreLNTransformerLayers",
                "data_preparator_type": "rectools.models.nn.bert4rec.BERT4RecDataPreparator",
                "lightning_module_type": "rectools.models.nn.transformer_base.SessionEncoderLightningModule",
                "get_val_mask_func": "tests.models.nn.utils.leave_one_out_mask",
            }
            expected.update(simple_types_params)

        assert config == expected

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(
        self, simple_types: bool, initial_config: tp.Dict[str, tp.Any]
    ) -> None:
        dataset = DATASET
        model = BERT4RecModel
        updated_params = {
            "n_blocks": 1,
            "n_heads": 1,
            "n_factors": 10,
            "session_max_len": 5,
            "epochs": 1,
        }
        config = initial_config.copy()
        config.update(updated_params)

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
