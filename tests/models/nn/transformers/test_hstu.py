import typing as tp

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger

import rectools.dataset.context as context_prep
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.models import HSTUModel
from rectools.models.nn.item_net import IdEmbeddingsItemNet, SumOfEmbeddingsConstructor
from rectools.models.nn.transformers.base import LearnableInversePositionalEncoding, TransformerLightningModule
from rectools.models.nn.transformers.hstu import STULayers
from rectools.models.nn.transformers.negative_sampler import CatalogUniformSampler
from rectools.models.nn.transformers.sasrec import SASRecDataPreparator
from rectools.models.nn.transformers.similarity import DistanceSimilarityModule
from rectools.models.nn.transformers.torch_backbone import TransformerTorchBackbone
from tests.models.data import DATASET
from tests.models.utils import assert_default_config_and_default_model_params_are_the_same

from .utils import custom_trainer, leave_one_out_mask


class TestHSTUModel:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset_devices(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 13, 2, "2021-11-26"],
                [40, 11, 1, "2021-11-25"],
                [50, 13, 1, "2021-11-25"],
                [10, 13, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def context_df(self) -> pd.DataFrame:
        # "2021-12-12" generation moment simulation
        df = pd.DataFrame(
            {
                Columns.User: [10, 20, 30, 40, 50],
                Columns.Datetime: ["2021-12-12", "2021-12-12", "2021-12-12", "2021-12-12", "2021-12-12"],
            }
        )
        return df

    @pytest.mark.parametrize(
        "accelerator,n_devices,recommend_torch_device",
        [
            ("cpu", 1, "cpu"),
            pytest.param(
                "cpu",
                1,
                "cuda",
                marks=pytest.mark.skipif(torch.cuda.is_available() is False, reason="GPU is not available"),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "relative_time_attention,relative_pos_attention,expected_reco",
        (
            (
                True,
                True,
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                False,
                True,
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 13, 12],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                True,
                False,
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                False,
                False,
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 13, 12],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_u2i(
        self,
        dataset_devices: Dataset,
        context_df: pd.DataFrame,
        accelerator: str,
        n_devices: int,
        recommend_torch_device: str,
        relative_time_attention: bool,
        relative_pos_attention: bool,
        expected_reco: pd.DataFrame,
    ) -> None:
        self.setup_method()

        def get_trainer() -> Trainer:
            return Trainer(
                max_epochs=2,
                min_epochs=2,
                deterministic=True,
                devices=n_devices,
                accelerator=accelerator,
                enable_checkpointing=False,
                logger=CSVLogger("test_logs"),
            )

        model = HSTUModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            relative_pos_attention=relative_pos_attention,
            relative_time_attention=relative_time_attention,
            recommend_torch_device=recommend_torch_device,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset_devices)
        users = np.array([10, 30, 40])
        if model.require_recommend_context:
            context = context_prep.get_context(context_df)
        else:
            context = None
        if relative_time_attention:
            with pytest.raises(ValueError):
                model.recommend(users=users, dataset=dataset_devices, k=3, filter_viewed=True, context=None)
        if not relative_time_attention:
            with pytest.raises(ValueError):
                model.recommend(
                    users=users,
                    dataset=dataset_devices,
                    k=3,
                    filter_viewed=True,
                    context=context_prep.get_context(context_df),
                )
        actual = model.recommend(users=users, dataset=dataset_devices, k=3, filter_viewed=True, context=context)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected_reco)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )


class TestHSTUModelConfiguration:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def context_df(self) -> pd.DataFrame:
        # "2021-12-12" generation moment simulation
        df = pd.DataFrame(
            {
                Columns.User: [10, 20, 30, 40, 50],
                Columns.Datetime: ["2021-12-12", "2021-12-12", "2021-12-12", "2021-12-12", "2021-12-12"],
            }
        )
        return df

    @pytest.fixture
    def initial_config(self) -> tp.Dict[str, tp.Any]:
        config = {
            "n_blocks": 2,
            "relative_time_attention": True,
            "relative_pos_attention": True,
            "n_heads": 4,
            "n_factors": 64,
            "use_pos_emb": True,
            "use_causal_attn": True,
            "use_key_padding_mask": False,
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
            "recommend_torch_device": None,
            "recommend_batch_size": 256,
            "train_min_user_interactions": 2,
            "item_net_block_types": (IdEmbeddingsItemNet,),
            "item_net_constructor_type": SumOfEmbeddingsConstructor,
            "pos_encoding_type": LearnableInversePositionalEncoding,
            "transformer_layers_type": STULayers,
            "data_preparator_type": SASRecDataPreparator,
            "lightning_module_type": TransformerLightningModule,
            "negative_sampler_type": CatalogUniformSampler,
            "similarity_module_type": DistanceSimilarityModule,
            "backbone_type": TransformerTorchBackbone,
            "get_val_mask_func": leave_one_out_mask,
            "get_trainer_func": None,
            "get_val_mask_func_kwargs": None,
            "get_trainer_func_kwargs": None,
            "data_preparator_kwargs": None,
            "transformer_layers_kwargs": None,
            "item_net_constructor_kwargs": None,
            "pos_encoding_kwargs": None,
            "lightning_module_kwargs": None,
            "negative_sampler_kwargs": None,
            "similarity_module_kwargs": None,
            "backbone_kwargs": None,
        }
        return config

    @pytest.mark.parametrize("use_custom_trainer", (True, False))
    def test_from_config(self, initial_config: tp.Dict[str, tp.Any], use_custom_trainer: bool) -> None:
        config = initial_config
        if use_custom_trainer:
            config["get_trainer_func"] = custom_trainer
        model = HSTUModel.from_config(initial_config)

        for key, config_value in initial_config.items():
            assert getattr(model, key) == config_value

        assert model._trainer is not None  # pylint: disable = protected-access

    @pytest.mark.parametrize("use_custom_trainer", (True, False))
    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config(
        self, simple_types: bool, initial_config: tp.Dict[str, tp.Any], use_custom_trainer: bool
    ) -> None:
        config = initial_config
        if use_custom_trainer:
            config["get_trainer_func"] = custom_trainer
        model = HSTUModel(**config)
        actual = model.get_config(simple_types=simple_types)

        expected = config.copy()
        expected["cls"] = HSTUModel

        if simple_types:
            simple_types_params = {
                "cls": "HSTUModel",
                "item_net_block_types": ["rectools.models.nn.item_net.IdEmbeddingsItemNet"],
                "item_net_constructor_type": "rectools.models.nn.item_net.SumOfEmbeddingsConstructor",
                "pos_encoding_type": "rectools.models.nn.transformers.net_blocks.LearnableInversePositionalEncoding",
                "transformer_layers_type": "rectools.models.nn.transformers.hstu.STULayers",
                "data_preparator_type": "rectools.models.nn.transformers.sasrec.SASRecDataPreparator",
                "lightning_module_type": "rectools.models.nn.transformers.lightning.TransformerLightningModule",
                "negative_sampler_type": "rectools.models.nn.transformers.negative_sampler.CatalogUniformSampler",
                "get_val_mask_func": "tests.models.nn.transformers.utils.leave_one_out_mask",
                "similarity_module_type": "rectools.models.nn.transformers.similarity.DistanceSimilarityModule",
                "backbone_type": "rectools.models.nn.transformers.torch_backbone.TransformerTorchBackbone",
            }
            expected.update(simple_types_params)
            if use_custom_trainer:
                expected["get_trainer_func"] = "tests.models.nn.transformers.utils.custom_trainer"

        assert actual == expected

    @pytest.mark.parametrize("use_custom_trainer", (True, False))
    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(
        self,
        context_df: pd.DataFrame,
        simple_types: bool,
        initial_config: tp.Dict[str, tp.Any],
        use_custom_trainer: bool,
    ) -> None:
        dataset = DATASET
        model = HSTUModel
        updated_params = {
            "n_blocks": 1,
            "n_heads": 1,
            "n_factors": 10,
            "session_max_len": 5,
            "epochs": 1,
        }
        config = initial_config.copy()
        config.update(updated_params)
        if use_custom_trainer:
            config["get_trainer_func"] = custom_trainer

        def get_reco(model: HSTUModel) -> pd.DataFrame:
            return model.fit(dataset).recommend(
                users=np.array([10, 20]),
                dataset=dataset,
                k=2,
                filter_viewed=False,
                context=context_prep.get_context(context_df),
            )

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
        model = HSTUModel()
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
