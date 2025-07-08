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
import typing as tp
from copy import deepcopy
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from pytest import FixtureRequest
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from torch import nn

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import BERT4RecModel, SASRecModel, load_model
from rectools.models.nn.transformers.base import TransformerModelBase
from rectools.models.nn.transformers.lightning import TransformerLightningModule
from tests.models.data import INTERACTIONS
from tests.models.utils import assert_save_load_do_not_change_model

from .utils import custom_trainer, custom_trainer_ckpt, custom_trainer_multiple_ckpt, leave_one_out_mask


def assert_torch_models_equal(model_a: nn.Module, model_b: nn.Module) -> None:
    assert type(model_a) is type(model_b), "different types"

    with torch.no_grad():
        for (apn, apv), (bpn, bpv) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert apn == bpn, "different parameter name"
            assert torch.isclose(apv, bpv).all(), "different parameter value"


def assert_pl_models_equal(model_a: pl.LightningModule, model_b: pl.LightningModule) -> None:
    """Assert pl modules are equal in terms of weights and trainer"""
    assert_torch_models_equal(model_a, model_b)

    trainer_a = model_a.trainer
    trainer_b = model_a.trainer

    assert_pl_trainers_equal(trainer_a, trainer_b)


def assert_pl_trainers_equal(trainer_a: Trainer, trainer_b: Trainer) -> None:
    """Assert pl trainers are equal in terms of optimizers state"""
    assert len(trainer_a.optimizers) == len(trainer_b.optimizers), "Different number of optimizers"

    for opt_a, opt_b in zip(trainer_b.optimizers, trainer_b.optimizers):
        # Check optimizer class
        assert type(opt_a) is type(opt_b), f"Optimizer types differ: {type(opt_a)} vs {type(opt_b)}"
        assert opt_a.state_dict() == opt_b.state_dict(), "optimizers state dict differs"


class TestTransformerModelBase:
    def setup_method(self) -> None:
        torch.use_deterministic_algorithms(True)

    @pytest.fixture
    def trainer(self) -> Trainer:
        return Trainer(
            max_epochs=3, min_epochs=3, deterministic=True, accelerator="cpu", enable_checkpointing=False, devices=1
        )

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def dataset_item_features(self) -> Dataset:
        item_features = pd.DataFrame(
            [
                [12, "f1", "f1val1"],
                [12, "f2", "f2val2"],
                [13, "f1", "f1val1"],
                [13, "f2", "f2val3"],
                [14, "f1", "f1val2"],
                [14, "f2", "f2val1"],
                [15, "f1", "f1val2"],
                [15, "f2", "f2val2"],
                [17, "f1", "f1val2"],
                [17, "f2", "f2val3"],
                [16, "f1", "f1val2"],
                [16, "f2", "f2val3"],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
                [17, "f3", 5],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("default_trainer", (True, False))
    def test_save_load_for_unfitted_model(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset: Dataset,
        default_trainer: bool,
    ) -> None:
        config: tp.Dict[str, tp.Any] = {"deterministic": True}
        if not default_trainer:
            config["get_trainer_func"] = custom_trainer
        model = model_cls.from_config(config)

        with NamedTemporaryFile() as f:
            model.save(f.name)
            recovered_model = load_model(f.name)

        assert isinstance(recovered_model, model_cls)
        original_model_config = model.get_config()
        recovered_model_config = recovered_model.get_config()
        assert recovered_model_config == original_model_config

        seed_everything(32, workers=True)
        model.fit(dataset)
        seed_everything(32, workers=True)
        recovered_model.fit(dataset)

        self._assert_same_reco(model, recovered_model, dataset)

    def _assert_same_reco(self, model_1: TransformerModelBase, model_2: TransformerModelBase, dataset: Dataset) -> None:
        users = dataset.user_id_map.external_ids[:2]
        original_reco = model_1.recommend(users=users, dataset=dataset, k=2, filter_viewed=False)
        recovered_reco = model_2.recommend(users=users, dataset=dataset, k=2, filter_viewed=False)
        pd.testing.assert_frame_equal(original_reco, recovered_reco)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("default_trainer", (True, False))
    def test_save_load_for_fitted_model(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset_item_features: Dataset,
        default_trainer: bool,
    ) -> None:
        config: tp.Dict[str, tp.Any] = {"deterministic": True}
        if not default_trainer:
            config["get_trainer_func"] = custom_trainer
        model = model_cls.from_config(config)
        model.fit(dataset_item_features)
        assert_save_load_do_not_change_model(model, dataset_item_features)

    @pytest.mark.parametrize("test_dataset", ("dataset", "dataset_item_features"))
    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize(
        "map_location",
        (
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(torch.cuda.is_available() is False, reason="GPU is not available"),
            ),
            None,
        ),
    )
    @pytest.mark.parametrize(
        "model_params_update",
        (
            {
                "get_val_mask_func": "tests.models.nn.transformers.utils.leave_one_out_mask",
                "get_trainer_func": "tests.models.nn.transformers.utils.custom_trainer",
            },
            {
                "get_val_mask_func": None,
                "get_trainer_func": None,
            },
            None,
        ),
    )
    def test_load_from_checkpoint(
        self,
        model_cls: tp.Type[TransformerModelBase],
        test_dataset: str,
        map_location: tp.Optional[tp.Union[str, torch.device]],
        model_params_update: tp.Optional[tp.Dict[str, tp.Any]],
        request: FixtureRequest,
    ) -> None:

        model = model_cls.from_config(
            {
                "deterministic": True,
                "get_trainer_func": custom_trainer_ckpt,
            }
        )
        dataset = request.getfixturevalue(test_dataset)
        model.fit(dataset)

        assert model.fit_trainer is not None
        if model.fit_trainer.log_dir is None:
            raise ValueError("No log dir")
        ckpt_path = os.path.join(model.fit_trainer.log_dir, "checkpoints", "last_epoch.ckpt")
        assert os.path.isfile(ckpt_path)
        recovered_model = model_cls.load_from_checkpoint(
            ckpt_path, map_location=map_location, model_params_update=model_params_update
        )
        assert isinstance(recovered_model, model_cls)

        self._assert_same_reco(model, recovered_model, dataset)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_load_weights_from_checkpoint(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset: Dataset,
    ) -> None:

        model = model_cls.from_config(
            {
                "deterministic": True,
                "get_trainer_func": custom_trainer_multiple_ckpt,
            }
        )
        model.fit(dataset)
        assert model.fit_trainer is not None
        if model.fit_trainer.log_dir is None:
            raise ValueError("No log dir")
        ckpt_path = os.path.join(model.fit_trainer.log_dir, "checkpoints", "epoch=1.ckpt")
        assert os.path.isfile(ckpt_path)

        recovered_model = model_cls.load_from_checkpoint(ckpt_path)
        model.load_weights_from_checkpoint(ckpt_path)

        self._assert_same_reco(model, recovered_model, dataset)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_raises_when_load_weights_from_checkpoint_not_fitted_model(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset: Dataset,
    ) -> None:
        model = model_cls.from_config(
            {
                "deterministic": True,
                "get_trainer_func": custom_trainer_ckpt,
            }
        )
        model.fit(dataset)
        assert model.fit_trainer is not None
        if model.fit_trainer.log_dir is None:
            raise ValueError("No log dir")
        ckpt_path = os.path.join(model.fit_trainer.log_dir, "checkpoints", "last_epoch.ckpt")

        model_unfitted = model_cls.from_config(
            {
                "deterministic": True,
                "get_trainer_func": custom_trainer_ckpt,
            }
        )
        with pytest.raises(RuntimeError):
            model_unfitted.load_weights_from_checkpoint(ckpt_path)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("verbose", (1, 0))
    @pytest.mark.parametrize(
        "is_val_mask_func, expected_columns",
        (
            (False, ["epoch", "step", "train_loss"]),
            (True, ["epoch", "step", "train_loss", "val_loss"]),
        ),
    )
    @pytest.mark.parametrize("loss", ("softmax", "BCE", "gBCE", "sampled_softmax"))
    def test_log_metrics(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset: Dataset,
        tmp_path: str,
        verbose: int,
        loss: str,
        is_val_mask_func: bool,
        expected_columns: tp.List[str],
    ) -> None:
        logger = CSVLogger(save_dir=tmp_path)
        trainer = Trainer(
            default_root_dir=tmp_path,
            max_epochs=2,
            min_epochs=2,
            deterministic=True,
            accelerator="cpu",
            devices=1,
            logger=logger,
            enable_checkpointing=False,
        )
        get_val_mask_func = leave_one_out_mask if is_val_mask_func else None
        model = model_cls.from_config(
            {
                "verbose": verbose,
                "get_val_mask_func": get_val_mask_func,
                "loss": loss,
            }
        )
        model._trainer = trainer  # pylint: disable=protected-access
        model.fit(dataset=dataset)

        assert model.fit_trainer is not None
        assert model.fit_trainer.logger is not None
        assert model.fit_trainer.log_dir is not None
        has_val_mask_func = model.get_val_mask_func is not None
        assert has_val_mask_func is is_val_mask_func

        metrics_path = os.path.join(model.fit_trainer.log_dir, "metrics.csv")
        assert os.path.isfile(metrics_path)

        actual_columns = list(pd.read_csv(metrics_path).columns)
        assert actual_columns == expected_columns

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_fit_partial(
        self,
        dataset: Dataset,
        model_cls: tp.Type[TransformerModelBase],
    ) -> None:

        class FixSeedLightningModule(TransformerLightningModule):
            def on_train_epoch_start(self) -> None:
                seed_everything(32, workers=True)

        seed_everything(32, workers=True)
        model_1 = model_cls.from_config(
            {
                "epochs": 3,
                "data_preparator_kwargs": {"shuffle_train": False},
                "get_trainer_func": custom_trainer,
                "lightning_module_type": FixSeedLightningModule,
            }
        )
        model_1.fit(dataset)

        seed_everything(32, workers=True)
        model_2 = model_cls.from_config(
            {
                "data_preparator_kwargs": {"shuffle_train": False},
                "get_trainer_func": custom_trainer,
                "lightning_module_type": FixSeedLightningModule,
            }
        )
        model_2.fit_partial(dataset, min_epochs=2, max_epochs=2)
        model_2.fit_partial(dataset, min_epochs=1, max_epochs=1)

        self._assert_same_reco(model_1, model_2, dataset)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_fit_partial_from_checkpoint(
        self,
        dataset: Dataset,
        model_cls: tp.Type[TransformerModelBase],
    ) -> None:
        fit_partial_model = model_cls.from_config(
            {"data_preparator_kwargs": {"shuffle_train": False}, "get_trainer_func": custom_trainer_ckpt}
        )
        fit_partial_model.fit_partial(dataset, min_epochs=1, max_epochs=1)

        assert fit_partial_model.fit_trainer is not None
        if fit_partial_model.fit_trainer.log_dir is None:
            raise ValueError("No log dir")
        ckpt_path = os.path.join(fit_partial_model.fit_trainer.log_dir, "checkpoints", "last_epoch.ckpt")
        assert os.path.isfile(ckpt_path)
        recovered_fit_partial_model = model_cls.load_from_checkpoint(ckpt_path)

        seed_everything(32, workers=True)
        fit_partial_model.fit_partial(dataset, min_epochs=1, max_epochs=1)

        seed_everything(32, workers=True)
        recovered_fit_partial_model.fit_partial(dataset, min_epochs=1, max_epochs=1)

        self._assert_same_reco(fit_partial_model, recovered_fit_partial_model, dataset)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_raises_when_incorrect_similarity_dist(
        self, model_cls: tp.Type[TransformerModelBase], dataset: Dataset
    ) -> None:
        model_config = {
            "similarity_module_kwargs": {"distance": "euclidean"},
        }
        with pytest.raises(ValueError):
            model = model_cls.from_config(model_config)
            model.fit(dataset=dataset)

    @pytest.mark.parametrize("fit", (True, False))
    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("default_trainer", (True, False))
    def test_resaving(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset: Dataset,
        default_trainer: bool,
        fit: bool,
    ) -> None:
        config: tp.Dict[str, tp.Any] = {"deterministic": True}
        if not default_trainer:
            config["get_trainer_func"] = custom_trainer
        model = model_cls.from_config(config)

        seed_everything(32, workers=True)
        if fit:
            model.fit(dataset)

        with NamedTemporaryFile() as f:
            model.save(f.name)
            recovered_model = model_cls.load(f.name)

        with NamedTemporaryFile() as f:
            recovered_model.save(f.name)
            second_recovered_model = model_cls.load(f.name)

        assert isinstance(recovered_model, model_cls)

        original_model_config = model.get_config()
        second_recovered_model_config = recovered_model.get_config()
        assert second_recovered_model_config == original_model_config

        if fit:
            assert_pl_models_equal(model.lightning_model, second_recovered_model.lightning_model)

    # check if trainer keep state on multiple call partial fit
    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_fit_partial_multiple_times(
        self,
        dataset: Dataset,
        model_cls: tp.Type[TransformerModelBase],
    ) -> None:
        class FixSeedLightningModule(TransformerLightningModule):
            def on_train_epoch_start(self) -> None:
                seed_everything(32, workers=True)

        seed_everything(32, workers=True)
        model = model_cls.from_config(
            {
                "epochs": 3,
                "data_preparator_kwargs": {"shuffle_train": False},
                "get_trainer_func": custom_trainer,
                "lightning_module_type": FixSeedLightningModule,
            }
        )
        model.fit_partial(dataset, min_epochs=1, max_epochs=1)
        t1 = deepcopy(model.fit_trainer)
        model.fit_partial(
            Dataset.construct(pd.DataFrame(columns=Columns.Interactions)),
            min_epochs=1,
            max_epochs=1,
        )
        t2 = deepcopy(model.fit_trainer)

        assert t1 is not None
        assert t2 is not None
        assert_pl_trainers_equal(t1, t2)
