import os
import typing as tp
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import BERT4RecModel, SASRecModel, load_model
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.models.nn.transformer_base import TransformerModelBase
from tests.models.utils import assert_save_load_do_not_change_model

from .utils import custom_trainer, leave_one_out_mask


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

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("default_trainer", (True, False))
    def test_save_load_for_unfitted_model(
        self, model_cls: tp.Type[TransformerModelBase], dataset: Dataset, default_trainer: bool, trainer: Trainer
    ) -> None:
        config = {
            "deterministic": True,
            "item_net_block_types": (IdEmbeddingsItemNet,),  # TODO: add CatFeaturesItemNet
        }
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
        self, model_cls: tp.Type[TransformerModelBase], dataset: Dataset, default_trainer: bool, trainer: Trainer
    ) -> None:
        config = {
            "deterministic": True,
            "item_net_block_types": (IdEmbeddingsItemNet,),  # TODO: add CatFeaturesItemNet
        }
        if not default_trainer:
            config["get_trainer_func"] = custom_trainer
        model = model_cls.from_config(config)
        model.fit(dataset)
        assert_save_load_do_not_change_model(model, dataset)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    def test_load_from_checkpoint(
        self,
        model_cls: tp.Type[TransformerModelBase],
        tmp_path: str,
        dataset: Dataset,
    ) -> None:
        model = model_cls.from_config(
            {
                "deterministic": True,
                "item_net_block_types": (IdEmbeddingsItemNet,),  # TODO: add CatFeaturesItemNet
            }
        )
        model._trainer = Trainer(  # pylint: disable=protected-access
            default_root_dir=tmp_path,
            max_epochs=2,
            min_epochs=2,
            deterministic=True,
            accelerator="cpu",
            devices=1,
            callbacks=ModelCheckpoint(filename="last_epoch"),
        )
        model.fit(dataset)

        assert model.fit_trainer is not None
        if model.fit_trainer.log_dir is None:
            raise ValueError("No log dir")
        ckpt_path = os.path.join(model.fit_trainer.log_dir, "checkpoints", "last_epoch.ckpt")
        assert os.path.isfile(ckpt_path)
        recovered_model = model_cls.load_from_checkpoint(ckpt_path)
        assert isinstance(recovered_model, model_cls)

        self._assert_same_reco(model, recovered_model, dataset)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("verbose", (1, 0))
    @pytest.mark.parametrize(
        "is_val_mask_func, expected_columns",
        (
            (False, ["epoch", "step", "train_loss"]),
            (True, ["epoch", "step", "train_loss", "val_loss"]),
        ),
    )
    def test_log_metrics(
        self,
        model_cls: tp.Type[TransformerModelBase],
        dataset: Dataset,
        tmp_path: str,
        verbose: int,
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
