import typing as tp
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import BERT4RecModel, SASRecModel, load_model
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.models.nn.transformer_base import TransformerModelBase
from tests.models.utils import assert_save_load_do_not_change_model


class TestTransformerModelBase:
    def setup_method(self) -> None:
        torch.use_deterministic_algorithms(True)

    @pytest.fixture
    def trainer(self) -> Trainer:
        return Trainer(max_epochs=2, min_epochs=2, deterministic=True, accelerator="cpu", enable_checkpointing=False)

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
        if default_trainer:
            trainer = None
        seed_everything(32, workers=True)
        model = model_cls(
            trainer=trainer,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),  # TODO: add CatFeaturesItemNet
        )
        with NamedTemporaryFile() as f:
            model.save(f.name)
            seed_everything(32, workers=True)
            recovered_model = load_model(f.name)

        assert isinstance(recovered_model, model_cls)
        original_model_config = model.get_config()
        recovered_model_config = recovered_model.get_config()
        assert recovered_model_config == original_model_config

        seed_everything(32, workers=True)
        model.fit(dataset)
        seed_everything(32, workers=True)
        recovered_model.fit(dataset)

        users = dataset.user_id_map.external_ids[:2]
        original_reco = model.recommend(users=users, dataset=dataset, k=2, filter_viewed=False)
        recovered_reco = recovered_model.recommend(users=users, dataset=dataset, k=2, filter_viewed=False)
        pd.testing.assert_frame_equal(original_reco, recovered_reco)

    @pytest.mark.parametrize("model_cls", (SASRecModel, BERT4RecModel))
    @pytest.mark.parametrize("default_trainer", (True, False))
    def test_save_load_for_fitted_model(
        self, model_cls: tp.Type[TransformerModelBase], dataset: Dataset, default_trainer: bool, trainer: Trainer
    ) -> None:
        if default_trainer:
            trainer = None
        model = model_cls(
            trainer=trainer,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),  # TODO: add CatFeaturesItemNet
        )
        model.fit(dataset)
        assert_save_load_do_not_change_model(model, dataset)
