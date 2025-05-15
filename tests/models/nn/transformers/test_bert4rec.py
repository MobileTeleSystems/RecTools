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

# pylint: disable=too-many-lines
import typing as tp
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything

from rectools import ExternalIds
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.models import BERT4RecModel
from rectools.models.nn.item_net import IdEmbeddingsItemNet, SumOfEmbeddingsConstructor
from rectools.models.nn.transformers.base import (
    LearnableInversePositionalEncoding,
    PreLNTransformerLayers,
    TrainerCallable,
    TransformerLightningModule,
)
from rectools.models.nn.transformers.bert4rec import MASKING_VALUE, BERT4RecDataPreparator, ValMaskCallable
from rectools.models.nn.transformers.data_preparator import InitKwargs
from rectools.models.nn.transformers.negative_sampler import CatalogUniformSampler, TransformerNegativeSamplerBase
from rectools.models.nn.transformers.similarity import DistanceSimilarityModule
from rectools.models.nn.transformers.torch_backbone import TransformerTorchBackbone
from tests.models.data import DATASET
from tests.models.utils import (
    assert_default_config_and_default_model_params_are_the_same,
    assert_second_fit_refits_model,
)

from .utils import custom_trainer, leave_one_out_mask


class TestBERT4RecModel:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

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
    def dataset_hot_users_items(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df[:-4])

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
    def get_trainer_func(self) -> TrainerCallable:
        def get_trainer() -> Trainer:
            return Trainer(
                max_epochs=2,
                min_epochs=2,
                deterministic=True,
                accelerator="cpu",
                enable_checkpointing=False,
                devices=1,
            )

        return get_trainer

    @pytest.fixture
    def get_custom_trainer_func(self) -> TrainerCallable:
        def get_trainer_func(max_epochs: int, accelerator: str) -> Trainer:
            return Trainer(
                max_epochs=max_epochs,
                min_epochs=2,
                deterministic=True,
                accelerator=accelerator,
                enable_checkpointing=False,
                devices=1,
            )

        return get_trainer_func

    @pytest.fixture
    def get_custom_val_mask_func(self) -> ValMaskCallable:
        def get_val_mask_func(interactions: pd.DataFrame, val_users: tp.List[int]) -> np.ndarray:
            rank = (
                interactions.sort_values(Columns.Datetime, ascending=False, kind="stable")
                .groupby(Columns.User, sort=False)
                .cumcount()
                + 1
            )
            val_mask = (interactions[Columns.User].isin(val_users)) & (rank <= 1)
            return val_mask.values

        return get_val_mask_func

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
            ("cpu", 2, "cpu"),
            pytest.param(
                "gpu",
                1,
                "cpu",
                marks=pytest.mark.skipif(torch.cuda.is_available() is False, reason="GPU is not available"),
            ),
            pytest.param(
                "gpu",
                1,
                "cuda",
                marks=pytest.mark.skipif(torch.cuda.is_available() is False, reason="GPU is not available"),
            ),
            pytest.param(
                "gpu",
                2,
                "cpu",
                marks=pytest.mark.skipif(
                    torch.cuda.is_available() is False or torch.cuda.device_count() < 2,
                    reason="GPU is not available or there is only one gpu device",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "filter_viewed,expected_cpu_1,expected_cpu_2,expected_gpu_1,expected_gpu_2",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
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
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [12, 13, 11, 12, 13, 11, 12, 13, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [12, 13, 11, 12, 13, 11, 12, 13, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [12, 13, 11, 13, 12, 11, 12, 13, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [12, 13, 11, 13, 12, 11, 12, 13, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    @pytest.mark.parametrize("u2i_dist", ("dot", "cosine"))
    def test_u2i(
        self,
        dataset_devices: Dataset,
        filter_viewed: bool,
        accelerator: str,
        n_devices: int,
        recommend_torch_device: str,
        expected_cpu_1: pd.DataFrame,
        expected_cpu_2: pd.DataFrame,
        expected_gpu_1: pd.DataFrame,
        expected_gpu_2: pd.DataFrame,
        u2i_dist: str,
    ) -> None:
        if n_devices != 1:
            pytest.skip("DEBUG: skipping multi-device tests")

        def get_trainer() -> Trainer:
            return Trainer(
                max_epochs=2,
                min_epochs=2,
                deterministic=True,
                devices=n_devices,
                accelerator=accelerator,
                enable_checkpointing=False,
            )

        model = BERT4RecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            recommend_torch_device=recommend_torch_device,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer,
            similarity_module_type=DistanceSimilarityModule,
            similarity_module_kwargs={"distance": u2i_dist},
        )
        model.fit(dataset=dataset_devices)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset_devices, k=3, filter_viewed=filter_viewed)
        if accelerator == "cpu" and n_devices == 1:
            expected = expected_cpu_1
        elif accelerator == "cpu" and n_devices == 2:
            expected = expected_cpu_2
        elif accelerator == "gpu" and n_devices == 1:
            expected = expected_gpu_1
        else:
            expected = expected_gpu_2
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "loss,expected",
        (
            (
                "BCE",
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                "gBCE",
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                "sampled_softmax",
                pd.DataFrame(
                    {
                        Columns.User: [30, 40, 40],
                        Columns.Item: [12, 12, 13],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
        ),
    )
    @pytest.mark.parametrize("u2i_dist", ("dot", "cosine"))
    def test_u2i_losses(
        self,
        dataset_devices: Dataset,
        loss: str,
        get_trainer_func: TrainerCallable,
        expected: pd.DataFrame,
        u2i_dist: str,
    ) -> None:
        model = BERT4RecModel(
            n_negatives=2,
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            mask_prob=0.6,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            loss=loss,
            similarity_module_type=DistanceSimilarityModule,
            similarity_module_kwargs={"distance": u2i_dist},
        )
        model.fit(dataset=dataset_devices)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset_devices, k=3, filter_viewed=True)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [40],
                        Columns.Item: [13],
                        Columns.Rank: [1],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 40, 40],
                        Columns.Item: [13, 11, 13, 11, 13, 11],
                        Columns.Rank: [1, 2, 1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(
        self,
        dataset_devices: Dataset,
        get_trainer_func: TrainerCallable,
        filter_viewed: bool,
        expected: pd.DataFrame,
    ) -> None:
        model = BERT4RecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset_devices)
        users = np.array([10, 30, 40])
        items_to_recommend = np.array([11, 13, 17])
        actual = model.recommend(
            users=users,
            dataset=dataset_devices,
            k=3,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [12, 17, 11, 14, 11, 15, 17, 12, 14],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [17, 11, 14, 11, 15, 17, 12, 14, 15],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                np.array([15, 13, 14]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 17, 17, 17],
                        Columns.Item: [14, 13, 15, 15, 13, 14, 15, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self,
        dataset: Dataset,
        get_trainer_func: TrainerCallable,
        filter_itself: bool,
        whitelist: tp.Optional[np.ndarray],
        expected: pd.DataFrame,
    ) -> None:

        model = BERT4RecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset)
        target_items = np.array([12, 14, 17])
        actual = model.recommend_to_items(
            target_items=target_items,
            dataset=dataset,
            k=3,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, dataset_hot_users_items: Dataset) -> None:
        model = BERT4RecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=custom_trainer,
            similarity_module_type=DistanceSimilarityModule,
        )
        assert_second_fit_refits_model(model, dataset_hot_users_items, pre_fit_callback=self._seed_everything)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20],
                        Columns.Item: [12, 11],
                        Columns.Rank: [1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [12, 13, 11],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_recommend_for_cold_user_with_hot_item(
        self, dataset_devices: Dataset, get_trainer_func: TrainerCallable, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = BERT4RecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset_devices)
        users = np.array([20])
        actual = model.recommend(
            users=users,
            dataset=dataset_devices,
            k=3,
            filter_viewed=filter_viewed,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "get_custom_trainer_func_kwargs, get_custom_val_mask_func_kwargs",
        (
            pytest.param(
                {
                    "max_epochs": 2,
                    "accelerator": "cpu",
                },
                {"val_users": [30, 40]},
                id="cpu_config",
            ),
            pytest.param(
                {
                    "max_epochs": 3,
                    "accelerator": "gpu",
                },
                {"val_users": [20, 30]},
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available"),
                id="gpu_config",
            ),
        ),
    )
    def test_customized_happy_path(
        self,
        dataset_devices: Dataset,
        get_custom_trainer_func: TrainerCallable,
        get_custom_val_mask_func: ValMaskCallable,
        get_custom_trainer_func_kwargs: InitKwargs,
        get_custom_val_mask_func_kwargs: InitKwargs,
    ) -> None:
        class NextActionDataPreparator(BERT4RecDataPreparator):
            def __init__(
                self,
                session_max_len: int,
                n_negatives: int,
                batch_size: int,
                dataloader_num_workers: int,
                train_min_user_interactions: int,
                mask_prob: float = 0.15,
                negative_sampler: tp.Optional[TransformerNegativeSamplerBase] = None,
                shuffle_train: bool = True,
                get_val_mask_func: tp.Optional[ValMaskCallable] = None,
                get_val_mask_func_kwargs: tp.Optional[InitKwargs] = None,
                n_last_targets: int = 1,  # custom kwarg
            ) -> None:
                super().__init__(
                    session_max_len=session_max_len,
                    n_negatives=n_negatives,
                    batch_size=batch_size,
                    dataloader_num_workers=dataloader_num_workers,
                    train_min_user_interactions=train_min_user_interactions,
                    negative_sampler=negative_sampler,
                    shuffle_train=shuffle_train,
                    get_val_mask_func=get_custom_val_mask_func,
                    get_val_mask_func_kwargs=get_custom_val_mask_func_kwargs,
                    mask_prob=mask_prob,
                )
                self.n_last_targets = n_last_targets

            def _collate_fn_train(
                self,
                batch: tp.List[tp.Tuple[tp.List[int], tp.List[float]]],
            ) -> tp.Dict[str, torch.Tensor]:
                batch_size = len(batch)
                x = np.zeros((batch_size, self.session_max_len))
                y = np.zeros((batch_size, self.session_max_len))
                yw = np.zeros((batch_size, self.session_max_len))
                for i, (ses, ses_weights) in enumerate(batch):
                    y[i, -self.n_last_targets] = ses[-self.n_last_targets]
                    yw[i, -self.n_last_targets] = ses_weights[-self.n_last_targets]
                    x[i, -len(ses) :] = ses
                    x[i, -self.n_last_targets] = self.extra_token_ids[MASKING_VALUE]  # Replace last tokens with "MASK"
                batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
                if self.negative_sampler is not None:
                    batch_dict["negatives"] = self.negative_sampler.get_negatives(
                        batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size
                    )
                return batch_dict

        model = BERT4RecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=4,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_custom_trainer_func,
            get_trainer_func_kwargs=get_custom_trainer_func_kwargs,
            data_preparator_type=NextActionDataPreparator,
            data_preparator_kwargs={"n_last_targets": 1},
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset_devices)

        assert model.data_preparator.n_last_targets == 1  # type: ignore

        users = np.array([10, 30, 40])
        items_to_recommend = np.array([11, 13, 17])
        actual = model.recommend(
            users=users,
            dataset=dataset_devices,
            k=3,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
        )
        expected = pd.DataFrame(
            {
                Columns.User: [10, 10, 30, 30, 40, 40],
                Columns.Item: [13, 11, 13, 11, 13, 11],
                Columns.Rank: [1, 2, 1, 2, 1, 2],
            }
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )


class TestBERT4RecDataPreparator:

    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset(self) -> Dataset:
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
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def dataset_one_session(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [10, 1, 1, "2021-11-30"],
                [10, 2, 1, "2021-11-30"],
                [10, 3, 1, "2021-11-30"],
                [10, 4, 1, "2021-11-30"],
                [10, 5, 1, "2021-11-30"],
                [10, 6, 1, "2021-11-30"],
                [10, 7, 1, "2021-11-30"],
                [10, 8, 1, "2021-11-30"],
                [10, 9, 1, "2021-11-30"],
                [10, 13, 1, "2021-11-30"],
                [10, 2, 1, "2021-11-30"],
                [10, 3, 1, "2021-11-30"],
                [10, 3, 1, "2021-11-30"],
                [10, 4, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-30"],
            ],
            columns=Columns.Interactions,
        )
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def data_preparator(self) -> BERT4RecDataPreparator:
        return BERT4RecDataPreparator(
            session_max_len=4,
            n_negatives=1,
            batch_size=4,
            dataloader_num_workers=0,
            train_min_user_interactions=2,
            mask_prob=0.5,
        )

    @pytest.fixture
    def data_preparator_val_mask(self) -> BERT4RecDataPreparator:
        def get_val_mask(interactions: pd.DataFrame, val_users: ExternalIds) -> np.ndarray:
            rank = (
                interactions.sort_values(Columns.Datetime, ascending=False, kind="stable")
                .groupby(Columns.User, sort=False)
                .cumcount()
                + 1
            )
            val_mask = (interactions[Columns.User].isin(val_users)) & (rank <= 1)
            return val_mask.values

        val_users = [10, 30]
        get_val_mask_func = partial(get_val_mask, val_users=val_users)
        return BERT4RecDataPreparator(
            session_max_len=4,
            n_negatives=2,
            train_min_user_interactions=2,
            mask_prob=0.5,
            batch_size=4,
            dataloader_num_workers=0,
            get_val_mask_func=get_val_mask_func,
        )

    @pytest.mark.parametrize(
        "train_batch",
        (
            (
                {
                    "x": torch.tensor([[6, 1, 4, 7], [0, 2, 4, 1], [0, 0, 3, 5]]),
                    "y": torch.tensor([[0, 3, 0, 0], [0, 0, 0, 3], [0, 0, 0, 0]]),
                    "yw": torch.tensor([[1, 1, 1, 1], [0, 1, 2, 1], [0, 0, 1, 1]], dtype=torch.float),
                    "negatives": torch.tensor([[[6], [2], [2], [7]], [[4], [5], [6], [3]], [[5], [3], [6], [7]]]),
                }
            ),
        ),
    )
    def test_get_dataloader_train(
        self, dataset: Dataset, data_preparator: BERT4RecDataPreparator, train_batch: tp.List
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataloader = data_preparator.get_dataloader_train()
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, train_batch[key])

    @pytest.mark.parametrize(
        "train_batch",
        (
            (
                {
                    "x": torch.tensor([[2, 1, 4, 5, 6, 7, 1, 9, 10, 11, 1, 1, 4, 6, 12]]),
                    "y": torch.tensor([[0, 3, 0, 0, 0, 0, 8, 0, 0, 0, 3, 4, 0, 5, 0]]),
                    "yw": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float),
                }
            ),
        ),
    )
    def test_get_dataloader_train_for_masked_session_with_random_replacement(
        self, dataset_one_session: Dataset, train_batch: tp.List
    ) -> None:
        data_preparator = BERT4RecDataPreparator(
            session_max_len=15,
            n_negatives=None,
            batch_size=14,
            dataloader_num_workers=0,
            train_min_user_interactions=2,
            mask_prob=0.5,
        )
        data_preparator.process_dataset_train(dataset_one_session)
        dataloader = data_preparator.get_dataloader_train()
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, train_batch[key])

    @pytest.mark.parametrize(
        "recommend_batch",
        (({"x": torch.tensor([[3, 4, 7, 1], [2, 4, 3, 1], [0, 3, 5, 1], [0, 0, 7, 1]])}),),
    )
    def test_get_dataloader_recommend(
        self, dataset: Dataset, data_preparator: BERT4RecDataPreparator, recommend_batch: torch.Tensor
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataset = data_preparator.transform_dataset_i2i(dataset)
        dataloader = data_preparator.get_dataloader_recommend(dataset, 4)
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, recommend_batch[key])

    @pytest.mark.parametrize(
        "val_batch",
        (
            (
                {
                    "x": torch.tensor([[0, 2, 4, 1]]),
                    "y": torch.tensor([[3]]),
                    "yw": torch.tensor([[1.0]]),
                    "negatives": torch.tensor([[[5, 2]]]),
                }
            ),
        ),
    )
    def test_get_dataloader_val(
        self, dataset: Dataset, data_preparator_val_mask: BERT4RecDataPreparator, val_batch: tp.List
    ) -> None:
        data_preparator_val_mask.process_dataset_train(dataset)
        dataloader = data_preparator_val_mask.get_dataloader_val()
        actual = next(iter(dataloader))  # type: ignore
        for key, value in actual.items():
            assert torch.equal(value, val_batch[key])

    @pytest.mark.parametrize(
        "val_batch, val_users",
        (
            (
                {
                    "x": torch.tensor([[0, 2, 4, 1]]),
                    "y": torch.tensor([[3]]),
                    "yw": torch.tensor([[1.0]]),
                    "negatives": torch.tensor([[[5, 2]]]),
                },
                [10, 30],
            ),
            (
                {
                    "x": torch.tensor([[0, 2, 4, 1]]),
                    "y": torch.tensor([[3]]),
                    "yw": torch.tensor([[1.0]]),
                    "negatives": torch.tensor([[[5, 2]]]),
                },
                [30],
            ),
        ),
    )
    def test_get_dataloader_val_with_kwargs(
        self,
        dataset: Dataset,
        val_batch: tp.Dict[tp.Any, tp.Any],
        val_users: tp.List,
    ) -> None:

        def get_custom_val_mask_func(interactions: pd.DataFrame, val_users: tp.List[int]) -> np.ndarray:
            rank = (
                interactions.sort_values(Columns.Datetime, ascending=False, kind="stable")
                .groupby(Columns.User, sort=False)
                .cumcount()
                + 1
            )
            val_mask = (interactions[Columns.User].isin(val_users)) & (rank <= 1)
            return val_mask.values

        get_custom_val_mask_func_kwargs = {"val_users": val_users}
        data_preparator_val_mask = BERT4RecDataPreparator(
            session_max_len=4,
            n_negatives=2,
            train_min_user_interactions=2,
            mask_prob=0.5,
            batch_size=4,
            dataloader_num_workers=0,
            get_val_mask_func=get_custom_val_mask_func,
            get_val_mask_func_kwargs=get_custom_val_mask_func_kwargs,
        )
        data_preparator_val_mask.process_dataset_train(dataset)
        dataloader = data_preparator_val_mask.get_dataloader_val()
        actual = next(iter(dataloader))  # type: ignore
        for key, value in actual.items():
            assert torch.equal(value, val_batch[key])


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
            "recommend_torch_device": None,
            "recommend_batch_size": 256,
            "train_min_user_interactions": 2,
            "item_net_block_types": (IdEmbeddingsItemNet,),
            "item_net_constructor_type": SumOfEmbeddingsConstructor,
            "pos_encoding_type": LearnableInversePositionalEncoding,
            "transformer_layers_type": PreLNTransformerLayers,
            "data_preparator_type": BERT4RecDataPreparator,
            "lightning_module_type": TransformerLightningModule,
            "negative_sampler_type": CatalogUniformSampler,
            "similarity_module_type": DistanceSimilarityModule,
            "backbone_type": TransformerTorchBackbone,
            "mask_prob": 0.15,
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
        model = BERT4RecModel.from_config(initial_config)

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
        model = BERT4RecModel(**config)
        actual = model.get_config(simple_types=simple_types)

        expected = config.copy()
        expected["cls"] = BERT4RecModel

        if simple_types:
            simple_types_params = {
                "cls": "BERT4RecModel",
                "item_net_block_types": ["rectools.models.nn.item_net.IdEmbeddingsItemNet"],
                "item_net_constructor_type": "rectools.models.nn.item_net.SumOfEmbeddingsConstructor",
                "pos_encoding_type": "rectools.models.nn.transformers.net_blocks.LearnableInversePositionalEncoding",
                "transformer_layers_type": "rectools.models.nn.transformers.net_blocks.PreLNTransformerLayers",
                "data_preparator_type": "rectools.models.nn.transformers.bert4rec.BERT4RecDataPreparator",
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
        simple_types: bool,
        initial_config: tp.Dict[str, tp.Any],
        use_custom_trainer: bool,
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
        if use_custom_trainer:
            config["get_trainer_func"] = custom_trainer

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
