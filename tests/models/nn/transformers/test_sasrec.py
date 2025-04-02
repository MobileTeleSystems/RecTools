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
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.models import SASRecModel
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet, SumOfEmbeddingsConstructor
from rectools.models.nn.transformers.base import (
    LearnableInversePositionalEncoding,
    TrainerCallable,
    TransformerLightningModule,
    TransformerTorchBackbone,
)
from rectools.models.nn.transformers.sasrec import SASRecDataPreparator, SASRecTransformerLayers
from rectools.models.nn.transformers.similarity import DistanceSimilarityModule
from tests.models.data import DATASET
from tests.models.utils import (
    assert_default_config_and_default_model_params_are_the_same,
    assert_second_fit_refits_model,
)
from tests.testing_utils import assert_id_map_equal, assert_interactions_set_equal

from .utils import custom_trainer, leave_one_out_mask


class TestSASRecModel:
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
    def dataset_devices(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 13, 2, "2021-11-26"],
                [40, 11, 1, "2021-11-25"],
                [40, 14, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def dataset_item_features(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 13, 2, "2021-11-26"],
                [40, 11, 1, "2021-11-25"],
                [40, 14, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        item_features = pd.DataFrame(
            [
                [11, "f1", "f1val1"],
                [11, "f2", "f2val1"],
                [12, "f1", "f1val1"],
                [12, "f2", "f2val2"],
                [13, "f1", "f1val1"],
                [13, "f2", "f2val3"],
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            interactions_df,
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    @pytest.fixture
    def dataset_hot_users_items(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df[:-4])

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

    @pytest.mark.parametrize(
        "accelerator,devices,recommend_torch_device",
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
                [0, 1],
                "cpu",
                marks=pytest.mark.skipif(
                    torch.cuda.is_available() is False or torch.cuda.device_count() < 2,
                    reason="GPU is not available or there is only one gpu device",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "filter_viewed,expected_cpu_1,expected_cpu_2,expected_gpu",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [30, 30, 40, 40],
                        Columns.Item: [12, 14, 12, 13],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [30, 30, 40, 40],
                        Columns.Item: [14, 12, 13, 12],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [30, 30, 40, 40],
                        Columns.Item: [12, 14, 12, 13],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 12, 11, 11, 12, 14, 14, 11, 12],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 14, 11, 11, 14, 12, 14, 11, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [12, 13, 11, 11, 12, 14, 12, 14, 11],
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
        devices: tp.Union[int, tp.List[int]],
        recommend_torch_device: str,
        expected_cpu_1: pd.DataFrame,
        expected_cpu_2: pd.DataFrame,
        expected_gpu: pd.DataFrame,
        u2i_dist: str,
    ) -> None:

        if devices != 1:
            pytest.skip("DEBUG: skipping multi-device tests")

        def get_trainer() -> Trainer:
            return Trainer(
                max_epochs=2,
                min_epochs=2,
                deterministic=True,
                devices=devices,
                accelerator=accelerator,
                enable_checkpointing=False,
            )

        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=3,
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
        if accelerator == "cpu" and devices == 1:
            expected = expected_cpu_1
        elif accelerator == "cpu" and devices == 2:
            expected = expected_cpu_2
        else:
            expected = expected_gpu
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "loss,expected,u2i_dist",
        (
            (
                "BCE",
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 13, 17, 14, 13, 14, 15],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
                "dot",
            ),
            (
                "gBCE",
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 13, 17, 14, 13, 14, 15],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
                "dot",
            ),
            (
                "sampled_softmax",
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 13, 17, 14, 13, 14, 15],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
                "dot",
            ),
            (
                "BCE",
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 13, 14, 17, 13, 14, 15],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
                "cosine",
            ),
            (
                "gBCE",
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 13, 14, 17, 13, 14, 15],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
                "cosine",
            ),
            (
                "sampled_softmax",
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 13, 14, 17, 13, 14, 15],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
                "cosine",
            ),
        ),
    )
    def test_u2i_losses(
        self,
        dataset: Dataset,
        loss: str,
        get_trainer_func: TrainerCallable,
        expected: pd.DataFrame,
        u2i_dist: str,
    ) -> None:
        model = SASRecModel(
            n_negatives=2,
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            loss=loss,
            similarity_module_type=DistanceSimilarityModule,
            similarity_module_kwargs={"distance": u2i_dist},
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=True)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "expected",
        (
            pd.DataFrame(
                {
                    Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                    Columns.Item: [13, 17, 11, 11, 13, 15, 17, 13, 11],
                    Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                }
            ),
        ),
    )
    def test_u2i_with_key_and_attn_masks(
        self,
        dataset: Dataset,
        get_trainer_func: TrainerCallable,
        expected: pd.DataFrame,
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            use_key_padding_mask=True,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=False)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "expected",
        (
            pd.DataFrame(
                {
                    Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                    Columns.Item: [13, 12, 11, 11, 12, 13, 13, 14, 12],
                    Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                }
            ),
        ),
    )
    def test_u2i_with_item_features(
        self,
        dataset_item_features: Dataset,
        get_trainer_func: TrainerCallable,
        expected: pd.DataFrame,
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            n_heads=1,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet, CatFeaturesItemNet),
            get_trainer_func=get_trainer_func,
            use_key_padding_mask=True,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset_item_features)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset_item_features, k=3, filter_viewed=False)
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
                        Columns.User: [10, 30, 30, 40],
                        Columns.Item: [17, 13, 17, 13],
                        Columns.Rank: [1, 1, 2, 1],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 17, 11, 11, 13, 17, 17, 13, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(
        self,
        dataset: Dataset,
        get_trainer_func: TrainerCallable,
        filter_viewed: bool,
        expected: pd.DataFrame,
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        items_to_recommend = np.array([11, 13, 17])
        actual = model.recommend(
            users=users,
            dataset=dataset,
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
                        Columns.Item: [12, 13, 14, 14, 12, 15, 17, 13, 14],
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
                        Columns.Item: [13, 14, 11, 12, 15, 17, 13, 14, 11],
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
                        Columns.Item: [13, 14, 15, 15, 13, 13, 14, 15],
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
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
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
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
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
                        Columns.User: [20, 20, 20],
                        Columns.Item: [11, 12, 17],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [13, 11, 12],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_recommend_for_cold_user_with_hot_item(
        self, dataset: Dataset, get_trainer_func: TrainerCallable, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset)
        users = np.array([20])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
        )
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
                        Columns.User: [10, 10, 20, 20, 20],
                        Columns.Item: [17, 15, 11, 12, 17],
                        Columns.Rank: [1, 2, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 20, 20, 20],
                        Columns.Item: [13, 17, 11, 13, 11, 12],
                        Columns.Rank: [1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_warn_when_hot_user_has_cold_items_in_recommend(
        self, dataset: Dataset, get_trainer_func: TrainerCallable, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            get_trainer_func=get_trainer_func,
            similarity_module_type=DistanceSimilarityModule,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 20, 50])
        with pytest.warns() as record:
            actual = model.recommend(
                users=users,
                dataset=dataset,
                k=3,
                filter_viewed=filter_viewed,
                on_unsupported_targets="warn",
            )
            pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
            pd.testing.assert_frame_equal(
                actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
                actual,
            )
        assert str(record[0].message) == "1 target users were considered cold because of missing known items"
        assert str(record[1].message).startswith(
            """
                Model `<class 'rectools.models.nn.transformers.sasrec.SASRecModel'>` doesn't support"""
        )

    def test_raises_when_loss_is_not_supported(self, dataset: Dataset) -> None:
        model = SASRecModel(loss="gbce", similarity_module_type=DistanceSimilarityModule)
        with pytest.raises(ValueError):
            model.fit(dataset=dataset)

    def test_torch_model(self, dataset: Dataset) -> None:
        model = SASRecModel(similarity_module_type=DistanceSimilarityModule)
        model.fit(dataset)
        assert isinstance(model.torch_model, TransformerTorchBackbone)


class TestSASRecDataPreparator:

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
    def data_preparator(self) -> SASRecDataPreparator:
        return SASRecDataPreparator(session_max_len=3, batch_size=4, dataloader_num_workers=0)

    @pytest.fixture
    def data_preparator_val_mask(self) -> SASRecDataPreparator:
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
        return SASRecDataPreparator(
            session_max_len=3,
            batch_size=4,
            dataloader_num_workers=0,
            n_negatives=2,
            get_val_mask_func=get_val_mask_func,
        )

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_train_interactions, expected_val_interactions",
        (
            (
                IdMap.from_values([30, 40, 10]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 16, 14]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 1, 1.0, "2021-11-25"],
                            [1, 2, 1.0, "2021-11-25"],
                            [0, 3, 2.0, "2021-11-26"],
                            [1, 4, 1.0, "2021-11-26"],
                            [2, 5, 1.0, "2021-11-27"],
                            [2, 6, 1.0, "2021-11-28"],
                            [2, 2, 1.0, "2021-11-29"],
                            [2, 3, 1.0, "2021-11-29"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 1, 0.0, "2021-11-25"],
                            [0, 3, 0.0, "2021-11-26"],
                            [0, 2, 1.0, "2021-11-27"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_process_dataset_with_val_mask(
        self,
        dataset: Dataset,
        data_preparator_val_mask: SASRecDataPreparator,
        expected_train_interactions: Interactions,
        expected_val_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator_val_mask.process_dataset_train(dataset)
        actual_train_dataset = data_preparator_val_mask.train_dataset
        actual_val_interactions = data_preparator_val_mask.val_interactions
        assert_id_map_equal(actual_train_dataset.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual_train_dataset.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual_train_dataset.interactions, expected_train_interactions)
        pd.testing.assert_frame_equal(actual_val_interactions, expected_val_interactions.df)

    @pytest.mark.parametrize(
        "train_batch",
        (
            (
                {
                    "x": torch.tensor([[5, 2, 3], [0, 1, 3], [0, 0, 2]]),
                    "y": torch.tensor([[2, 3, 6], [0, 3, 2], [0, 0, 4]]),
                    "yw": torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
                    "negatives": torch.tensor([[[5], [1], [1]], [[6], [3], [4]], [[5], [2], [4]]]),
                }
            ),
        ),
    )
    def test_get_dataloader_train(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, train_batch: tp.List
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataloader = data_preparator.get_dataloader_train()
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, train_batch[key])

    @pytest.mark.parametrize(
        "val_batch",
        (
            (
                {
                    "x": torch.tensor([[0, 1, 3]]),
                    "y": torch.tensor([[2]]),
                    "yw": torch.tensor([[1.0]]),
                    "negatives": torch.tensor([[[4, 1]]]),
                }
            ),
        ),
    )
    def test_get_dataloader_val(
        self, dataset: Dataset, data_preparator_val_mask: SASRecDataPreparator, val_batch: tp.List
    ) -> None:
        data_preparator_val_mask.process_dataset_train(dataset)
        dataloader = data_preparator_val_mask.get_dataloader_val()
        actual = next(iter(dataloader))  # type: ignore
        for key, value in actual.items():
            assert torch.equal(value, val_batch[key])

    @pytest.mark.parametrize(
        "recommend_batch",
        (({"x": torch.tensor([[2, 3, 6], [1, 3, 2], [0, 2, 4], [0, 0, 6]])}),),
    )
    def test_get_dataloader_recommend(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, recommend_batch: torch.Tensor
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataset = data_preparator.transform_dataset_i2i(dataset)
        dataloader = data_preparator.get_dataloader_recommend(dataset, 4)
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, recommend_batch[key])


class TestSASRecModelConfiguration:
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
            "transformer_layers_type": SASRecTransformerLayers,
            "data_preparator_type": SASRecDataPreparator,
            "lightning_module_type": TransformerLightningModule,
            "similarity_module_type": DistanceSimilarityModule,
            "get_val_mask_func": leave_one_out_mask,
            "get_trainer_func": None,
            "data_preparator_kwargs": None,
            "transformer_layers_kwargs": None,
            "item_net_constructor_kwargs": None,
            "pos_encoding_kwargs": None,
            "lightning_module_kwargs": None,
            "similarity_module_kwargs": None,
        }
        return config

    @pytest.mark.parametrize("use_custom_trainer", (True, False))
    def test_from_config(self, initial_config: tp.Dict[str, tp.Any], use_custom_trainer: bool) -> None:
        config = initial_config
        if use_custom_trainer:
            config["get_trainer_func"] = custom_trainer
        model = SASRecModel.from_config(config)

        for key, config_value in config.items():
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
        model = SASRecModel(**config)
        actual = model.get_config(simple_types=simple_types)

        expected = config.copy()
        expected["cls"] = SASRecModel

        if simple_types:
            simple_types_params = {
                "cls": "SASRecModel",
                "item_net_block_types": ["rectools.models.nn.item_net.IdEmbeddingsItemNet"],
                "item_net_constructor_type": "rectools.models.nn.item_net.SumOfEmbeddingsConstructor",
                "pos_encoding_type": "rectools.models.nn.transformers.net_blocks.LearnableInversePositionalEncoding",
                "transformer_layers_type": "rectools.models.nn.transformers.sasrec.SASRecTransformerLayers",
                "data_preparator_type": "rectools.models.nn.transformers.sasrec.SASRecDataPreparator",
                "lightning_module_type": "rectools.models.nn.transformers.lightning.TransformerLightningModule",
                "get_val_mask_func": "tests.models.nn.transformers.utils.leave_one_out_mask",
                "similarity_module_type": "rectools.models.nn.transformers.similarity.DistanceSimilarityModule",
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
        model = SASRecModel
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

        def get_reco(model: SASRecModel) -> pd.DataFrame:
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
        model = SASRecModel()
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
