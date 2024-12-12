#  Copyright 2022-2024 MTS (Mobile Telesystems)
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
from lightning_fabric import seed_everything

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models import DSSMModel
from rectools.models.dssm import DSSM
from rectools.models.vector import ImplicitRanker
from tests.models.utils import assert_dumps_loads_do_not_change_model, assert_second_fit_refits_model

from .data import INTERACTIONS


@pytest.mark.filterwarnings("ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDSSMModel:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        seed_everything(42, workers=True)

    @pytest.fixture
    def dataset(self) -> Dataset:
        item_features = pd.DataFrame(
            [
                [11, "f1", "f1val1"],
                [11, "f2", "f2val1"],
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
            ],
            columns=["id", "feature", "value"],
        )
        user_features = pd.DataFrame(
            [
                [10, "f1", "f1val1"],
                [10, "f2", "f2val1"],
                [20, "f1", "f1val1"],
                [20, "f2", "f2val1"],
                [30, "f1", "f1val1"],
                [30, "f2", "f2val1"],
                [40, "f1", "f1val1"],
                [40, "f2", "f2val1"],
                [50, "f1", "f1val1"],
                [50, "f2", "f2val1"],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            user_features_df=user_features,
            cat_user_features=["f1", "f2"],
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 20, 20, 20, 50, 50, 50],
                        Columns.Item: [13, 15, 17, 14, 15, 17, 11, 12, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 20, 20, 20, 50, 50, 50],
                        Columns.Item: [11, 12, 13, 11, 12, 13, 11, 12, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    @pytest.mark.parametrize("default_base_model", (True, False))
    def test_u2i(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame, default_base_model: bool) -> None:
        if default_base_model:
            base_model = None
        else:
            base_model = DSSM(
                n_factors_item=32,
                n_factors_user=32,
                dim_input_item=dataset.item_features.get_sparse().shape[1],  # type: ignore
                dim_input_user=dataset.user_features.get_sparse().shape[1],  # type: ignore
                dim_interactions=dataset.get_user_item_matrix().shape[1],
            )
        model = DSSMModel(
            model=base_model,
            n_factors=32,
            max_epochs=3,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset, dataset_valid=dataset)
        users = np.array([10, 20, 50])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=filter_viewed)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, True]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 50, 50, 50],
                        Columns.Item: [13, 17, 11, 13, 17],
                        Columns.Rank: [1, 2, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 50, 50, 50],
                        Columns.Item: [11, 13, 17, 11, 13, 17],
                        Columns.Rank: [1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = DSSMModel(
            n_factors=32,
            max_epochs=3,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 50])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([17, 13, 11]),
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, True]).reset_index(drop=True),
            actual,
        )

    def test_get_vectors(self, dataset: Dataset) -> None:
        base_model = DSSM(
            n_factors_item=32,
            n_factors_user=32,
            dim_input_item=dataset.item_features.get_sparse().shape[1],  # type: ignore
            dim_input_user=dataset.user_features.get_sparse().shape[1],  # type: ignore
            dim_interactions=dataset.get_user_item_matrix().shape[1],
            lr=0.001,
            triplet_loss_margin=0.2,
        )
        model = DSSMModel(
            model=base_model,
            max_epochs=3,
            batch_size=4,
            dataloader_num_workers=0,
            callbacks=None,
        )
        model.fit(dataset=dataset)
        user_embeddings, item_embeddings = model.get_vectors(dataset)
        ranker = ImplicitRanker(model.u2i_dist, user_embeddings, item_embeddings)
        _, vectors_reco, vectors_scores = ranker.rank(
            dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40])), k=5
        )
        (
            _,
            reco_item_ids,
            reco_scores,
        ) = model._recommend_u2i(  # pylint: disable=protected-access
            user_ids=dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40])),
            dataset=dataset,
            k=5,
            filter_viewed=False,
            sorted_item_ids_to_recommend=None,
        )
        np.testing.assert_equal(vectors_reco, reco_item_ids)
        np.testing.assert_almost_equal(vectors_scores, reco_scores, decimal=5)

    def test_raises_when_get_vectors_from_not_fitted(self, dataset: Dataset) -> None:
        base_model = DSSM(
            n_factors_item=32,
            n_factors_user=32,
            dim_input_item=dataset.item_features.get_sparse().shape[1],  # type: ignore
            dim_input_user=dataset.user_features.get_sparse().shape[1],  # type: ignore
            dim_interactions=dataset.get_user_item_matrix().shape[1],
            lr=0.001,
            triplet_loss_margin=0.2,
        )
        model = DSSMModel(
            model=base_model,
            max_epochs=3,
            batch_size=4,
            dataloader_num_workers=0,
            callbacks=None,
        )
        with pytest.raises(NotFittedError):
            model.get_vectors(dataset)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 11, 12, 12, 12, 16, 16, 16],
                        Columns.Item: [11, 13, 17, 12, 16, 17, 16, 17, 14],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 11, 12, 12, 12, 16, 16, 16],
                        Columns.Item: [13, 16, 17, 16, 17, 14, 17, 14, 12],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                np.array([11, 15, 12]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12, 16, 16, 16],
                        Columns.Item: [12, 15, 15, 11, 12, 11, 15],
                        Columns.Rank: [1, 2, 1, 2, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        model = DSSMModel(
            n_factors=2,
            max_epochs=3,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset, dataset_valid=dataset)
        target_items = np.array([11, 12, 16])
        actual = model.recommend_to_items(
            target_items=target_items,
            dataset=dataset,
            k=3,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, True]).reset_index(drop=True),
            actual,
        )

    def test_u2i_with_cold_users(self, dataset: Dataset) -> None:
        model = DSSMModel().fit(dataset)
        with pytest.raises(ValueError, match="doesn't support recommendations for cold users"):
            model.recommend(
                users=[10, 60],
                dataset=dataset,
                k=2,
                filter_viewed=False,
            )

    def test_i2i_with_cold_items(self, dataset: Dataset) -> None:
        model = DSSMModel().fit(dataset)
        with pytest.raises(ValueError, match="doesn't support recommendations for cold items"):
            model.recommend_to_items(
                target_items=[11, 18],
                dataset=dataset,
                k=2,
            )

    @pytest.mark.parametrize("exclude_features", ("user", "item"))
    def test_raises_when_no_features_in_dataset(self, dataset: Dataset, exclude_features: str) -> None:
        dataset = Dataset(
            dataset.user_id_map,
            dataset.item_id_map,
            dataset.interactions,
            dataset.user_features if exclude_features != "user" else None,
            dataset.item_features if exclude_features != "item" else None,
        )
        model = DSSMModel()
        with pytest.raises(ValueError, match="requires user and item features"):
            model.fit(dataset)

    def test_second_fit_refits_model(self, dataset: Dataset) -> None:
        model = DSSMModel(deterministic=True)
        assert_second_fit_refits_model(model, dataset, pre_fit_callback=self._seed_everything)

    def test_dumps_loads(self, dataset: Dataset) -> None:
        model = DSSMModel()
        model.fit(dataset)
        assert_dumps_loads_do_not_change_model(model, dataset, check_configs=False)
