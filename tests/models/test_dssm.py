#  Copyright 2022 MTS (Mobile Telesystems)
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

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.dataset.torch_datasets import DSSMDataset
from rectools.exceptions import NotFittedError
from rectools.models import DSSMModel
from rectools.models.dssm import DSSM
from rectools.models.utils import recommend_from_scores
from rectools.models.vector import ScoreCalculator

from .data import INTERACTIONS


@pytest.mark.filterwarnings("ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDSSMModel:
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
                {10: {13, 15, 17}, 20: {14, 15, 17}},
            ),
        ),
    )
    def test_basic(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
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
            dataset_type=DSSMDataset,  # type: ignore
            max_epochs=3,
            batch_size=4,
            dataloader_num_workers=0,
            callbacks=None,
        )
        model.fit(dataset=dataset, dataset_valid=dataset)
        users = np.array([10, 20])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=filter_viewed)
        assert np.allclose(actual[Columns.User].unique(), users)
        for uid in users:
            assert set(actual[actual[Columns.User] == uid][Columns.Item].unique()) == expected[uid]

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                {10: {13, 15, 17}, 20: {14, 15, 17}},
            ),
        ),
    )
    def test_basic_default(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = DSSMModel(
            dataset_type=DSSMDataset,  # type: ignore
            max_epochs=3,
            batch_size=4,
            dataloader_num_workers=0,
            callbacks=None,
        )
        model.fit(dataset=dataset, dataset_valid=dataset)
        users = np.array([10, 20])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=filter_viewed)
        assert np.allclose(actual[Columns.User].unique(), users)
        for uid in users:
            assert set(actual[actual[Columns.User] == uid][Columns.Item].unique()) == expected[uid]

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                False,
                {10: {11, 13, 17}},
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
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
            dataset_type=DSSMDataset,  # type: ignore
            max_epochs=3,
            batch_size=4,
            dataloader_num_workers=0,
            callbacks=None,
        )
        model.fit(dataset=dataset)
        users = np.array([10])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([17, 13, 11]),
        )
        assert np.allclose(actual[Columns.User].unique(), users)
        for uid in users:
            assert set(actual[actual[Columns.User] == uid][Columns.Item].unique()) == expected[uid]

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
            dataset_type=DSSMDataset,  # type: ignore
            max_epochs=3,
            batch_size=4,
            dataloader_num_workers=0,
            callbacks=None,
        )
        model.fit(dataset=dataset)
        user_embeddings, item_embeddings = model.get_vectors(dataset)
        score_calculator = ScoreCalculator(model.u2i_dist, user_embeddings, item_embeddings)
        predictions = [
            score_calculator.calc(uid) for uid in dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40]))
        ]
        vectors_predictions = [recommend_from_scores(-predictions[i].flatten(), k=5) for i in range(4)]
        vectors_reco = np.array([vp[0] for vp in vectors_predictions]).ravel()
        vectors_scores = np.array([vp[1] for vp in vectors_predictions]).ravel()
        (_, reco_item_ids, reco_scores,) = model._recommend_u2i(  # pylint: disable=protected-access
            user_ids=dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40])),
            dataset=dataset,
            k=5,
            filter_viewed=False,
            sorted_item_ids_to_recommend=None,
        )
        np.testing.assert_equal(vectors_reco, reco_item_ids)
        np.testing.assert_almost_equal(-vectors_scores, reco_scores, decimal=5)

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
            dataset_type=DSSMDataset,  # type: ignore
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
                {11: {11, 12, 13, 14, 15, 17}, 12: {11, 12, 13, 14, 15, 17}},
            ),
            # (
            #     True,
            #     None,
            #     {11: {12, 13, 14, 15, 17}, 12: {11, 13, 14, 15, 17}},
            # ),
            # (
            #     False,
            #     np.array([11, 15, 12]),
            #     {11: {11, 12, 15}, 12: {11, 12, 15}},
            # ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        base_model = DSSM(
            n_factors_item=2,
            n_factors_user=2,
            dim_input_item=dataset.item_features.get_sparse().shape[1],  # type: ignore
            dim_input_user=dataset.user_features.get_sparse().shape[1],  # type: ignore
            dim_interactions=dataset.get_user_item_matrix().shape[1],
        )
        model = DSSMModel(
            model=base_model,
            dataset_type=DSSMDataset,  # type: ignore
            max_epochs=3,
            batch_size=4,
        )
        model.fit(dataset=dataset, dataset_valid=dataset)
        target_items = np.array([11, 12])
        actual = model.recommend_to_items(
            target_items=target_items,
            dataset=dataset,
            k=6,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        assert np.allclose(actual[Columns.TargetItem].unique(), target_items)
        for tid in target_items:
            assert set(actual[actual[Columns.TargetItem] == tid][Columns.Item].unique()) == expected[tid]

        # If it's allowed to recommend itself, it must be on the first place
        if not filter_itself and whitelist is None:
            tol_kwargs: tp.Dict[str, float] = {"check_less_precise": 1} if pd.__version__ < "1" else {"atol": 0.01}
            # actual is on the 2-nd place because of strange behaviour of assert function for pd.__version__ < "1"
            pd.testing.assert_frame_equal(
                pd.DataFrame(
                    {
                        Columns.TargetItem: target_items,
                        Columns.Item: target_items,
                        Columns.Score: [0.0, 0.0],
                        Columns.Rank: [1, 1],
                    }
                ),
                actual.groupby(Columns.TargetItem, sort=False).head(1).reset_index(drop=True),
                check_dtype=False,
                **tol_kwargs,
            )
