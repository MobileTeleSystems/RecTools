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
import torch
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.models import SpotlightFactorizationWrapperModel
from rectools.models.utils import recommend_from_scores

HAS_CUDA = torch.cuda.is_available()

ITEMS = set(range(11, 22 + 1))


@pytest.mark.parametrize("use_cuda", (False, True) if HAS_CUDA else (False,))
@pytest.mark.parametrize("model_type", ("explicit", "implicit"))
class TestSpotlightFactorizationWrapperModel:
    @pytest.fixture
    def base_model(
        self, use_cuda: bool, model_type: str
    ) -> tp.Union[ImplicitFactorizationModel, ExplicitFactorizationModel]:
        if model_type == "implicit":
            base_model = ImplicitFactorizationModel(loss="bpr", n_iter=200, use_cuda=use_cuda)
        elif model_type == "explicit":
            base_model = ExplicitFactorizationModel(loss="regression", n_iter=100, use_cuda=use_cuda)
        return base_model

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions = pd.DataFrame(
            [
                [10, 11, 8],
                [10, 11, 8],
                [10, 11, 8],
                [10, 12, 4],
                [10, 13, 1],
                [10, 14, 1],
                [10, 11, 8],
                [10, 12, 4],
                [10, 12, 4],
                [10, 11, 8],
                [10, 12, 1],
                [10, 12, 4],
                [10, 11, 8],
                [10, 11, 8],
                [10, 11, 8],
                [10, 14, 1],
                [10, 15, 1],
                [10, 16, 1],
                [10, 17, 1],
                [10, 18, 1],
                [10, 19, 1],
                [10, 20, 1],
                [10, 21, 1],
                [20, 11, 8],
                [20, 11, 8],
                [20, 11, 8],
                [20, 11, 8],
                [20, 12, 4],
                [20, 13, 1],
                [20, 15, 1],
                [20, 11, 8],
                [20, 12, 4],
                [20, 12, 4],
                [20, 12, 1],
                [20, 11, 8],
                [20, 11, 8],
                [20, 11, 8],
                [20, 14, 1],
                [20, 15, 1],
                [20, 16, 1],
                [20, 17, 1],
                [20, 18, 1],
                [20, 19, 1],
                [20, 20, 1],
                [20, 22, 1],
                [30, 11, 1],
                [30, 12, 1],
                [30, 15, 1],
                [40, 11, 1],
                [40, 12, 1],
                [50, 11, 1],
                [50, 12, 1],
                [50, 14, 1],
                [60, 16, 1],
                [60, 12, 2],
                [70, 17, 1],
                [70, 12, 1],
                [80, 18, 1],
                [90, 19, 1],
                [100, 20, 1],
                [110, 21, 1],
                [120, 22, 1],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight],
        )
        interactions[Columns.Datetime] = "2021-09-09"
        return interactions

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 20],
                        Columns.Item: [22, 21],
                        Columns.Rank: [1, 1],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [11, 12, 11, 12],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_base(
        self,
        base_model: tp.Union[ImplicitFactorizationModel, ExplicitFactorizationModel],
        dataset: Dataset,
        filter_viewed: bool,
        expected: pd.DataFrame,
    ) -> None:
        model = SpotlightFactorizationWrapperModel(base_model).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [],
                        Columns.Item: [],
                        Columns.Rank: [],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [11, 15, 11, 15],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(
        self,
        base_model: tp.Union[ImplicitFactorizationModel, ExplicitFactorizationModel],
        dataset: Dataset,
        filter_viewed: bool,
        expected: pd.DataFrame,
    ) -> None:
        wrapper = SpotlightFactorizationWrapperModel(base_model).fit(dataset)
        actual = wrapper.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([11, 15]),
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score).astype("float64"), expected.astype("float64"))
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_get_vectors(
        self,
        base_model: tp.Union[ImplicitFactorizationModel, ExplicitFactorizationModel],
        dataset: Dataset,
    ) -> None:
        wrapper = SpotlightFactorizationWrapperModel(base_model)
        wrapper.fit(dataset)
        user_embeddings, item_embeddings = wrapper.get_vectors(add_biases=True)
        predictions = user_embeddings @ item_embeddings.T
        vectors_predictions = [recommend_from_scores(predictions[i], k=5) for i in range(6)]
        vectors_reco = np.array([vp[0] for vp in vectors_predictions]).ravel()
        vectors_scores = np.array([vp[1] for vp in vectors_predictions]).ravel()
        _, reco_item_ids, reco_scores = wrapper._recommend_u2i(  # pylint: disable=protected-access
            user_ids=dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40, 50, 60])),
            dataset=dataset,
            k=5,
            filter_viewed=False,
            sorted_item_ids_to_recommend=None,
        )
        np.testing.assert_equal(vectors_reco, reco_item_ids)
        np.testing.assert_almost_equal(vectors_scores, reco_scores, decimal=5)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                {11: ITEMS, 12: ITEMS},
            ),
            (
                True,
                None,
                {11: ITEMS - {11}, 12: ITEMS - {12}},
            ),
            (
                False,
                np.array([11, 15, 12]),
                {11: {11, 12, 15}, 12: {11, 12, 15}},
            ),
        ),
    )
    def test_i2i(
        self,
        dataset: Dataset,
        filter_itself: bool,
        whitelist: tp.Optional[np.ndarray],
        expected: pd.DataFrame,
        base_model: tp.Union[ImplicitFactorizationModel, ExplicitFactorizationModel],
    ) -> None:
        wrapper = SpotlightFactorizationWrapperModel(base_model).fit(dataset)
        target_items = np.array([11, 12])
        actual = wrapper.recommend_to_items(
            target_items=target_items,
            dataset=dataset,
            k=len(ITEMS),
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        assert np.allclose(actual[Columns.TargetItem].unique(), target_items)
        for tid in target_items:
            assert set(actual[actual[Columns.TargetItem] == tid][Columns.Item].unique()) == expected[tid]

        # If it's allowed to recommend itself, it must be on the first place
        if not filter_itself and whitelist is None:
            pd.testing.assert_frame_equal(
                actual.groupby(Columns.TargetItem, sort=False).head(1).reset_index(drop=True),
                pd.DataFrame(
                    {
                        Columns.TargetItem: target_items,
                        Columns.Item: target_items,
                        Columns.Score: [1.0, 1.0],
                        Columns.Rank: [1, 1],
                    }
                ),
                check_dtype=False,
            )
