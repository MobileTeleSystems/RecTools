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

import implicit.gpu
import numpy as np
import pandas as pd
import pytest
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA

from rectools import Columns
from rectools.dataset import Dataset, DenseFeatures, IdMap, Interactions, SparseFeatures
from rectools.exceptions import NotFittedError
from rectools.models import ImplicitALSWrapperModel
from rectools.models.implicit_als import AnyAlternatingLeastSquares, GPUAlternatingLeastSquares
from rectools.models.utils import recommend_from_scores

from .data import DATASET
from .utils import assert_second_fit_refits_model


@pytest.mark.filterwarnings("ignore:Converting sparse features to dense")
@pytest.mark.parametrize("use_gpu", (False, True) if HAS_CUDA else (False,))
class TestImplicitALSWrapperModel:
    @staticmethod
    def _init_model_factors_inplace(model: AnyAlternatingLeastSquares, dataset: Dataset) -> None:
        """Init factors to make the test deterministic"""
        n_factors = model.factors
        n_users = dataset.user_id_map.to_internal.size
        n_items = dataset.item_id_map.to_internal.size
        user_factors = np.linspace(0.1, 0.5, n_users * n_factors, dtype=np.float32).reshape(n_users, n_factors)
        item_factors = np.linspace(0.1, 0.5, n_items * n_factors, dtype=np.float32).reshape(n_items, n_factors)

        if isinstance(model, GPUAlternatingLeastSquares):
            user_factors = implicit.gpu.Matrix(user_factors)
            item_factors = implicit.gpu.Matrix(item_factors)

        model.user_factors = user_factors
        model.item_factors = item_factors

    @pytest.fixture
    def dataset(self) -> Dataset:
        return DATASET

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [13, 15, 14, 15],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [12, 11, 12, 11],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    @pytest.mark.parametrize("fit_features_together", (False, True))
    def test_basic(
        self,
        dataset: Dataset,
        fit_features_together: bool,
        filter_viewed: bool,
        expected: pd.DataFrame,
        use_gpu: bool,
    ) -> None:
        base_model = AlternatingLeastSquares(factors=2, num_threads=2, iterations=100)
        self._init_model_factors_inplace(base_model, dataset)
        model = ImplicitALSWrapperModel(model=base_model, fit_features_together=fit_features_together).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
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
                {10: {13, 17}, 20: {17}},
            ),
            (
                False,
                {10: {11, 13, 17}, 20: {11, 13, 17}},
            ),
        ),
    )
    def test_with_whitelist(
        self,
        dataset: Dataset,
        filter_viewed: bool,
        expected: tp.Dict[int, tp.Set[int]],
        use_gpu: bool,
    ) -> None:
        base_model = AlternatingLeastSquares(factors=32, num_threads=2, use_gpu=use_gpu)
        model = ImplicitALSWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([11, 13, 17]),
        )
        for uid in (10, 20):
            assert set(actual.query(f"user_id == {uid}")["item_id"]) == expected[uid]

    @pytest.mark.parametrize("filter_viewed", (True, False))
    def test_raises_when_new_user(self, dataset: Dataset, filter_viewed: bool, use_gpu: bool) -> None:
        base_model = AlternatingLeastSquares(factors=2, num_threads=2, random_state=1, use_gpu=use_gpu)
        model = ImplicitALSWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(KeyError):
            model.recommend(
                users=np.array([10, 50]),
                dataset=dataset,
                k=2,
                filter_viewed=filter_viewed,
            )

    @pytest.mark.parametrize(
        "fit_features_together,expected",
        (
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: ["u1", "u1", "u2", "u3", "u3"],
                        Columns.Item: ["i2", "i4", "i4", "i2", "i3"],
                        Columns.Rank: [1, 2, 1, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_happy_path_with_features(self, fit_features_together: bool, expected: pd.DataFrame, use_gpu: bool) -> None:
        user_id_map = IdMap.from_values(["u1", "u2", "u3"])
        item_id_map = IdMap.from_values(["i1", "i2", "i3", "i4"])
        interactions_df = pd.DataFrame(
            [
                ["u1", "i1", 0.1, "2021-09-09"],
                ["u2", "i1", 0.1, "2021-09-09"],
                ["u2", "i2", 0.5, "2021-09-05"],
                ["u2", "i3", 0.2, "2021-09-05"],
                ["u1", "i3", 0.2, "2021-09-05"],
                ["u3", "i1", 0.2, "2021-09-05"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        )
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map)
        user_features_df = pd.DataFrame({"id": ["u1", "u2", "u3"], "f1": [0.3, 0.4, 0.5]})
        user_features = DenseFeatures.from_dataframe(user_features_df, user_id_map)
        item_features_df = pd.DataFrame({"id": ["i1", "i1"], "feature": ["f1", "f2"], "value": [2.1, 100]})
        item_features = SparseFeatures.from_flatten(item_features_df, item_id_map)
        dataset = Dataset(user_id_map, item_id_map, interactions, user_features, item_features)

        # In case of big number of iterations there are differences between CPU and GPU results
        base_model = AlternatingLeastSquares(factors=32, num_threads=2, use_gpu=use_gpu)
        self._init_model_factors_inplace(base_model, dataset)
        # Make common number of factors 32, so that CPU and GPU results be equal
        if fit_features_together:
            base_model.factors = 32 - user_features.values.shape[1] - item_features.values.shape[1]

        model = ImplicitALSWrapperModel(model=base_model, fit_features_together=fit_features_together).fit(dataset)
        actual = model.recommend(
            users=np.array(["u1", "u2", "u3"]),
            dataset=dataset,
            k=2,
            filter_viewed=True,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_get_vectors(self, dataset: Dataset, use_gpu: bool) -> None:
        base_model = AlternatingLeastSquares(use_gpu=use_gpu)
        model = ImplicitALSWrapperModel(model=base_model).fit(dataset)
        user_embeddings, item_embeddings = model.get_vectors()
        predictions = user_embeddings @ item_embeddings.T
        vectors_predictions = [recommend_from_scores(predictions[i], k=5) for i in range(4)]
        vectors_reco = np.array([vp[0] for vp in vectors_predictions]).ravel()
        vectors_scores = np.array([vp[1] for vp in vectors_predictions]).ravel()
        _, reco_item_ids, reco_scores = model._recommend_u2i(  # pylint: disable=protected-access
            user_ids=dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40])),
            dataset=dataset,
            k=5,
            filter_viewed=False,
            sorted_item_ids_to_recommend=None,
        )
        np.testing.assert_equal(vectors_reco, reco_item_ids)
        np.testing.assert_almost_equal(vectors_scores, reco_scores, decimal=5)

    def test_raises_when_get_vectors_from_not_fitted(self, use_gpu: bool) -> None:
        model = ImplicitALSWrapperModel(model=AlternatingLeastSquares(use_gpu=use_gpu))
        with pytest.raises(NotFittedError):
            model.get_vectors()

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 14, 12, 14],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [14, 15, 14, 13],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                np.array([11, 15, 14]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 14, 14, 11],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self,
        dataset: Dataset,
        filter_itself: bool,
        whitelist: tp.Optional[np.ndarray],
        expected: pd.DataFrame,
        use_gpu: bool,
    ) -> None:
        base_model = AlternatingLeastSquares(factors=2, iterations=100, num_threads=2, use_gpu=use_gpu)
        self._init_model_factors_inplace(base_model, dataset)
        model = ImplicitALSWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=dataset,
            k=2,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, use_gpu: bool, dataset: Dataset) -> None:
        base_model = AlternatingLeastSquares(factors=8, num_threads=2, use_gpu=use_gpu, random_state=1)
        model = ImplicitALSWrapperModel(model=base_model)
        assert_second_fit_refits_model(model, dataset)
