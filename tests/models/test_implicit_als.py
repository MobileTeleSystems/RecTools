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
from copy import deepcopy

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
from rectools.models.implicit_als import (
    AnyAlternatingLeastSquares,
    CPUAlternatingLeastSquares,
    GPUAlternatingLeastSquares,
)
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
        user_factors: np.ndarray = np.linspace(0.1, 0.5, n_users * n_factors, dtype=np.float32).reshape(n_users, -1)
        item_factors: np.ndarray = np.linspace(0.1, 0.5, n_items * n_factors, dtype=np.float32).reshape(n_items, -1)

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

    @pytest.mark.parametrize("fit_features_together", (False, True))
    @pytest.mark.parametrize("init_model_before_fit", (False, True))
    def test_consistent_with_pure_implicit(
        self, dataset: Dataset, fit_features_together: bool, use_gpu: bool, init_model_before_fit: bool
    ) -> None:
        base_model = AlternatingLeastSquares(factors=10, num_threads=2, iterations=30, use_gpu=use_gpu, random_state=32)
        if init_model_before_fit:
            self._init_model_factors_inplace(base_model, dataset)
        users = np.array([10, 20, 30, 40])

        model_for_wrap = deepcopy(base_model)
        wrapped_model = ImplicitALSWrapperModel(model=model_for_wrap, fit_features_together=fit_features_together)
        wrapped_model.fit(dataset)
        actual_reco = wrapped_model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=False,
        )

        ui_csr = dataset.get_user_item_matrix(include_weights=True)
        base_model.fit(ui_csr)
        for user_id in users:
            internal_id = dataset.user_id_map.convert_to_internal([user_id])[0]
            expected_ids, expected_scores = base_model.recommend(
                userid=internal_id,
                user_items=ui_csr[internal_id],
                N=3,
                filter_already_liked_items=False,
            )
            actual_ids = actual_reco.loc[actual_reco[Columns.User] == user_id, Columns.Item].values
            actual_internal_ids = dataset.item_id_map.convert_to_internal(actual_ids)
            actual_scores = actual_reco.loc[actual_reco[Columns.User] == user_id, Columns.Score].values
            np.testing.assert_equal(actual_internal_ids, expected_ids)
            np.testing.assert_allclose(actual_scores, expected_scores, atol=0.01)

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
            assert set(actual.loc[actual[Columns.User] == uid, Columns.Item]) == expected[uid]

    @pytest.mark.parametrize(
        "fit_features_together,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: ["u1", "u3", "u3"],
                        Columns.Item: ["i2", "i3", "i2"],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: ["u1", "u3", "u3"],
                        Columns.Item: ["i2", "i2", "i3"],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_happy_path_with_features(self, fit_features_together: bool, expected: pd.DataFrame, use_gpu: bool) -> None:
        user_id_map = IdMap.from_values(["u1", "u2", "u3"])
        item_id_map = IdMap.from_values(["i1", "i2", "i3"])
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

    def test_u2i_with_cold_users(self, use_gpu: bool, dataset: Dataset) -> None:
        base_model = AlternatingLeastSquares(use_gpu=use_gpu)
        model = ImplicitALSWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(ValueError, match="doesn't support recommendations for cold users"):
            model.recommend(
                users=[10, 20, 50],
                dataset=dataset,
                k=2,
                filter_viewed=False,
            )

    def test_i2i_with_warm_and_cold_items(self, use_gpu: bool, dataset: Dataset) -> None:
        base_model = AlternatingLeastSquares(use_gpu=use_gpu)
        model = ImplicitALSWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(ValueError, match="doesn't support recommendations for cold items"):
            model.recommend_to_items(
                target_items=[11, 12, 16],
                dataset=dataset,
                k=2,
            )


class CustomALS(CPUAlternatingLeastSquares):
    pass


class TestImplicitALSWrapperModelConfiguration:

    def setup_method(self) -> None:
        implicit.gpu.HAS_CUDA = True  # To avoid errors when test without cuda

    @pytest.mark.parametrize("use_gpu", (False, True))
    def test_from_config(self, use_gpu: bool) -> None:
        config = {
            "model": {
                "params": {
                    "factors": 16,
                    "num_threads": 2,
                    "iterations": 100,
                    "use_gpu": use_gpu,
                },
            },
            "fit_features_together": True,
            "verbose": 1,
        }
        model = ImplicitALSWrapperModel.from_config(config)
        assert model.fit_features_together is True
        assert model.verbose == 1
        inner_model = model._model  # pylint: disable=protected-access
        assert inner_model.factors == 16
        assert inner_model.iterations == 100
        if not use_gpu:
            assert inner_model.num_threads == 2
        expected_model_class = GPUAlternatingLeastSquares if use_gpu else CPUAlternatingLeastSquares
        assert isinstance(inner_model, expected_model_class)

    @pytest.mark.parametrize("use_gpu", (False, True))
    @pytest.mark.parametrize("random_state", (None, 42))
    @pytest.mark.parametrize("simple_types", (False, True))
    def test_to_config(self, use_gpu: bool, random_state: tp.Optional[int], simple_types: bool) -> None:
        model = ImplicitALSWrapperModel(
            model=AlternatingLeastSquares(factors=16, num_threads=2, use_gpu=use_gpu, random_state=random_state),
            fit_features_together=True,
            verbose=1,
        )
        config = model.get_config(simple_types=simple_types)
        expected_model_params = {
            "factors": 16,
            "regularization": 0.01,
            "alpha": 1.0,
            "dtype": np.float32 if not simple_types else "float32",
            "iterations": 15,
            "calculate_training_loss": False,
            "random_state": random_state,
            "use_gpu": use_gpu,
        }
        if not use_gpu:
            expected_model_params.update(
                {
                    "use_native": True,
                    "use_cg": True,
                    "num_threads": 2,
                }
            )
        expected = {
            "model": {
                "cls": None,
                "params": expected_model_params,
            },
            "fit_features_together": True,
            "verbose": 1,
        }
        assert config == expected

    def test_to_config_fails_when_random_state_is_object(self) -> None:
        model = ImplicitALSWrapperModel(model=AlternatingLeastSquares(random_state=np.random.RandomState()))
        with pytest.raises(
            ValueError,
            match="`random_state` must be ``None`` or have ``int`` type to convert it to simple type",
        ):
            model.get_config(simple_types=True)

    def test_custom_model_class(self) -> None:
        cls_path = "tests.models.test_implicit_als.CustomALS"

        config = {
            "model": {
                "cls": cls_path,
            }
        }
        model = ImplicitALSWrapperModel.from_config(config)

        assert isinstance(model._model, CustomALS)  # pylint: disable=protected-access

        returned_config = model.get_config(simple_types=True)
        assert returned_config["model"]["cls"] == cls_path  # pylint: disable=unsubscriptable-object

        assert model.get_config()["model"]["cls"] == CustomALS  # pylint: disable=unsubscriptable-object

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(self, simple_types: bool) -> None:
        def get_reco(model: ImplicitALSWrapperModel) -> pd.DataFrame:
            return model.fit(DATASET).recommend(users=[10, 20], dataset=DATASET, k=2, filter_viewed=False)

        initial_config = {
            "model": {
                "params": {"factors": 16, "num_threads": 2, "iterations": 3, "random_state": 42},
            },
            "verbose": 1,
        }

        model_1 = ImplicitALSWrapperModel.from_config(initial_config)
        reco_1 = get_reco(model_1)
        config_1 = model_1.get_config(simple_types=simple_types)

        model_2 = ImplicitALSWrapperModel.from_config(config_1)
        reco_2 = get_reco(model_2)
        config_2 = model_2.get_config(simple_types=simple_types)

        assert config_1 == config_2
        pd.testing.assert_frame_equal(reco_1, reco_2)

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        model_from_config = ImplicitALSWrapperModel.from_config({"model": {}})
        model_from_params = ImplicitALSWrapperModel(model=AlternatingLeastSquares())
        assert model_from_config.get_config() == model_from_params.get_config()
