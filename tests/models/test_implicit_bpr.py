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

import typing as tp
from copy import deepcopy

import implicit.gpu
import numpy as np
import pandas as pd
import pytest
from implicit.bpr import BayesianPersonalizedRanking

# pylint: disable=no-name-in-module
from implicit.cpu.bpr import BayesianPersonalizedRanking as CPUBayesianPersonalizedRanking
from implicit.gpu import HAS_CUDA
from implicit.gpu.bpr import BayesianPersonalizedRanking as GPUBayesianPersonalizedRanking

# pylint: enable=no-name-in-module
from rectools.columns import Columns
from rectools.dataset.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.base import ModelBase
from rectools.models.implicit_bpr import AnyBayesianPersonalizedRanking, ImplicitBPRWrapperModel
from rectools.models.utils import recommend_from_scores
from tests.models.data import DATASET
from tests.models.utils import (
    assert_default_config_and_default_model_params_are_the_same,
    assert_dumps_loads_do_not_change_model,
    assert_second_fit_refits_model,
)

# Note that num_threads > 1 for BayesianPersonalizedRanking CPU training will make model training undeterministic
# https://github.com/benfred/implicit/issues/710
# GPU training is always underministic


@pytest.mark.parametrize("use_gpu", (False, True) if HAS_CUDA else (False,))
class TestImplicitBPRWrapperModel:
    # Tries to make BPR model deterministic
    @staticmethod
    def _init_model_factors_inplace(model: AnyBayesianPersonalizedRanking, dataset: Dataset) -> None:
        n_factors = model.factors
        n_users = dataset.user_id_map.to_internal.size
        n_items = dataset.item_id_map.to_internal.size
        user_factors: np.ndarray = np.linspace(0.1, 0.5, n_users * n_factors, dtype=np.float32).reshape(n_users, -1)
        item_factors: np.ndarray = np.linspace(0.1, 0.5, n_items * n_factors, dtype=np.float32).reshape(n_items, -1)

        if isinstance(model, GPUBayesianPersonalizedRanking):
            user_factors = implicit.gpu.Matrix(user_factors)
            item_factors = implicit.gpu.Matrix(item_factors)

        model.user_factors = user_factors
        model.item_factors = item_factors

    @pytest.fixture
    def dataset(self) -> Dataset:
        return DATASET

    @pytest.mark.parametrize(
        "filter_viewed,expected_cpu,expected_gpu",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [17, 13, 17, 15],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [17, 15, 17, 15],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20],
                        Columns.Item: [11, 17],
                        Columns.Rank: [1, 2],
                    }
                ),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [17, 15, 17, 15],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_basic(
        self,
        dataset: Dataset,
        filter_viewed: bool,
        expected_cpu: pd.DataFrame,
        expected_gpu: pd.DataFrame,
        use_gpu: bool,
    ) -> None:
        base_model = BayesianPersonalizedRanking(
            factors=2, num_threads=1, iterations=100, use_gpu=use_gpu, random_state=42
        )
        self._init_model_factors_inplace(base_model, dataset)
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend(
            users=np.unique(expected_cpu[Columns.User]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
        )
        expected = expected_gpu if use_gpu else expected_cpu
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_consistent_with_pure_implicit(self, dataset: Dataset, use_gpu: bool) -> None:
        base_model = BayesianPersonalizedRanking(
            factors=2, num_threads=1, iterations=100, use_gpu=use_gpu, random_state=42
        )
        self._init_model_factors_inplace(base_model, dataset)
        users = np.array([10, 20, 30, 40])

        model_for_wrap = deepcopy(base_model)
        state = np.random.get_state()
        wrapper_model = ImplicitBPRWrapperModel(model=model_for_wrap).fit(dataset)
        actual_reco = wrapper_model.recommend(users=users, dataset=dataset, k=3, filter_viewed=False)

        ui_csr = dataset.get_user_item_matrix(include_weights=True)
        np.random.set_state(state)
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
            np.testing.assert_allclose(actual_scores, expected_scores, atol=0.03)

    @pytest.mark.skipif(not implicit.gpu.HAS_CUDA, reason="implicit cannot find CUDA for gpu ranking")
    def test_gpu_ranking_consistent_with_pure_implicit(
        self,
        dataset: Dataset,
        use_gpu: bool,
    ) -> None:
        base_model = BayesianPersonalizedRanking(
            factors=2, num_threads=1, iterations=100, use_gpu=False, random_state=42
        )
        self._init_model_factors_inplace(base_model, dataset)
        users = np.array([10, 20, 30, 40])

        ui_csr = dataset.get_user_item_matrix(include_weights=True)
        base_model.fit(ui_csr)
        gpu_model = base_model.to_gpu()

        wrapped_model = ImplicitBPRWrapperModel(model=gpu_model, recommend_use_gpu_ranking=True)
        wrapped_model.is_fitted = True
        wrapped_model.model = wrapped_model._model  # pylint: disable=protected-access

        actual_reco = wrapped_model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=False,
        )

        for user_id in users:
            internal_id = dataset.user_id_map.convert_to_internal([user_id])[0]
            expected_ids, expected_scores = gpu_model.recommend(
                userid=internal_id,
                user_items=ui_csr[internal_id],
                N=3,
                filter_already_liked_items=False,
            )
            actual_ids = actual_reco.loc[actual_reco[Columns.User] == user_id, Columns.Item].values
            actual_internal_ids = dataset.item_id_map.convert_to_internal(actual_ids)
            actual_scores = actual_reco.loc[actual_reco[Columns.User] == user_id, Columns.Score].values
            np.testing.assert_equal(actual_internal_ids, expected_ids)
            np.testing.assert_allclose(actual_scores, expected_scores, atol=0.00001)

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
        base_model = BayesianPersonalizedRanking(
            factors=32, num_threads=1, iterations=100, use_gpu=use_gpu, random_state=42
        )
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
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
        "filter_itself,allowlist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 12, 12, 11],
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
                        Columns.Item: [12, 14, 11, 14],
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
                        Columns.Item: [11, 14, 11, 14],
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
        allowlist: tp.Optional[np.ndarray],
        expected: pd.DataFrame,
        use_gpu: bool,
    ) -> None:
        base_model = BayesianPersonalizedRanking(
            factors=2, num_threads=1, iterations=100, use_gpu=use_gpu, random_state=1
        )
        self._init_model_factors_inplace(base_model, dataset)
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=dataset,
            k=2,
            filter_itself=filter_itself,
            items_to_recommend=allowlist,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Rank], ascending=[True, True]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, dataset: Dataset, use_gpu: bool) -> None:
        # GPU training is always nondeterministic so we only test for CPU training
        if use_gpu:
            pytest.skip("BPR is nondeterministic on GPU")
        base_model = BayesianPersonalizedRanking(factors=8, num_threads=1, use_gpu=use_gpu, random_state=1)
        model = ImplicitBPRWrapperModel(model=base_model)
        state = np.random.get_state()

        def set_random_state() -> None:
            np.random.set_state(state)

        assert_second_fit_refits_model(model, dataset, set_random_state)

    def test_dumps_loads(self, dataset: Dataset, use_gpu: bool) -> None:
        base_model = BayesianPersonalizedRanking(factors=8, num_threads=1, use_gpu=use_gpu, random_state=1)
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
        assert_dumps_loads_do_not_change_model(model, dataset)

    def test_get_vectors(self, dataset: Dataset, use_gpu: bool) -> None:
        base_model = BayesianPersonalizedRanking(use_gpu=use_gpu)
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
        users_embeddings, item_embeddings = model.get_vectors()
        predictions = users_embeddings @ item_embeddings.T
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
        model = ImplicitBPRWrapperModel(model=BayesianPersonalizedRanking(use_gpu=use_gpu))
        with pytest.raises(NotFittedError):
            model.get_vectors()

    def test_u2i_with_cold_users(self, use_gpu: bool, dataset: Dataset) -> None:
        base_model = BayesianPersonalizedRanking(use_gpu=use_gpu)
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(ValueError, match="doesn't support recommendations for cold users"):
            model.recommend(
                users=[10, 20, 50],
                dataset=dataset,
                k=2,
                filter_viewed=False,
            )

    def test_i2i_with_warm_and_cold_items(self, use_gpu: bool, dataset: Dataset) -> None:
        base_model = BayesianPersonalizedRanking(use_gpu=use_gpu)
        model = ImplicitBPRWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(ValueError, match="doesn't support recommendations for cold items"):
            model.recommend_to_items(
                target_items=[11, 12, 16],
                dataset=dataset,
                k=2,
            )


class CustomBPR(CPUBayesianPersonalizedRanking):
    pass


class TestImplicitBPRWrapperModelConfiguration:
    def setup_method(self) -> None:
        implicit.gpu.HAS_CUDA = True

    @pytest.mark.parametrize("use_gpu", (False, True))
    @pytest.mark.parametrize("cls", (None, "BayesianPersonalizedRanking", "implicit.bpr.BayesianPersonalizedRanking"))
    @pytest.mark.parametrize("recommend_use_gpu", (None, False, True))
    @pytest.mark.parametrize("recommend_n_threads", (None, 10))
    def test_from_config(
        self, use_gpu: bool, cls: tp.Any, recommend_use_gpu: tp.Optional[bool], recommend_n_threads: tp.Optional[int]
    ) -> None:
        config: tp.Dict = {
            "model": {
                "factors": 10,
                "learning_rate": 0.01,
                "regularization": 0.01,
                "iterations": 100,
                "num_threads": 2,
                "verify_negative_samples": False,
                "use_gpu": use_gpu,
            },
            "verbose": 1,
            "recommend_n_threads": recommend_n_threads,
            "recommend_use_gpu_ranking": recommend_use_gpu,
        }
        if cls is not None:
            config["model"]["cls"] = cls
        model = ImplicitBPRWrapperModel.from_config(config)
        assert model.verbose == 1
        inner_model = model._model  # pylint: disable=protected-access
        assert inner_model.factors == 10
        assert inner_model.learning_rate == 0.01
        assert inner_model.regularization == 0.01
        assert inner_model.iterations == 100
        assert inner_model.verify_negative_samples is False
        if not use_gpu:
            assert inner_model.num_threads == 2

        if recommend_n_threads is not None:
            assert model.recommend_n_threads == recommend_n_threads
        elif not use_gpu:
            assert model.recommend_n_threads == inner_model.num_threads
        else:
            assert model.recommend_n_threads == 0
        if recommend_use_gpu is not None:
            assert model.recommend_use_gpu_ranking == recommend_use_gpu
        else:
            assert model.recommend_use_gpu_ranking == use_gpu
        expected_model_class = GPUBayesianPersonalizedRanking if use_gpu else CPUBayesianPersonalizedRanking
        assert isinstance(inner_model, expected_model_class)

    @pytest.mark.parametrize("use_gpu", (False, True))
    @pytest.mark.parametrize("random_state", (None, 42))
    @pytest.mark.parametrize("simple_types", (False, True))
    @pytest.mark.parametrize("recommend_use_gpu", (None, False, True))
    @pytest.mark.parametrize("recommend_n_threads", (None, 10))
    def test_to_config(
        self,
        use_gpu: bool,
        random_state: tp.Optional[int],
        simple_types: bool,
        recommend_use_gpu: tp.Optional[bool],
        recommend_n_threads: tp.Optional[int],
    ) -> None:
        model = ImplicitBPRWrapperModel(
            model=BayesianPersonalizedRanking(
                factors=10,
                learning_rate=0.01,
                regularization=0.01,
                iterations=100,
                num_threads=2,
                verify_negative_samples=False,
                random_state=random_state,
                use_gpu=use_gpu,
            ),
            verbose=1,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu,
        )
        config = model.get_config(simple_types=simple_types)
        expected_inner_model_config = {
            "cls": "BayesianPersonalizedRanking",
            "dtype": np.float64 if not simple_types else "float64",
            "factors": 10,
            "learning_rate": 0.01,
            "regularization": 0.01,
            "iterations": 100,
            "verify_negative_samples": False,
            "use_gpu": use_gpu,
            "random_state": random_state,
        }
        if not use_gpu:
            expected_inner_model_config.update(
                {
                    "num_threads": 2,
                    "dtype": np.float32 if not simple_types else "float32",  # type: ignore
                }
            )
        expected = {
            "cls": "ImplicitBPRWrapperModel" if simple_types else ImplicitBPRWrapperModel,
            "model": expected_inner_model_config,
            "verbose": 1,
            "recommend_use_gpu_ranking": recommend_use_gpu,
            "recommend_n_threads": recommend_n_threads,
        }
        assert config == expected

    def test_to_config_fails_when_random_state_is_object(self) -> None:
        model = ImplicitBPRWrapperModel(model=BayesianPersonalizedRanking(random_state=np.random.RandomState()))
        with pytest.raises(
            TypeError,
            match="`random_state` must be ``None`` or have ``int`` type to convert it to simple type",
        ):
            model.get_config(simple_types=True)

    def test_custom_model_class(self) -> None:
        cls_path = "tests.models.test_implicit_bpr.CustomBPR"

        config = {
            "model": {
                "cls": cls_path,
            }
        }
        model = ImplicitBPRWrapperModel.from_config(config)

        assert isinstance(model._model, CustomBPR)  # pylint: disable=protected-access

        returned_config = model.get_config(simple_types=True)
        assert returned_config["model"]["cls"] == cls_path  # pylint: disable=unsubscriptable-object

        assert model.get_config()["model"]["cls"] == CustomBPR  # pylint: disable=unsubscriptable-object

    @pytest.mark.parametrize("simple_types", (False, True))
    @pytest.mark.parametrize("recommend_use_gpu", (None, False, True))
    @pytest.mark.parametrize("recommend_n_threads", (None, 10))
    def test_get_config_and_from_config_compatibility(
        self, simple_types: bool, recommend_use_gpu: tp.Optional[bool], recommend_n_threads: tp.Optional[int]
    ) -> None:
        initial_config = {
            "model": {"factors": 4, "num_threads": 1, "iterations": 2, "random_state": 42},
            "verbose": 1,
            "recommend_use_gpu_ranking": recommend_use_gpu,
            "recommend_n_threads": recommend_n_threads,
        }
        dataset = DATASET
        model = ImplicitBPRWrapperModel

        def get_reco(model: ModelBase) -> pd.DataFrame:
            return model.fit(dataset).recommend(users=np.array([10, 20]), dataset=dataset, k=2, filter_viewed=False)

        state = np.random.get_state()
        model_1 = model.from_config(initial_config)
        reco_1 = get_reco(model_1)
        config_1 = model_1.get_config(simple_types=simple_types)

        model_2 = model.from_config(config_1)
        np.random.set_state(state)
        reco_2 = get_reco(model_2)

        config_2 = model_2.get_config(simple_types=simple_types)

        assert config_1 == config_2
        pd.testing.assert_frame_equal(reco_1, reco_2, atol=0.01)

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        default_config: tp.Dict[str, tp.Any] = {"model": {}}
        model = ImplicitBPRWrapperModel(model=BayesianPersonalizedRanking())
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
