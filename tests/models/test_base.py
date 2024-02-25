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

# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from rectools import Columns
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.base import InternalRecoTriplet, ModelBase, Scores
from rectools.types import AnyIds, ExternalIds, InternalIds

from .data import DATASET, INTERACTIONS


def test_raise_when_recommend_u2i_from_not_fitted() -> None:
    model = ModelBase()
    with pytest.raises(NotFittedError):
        model.recommend(
            users=np.array([]),
            dataset=DATASET,
            k=5,
            filter_viewed=False,
        )


def test_raise_when_recommend_i2i_from_not_fitted() -> None:
    model = ModelBase()
    with pytest.raises(NotFittedError):
        model.recommend_to_items(
            target_items=np.array([]),
            dataset=DATASET,
            k=5,
        )


@pytest.mark.parametrize("k", (-4, 0))
def test_raise_when_k_is_not_positive_u2i(k: int) -> None:
    model = ModelBase()
    model.is_fitted = True
    with pytest.raises(ValueError):
        model.recommend(
            users=np.array([10, 20]),
            dataset=DATASET,
            k=k,
            filter_viewed=True,
        )


@pytest.mark.parametrize("k", (-4, 0))
def test_raise_when_k_is_not_positive_i2i(k: int) -> None:
    model = ModelBase()
    model.is_fitted = True
    with pytest.raises(ValueError):
        model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=DATASET,
            k=k,
        )


class TestRecommendWithInternalIds:
    def setup(self) -> None:
        class SomeModel(ModelBase):
            def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
                pass

            def _recommend_u2i(
                self,
                user_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                filter_viewed: bool,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
                return [0, 0, 1], [0, 1, 2], [0.1, 0.2, 0.3]

            def _recommend_i2i(
                self,
                target_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
                return [0, 0, 1], [0, 1, 2], [0.1, 0.2, 0.3]

        self.model = SomeModel().fit(DATASET)

    def test_u2i_success(self, mocker: MockerFixture) -> None:
        model = self.model
        users = [0, 1]
        items_to_recommend = [0, 1, 2]

        spy = mocker.spy(model, "_recommend_u2i")
        reco = model.recommend(
            users=users,
            dataset=DATASET,
            k=2,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
            assume_external_ids=False,
            add_rank_col=False,
        )

        args, _ = spy.call_args  # args and kwargs properties are unavailable in Python < 3.8
        assert list(args[0]) == users
        assert list(args[4]) == items_to_recommend

        excepted = pd.DataFrame(
            {
                Columns.User: [0, 0, 1],
                Columns.Item: [0, 1, 2],
                Columns.Score: [0.1, 0.2, 0.3],
            }
        )
        pd.testing.assert_frame_equal(reco, excepted.astype({Columns.Score: np.float32}))

    @pytest.mark.parametrize(
        "users, items_to_recommend, error_type",
        (
            (["u1", "u2"], [0, 1], TypeError),
            ([0, 1], ["i1", "i2"], TypeError),
            (["u1", "u2"], ["i1", "i2"], TypeError),
            ([0, 1], [-1, 1], ValueError),
            ([-1, 1], [0, 1], ValueError),
        ),
    )
    def test_u2i_with_incorrect_ids(self, users: AnyIds, items_to_recommend: AnyIds, error_type: tp.Type) -> None:
        with pytest.raises(error_type):
            self.model.recommend(
                users=users,
                dataset=DATASET,
                k=2,
                filter_viewed=False,
                items_to_recommend=items_to_recommend,
                assume_external_ids=False,
            )

    def test_i2i_success(self, mocker: MockerFixture) -> None:
        model = self.model
        target_items = [0, 1, 2]
        items_to_recommend = [0, 1, 2]

        spy = mocker.spy(model, "_recommend_i2i")
        reco = model.recommend_to_items(
            target_items=target_items,
            dataset=DATASET,
            k=2,
            items_to_recommend=items_to_recommend,
            assume_external_ids=False,
            add_rank_col=False,
            filter_itself=False,
        )

        args, _ = spy.call_args  # args and kwargs properties are unavailable in Python < 3.8
        assert list(args[0]) == target_items
        assert list(args[3]) == items_to_recommend

        excepted = pd.DataFrame(
            {
                Columns.TargetItem: [0, 0, 1],
                Columns.Item: [0, 1, 2],
                Columns.Score: [0.1, 0.2, 0.3],
            }
        )
        pd.testing.assert_frame_equal(reco, excepted.astype({Columns.Score: np.float32}))

    @pytest.mark.parametrize(
        "target_items, items_to_recommend, error_type",
        (
            (["i1", "i2"], [0, 1], TypeError),
            ([0, 1], ["i1", "i2"], TypeError),
            (["i1", "i2"], ["i1", "i2"], TypeError),
            ([0, 1], [-1, 1], ValueError),
            ([-1, 1], [0, 1], ValueError),
        ),
    )
    def test_i2i_with_incorrect_ids(
        self, target_items: AnyIds, items_to_recommend: AnyIds, error_type: tp.Type
    ) -> None:
        with pytest.raises(error_type):
            self.model.recommend_to_items(
                target_items=target_items,
                dataset=DATASET,
                k=2,
                items_to_recommend=items_to_recommend,
                assume_external_ids=False,
            )


class TestHotWarmCold:
    def setup(self) -> None:
        class HotModel(ModelBase):
            allow_cold = False
            allow_warm = False

            def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
                pass

            def _recommend_u2i(
                self,
                user_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                filter_viewed: bool,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
                return (
                    np.repeat(user_ids, k),
                    np.tile(np.arange(k), len(user_ids)),
                    np.tile(np.arange(1, k + 1) * 0.1, len(user_ids)),
                )

            def _recommend_i2i(
                self,
                target_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
                return (
                    np.repeat(target_ids, k),
                    np.tile(np.arange(k), len(target_ids)),
                    np.tile(np.arange(1, k + 1) * 0.1, len(target_ids)),
                )

        class HotWarmModel(HotModel):
            allow_warm = True

            def _recommend_u2i_warm(
                self,
                user_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> InternalRecoTriplet:
                return (
                    np.repeat(user_ids, k),
                    np.tile(np.arange(k), len(user_ids)),
                    np.tile(np.arange(1, k + 1) * 0.1 + 1, len(user_ids)),
                )

            def _recommend_i2i_warm(
                self,
                target_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> InternalRecoTriplet:
                return (
                    np.repeat(target_ids, k),
                    np.tile(np.arange(k), len(target_ids)),
                    np.tile(np.arange(1, k + 1) * 0.1 + 1, len(target_ids)),
                )

        class HotColdModel(HotModel):
            allow_cold = True

            def _get_cold_reco(
                self, k: int, sorted_item_ids_to_recommend: tp.Optional[np.ndarray]
            ) -> tp.Tuple[InternalIds, Scores]:
                return np.arange(k), np.arange(1, k + 1) * 0.1 + 2

        class HotWarmColdModel(HotWarmModel, HotColdModel):
            pass

        self.hot_model = HotModel().fit(DATASET)
        self.hot_warm_model = HotWarmModel().fit(DATASET)
        self.hot_cold_model = HotColdModel().fit(DATASET)
        self.hot_warm_cold_model = HotWarmColdModel().fit(DATASET)
        self.models = {
            "hot": self.hot_model,
            "hot_warm": self.hot_warm_model,
            "hot_cold": self.hot_cold_model,
            "hot_warm_cold": self.hot_warm_cold_model,
        }

        user_features = pd.DataFrame(
            {
                Columns.User: [40, 50],
                "feature": ["f1", "f1"],
                "value": [1, 2],
            }
        )
        item_features = pd.DataFrame(
            {
                Columns.Item: [16, 17],
                "feature": ["f1", "f1"],
                "value": [1, 2],
            }
        )
        self.datasets = {
            "no_features": DATASET,
            "with_features": Dataset.construct(
                INTERACTIONS, user_features_df=user_features, item_features_df=item_features
            ),
        }

        self.hots = {"u2i": [10], "i2i": [11]}
        self.warms = {"u2i": [50], "i2i": [16]}
        self.colds = {"u2i": [60], "i2i": [18]}

    def _get_reco(self, targers: ExternalIds, model_key: str, dataset_key: str, kind: str) -> pd.DataFrame:
        model = self.models[model_key]
        if kind == "u2i":
            reco = model.recommend(
                users=targers,
                dataset=self.datasets[dataset_key],
                k=2,
                filter_viewed=False,
                add_rank_col=False,
            )
            reco.rename(columns={Columns.User: "target"}, inplace=True)
        elif kind == "i2i":
            reco = model.recommend_to_items(
                target_items=targers,
                dataset=self.datasets[dataset_key],
                k=2,
                add_rank_col=False,
                filter_itself=False,
            )
            reco.rename(columns={Columns.TargetItem: "target"}, inplace=True)
        else:
            raise ValueError(f"Unexpected kind {kind}")
        reco = reco.astype({Columns.Score: np.float64})
        return reco

    def _assert_reco_equal(self, actual: pd.DataFrame, expected: pd.DataFrame) -> None:
        np.testing.assert_array_equal(actual["target"].values, expected["target"].values)
        np.testing.assert_array_equal(actual[Columns.Item].values, expected[Columns.Item].values)
        np.testing.assert_allclose(actual[Columns.Score].values, expected[Columns.Score].values)

    @pytest.mark.parametrize("dataset_key", ("no_features", "with_features"))
    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot", "hot_warm", "hot_cold", "hot_warm_cold"))
    def test_all_models_works_for_hot(self, dataset_key: str, kind: str, model_key: str) -> None:
        targets = self.hots[kind]
        reco = self._get_reco(targets, model_key, dataset_key, kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12],
                Columns.Score: [0.1, 0.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("dataset_key", ("no_features", "with_features"))
    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot_cold", "hot_warm_cold"))
    def test_cold_models_work_for_cold(self, dataset_key: str, kind: str, model_key: str) -> None:
        targets = self.colds[kind]
        reco = self._get_reco(targets, model_key, dataset_key, kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12],
                Columns.Score: [2.1, 2.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot_warm", "hot_warm_cold"))
    def test_warm_models_work_for_warm_with_features(self, kind: str, model_key: str) -> None:
        targets = self.warms[kind]
        reco = self._get_reco(targets, model_key, "with_features", kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12],
                Columns.Score: [1.1, 1.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot_cold", "hot_warm_cold"))
    def test_cold_models_work_for_warm_without_features(self, kind: str, model_key: str) -> None:
        targets = self.warms[kind]
        reco = self._get_reco(targets, model_key, "no_features", kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12],
                Columns.Score: [2.1, 2.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_cold_only_model_works_for_warm_with_features(self, kind: str) -> None:
        targets = self.warms[kind]
        reco = self._get_reco(targets, "hot_cold", "with_features", kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12],
                Columns.Score: [2.1, 2.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_full_model_works_for_all_with_features(self, kind: str) -> None:
        targets = self.hots[kind] + self.warms[kind] + self.colds[kind]
        reco = self._get_reco(targets, "hot_warm_cold", "with_features", kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12, 11, 12, 11, 12],
                Columns.Score: [0.1, 0.2, 1.1, 1.2, 2.1, 2.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_full_model_works_for_all_without_features(self, kind: str) -> None:
        targets = self.hots[kind] + self.warms[kind] + self.colds[kind]
        reco = self._get_reco(targets, "hot_warm_cold", "no_features", kind)
        excepted = pd.DataFrame(
            {
                "target": np.repeat(targets, 2),
                Columns.Item: [11, 12, 11, 12, 11, 12],
                Columns.Score: [0.1, 0.2, 2.1, 2.2, 2.1, 2.2],
            }
        )
        self._assert_reco_equal(reco, excepted)

    @pytest.mark.parametrize("dataset_key", ("no_features", "with_features"))
    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot", "hot_warm"))
    def test_not_cold_models_raise_on_cold(self, dataset_key: str, kind: str, model_key: str) -> None:
        targets = self.colds[kind]
        with pytest.raises(ValueError, match="doesn't support recommendations for cold"):
            self._get_reco(targets, model_key, dataset_key, kind)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_warm_only_model_raises_on_warm_without_features(self, kind: str) -> None:
        targets = self.warms[kind]
        with pytest.raises(ValueError, match="doesn't support recommendations for cold"):
            self._get_reco(targets, "hot_warm", "no_features", kind)
