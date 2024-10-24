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

# pylint: disable=attribute-defined-outside-init

import typing as tp
from datetime import timedelta
import warnings

import numpy as np
import pandas as pd
import pytest
import typing_extensions as tpe

from rectools import Columns
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.base import (
    ErrorBehaviour,
    FixedColdRecoModelMixin,
    InternalRecoTriplet,
    ModelBase,
    ModelConfig,
    Scores,
    SemiInternalRecoTriplet,
)
from rectools.types import ExternalIds, InternalIds
from rectools.utils.config import BaseConfig

from .data import DATASET, INTERACTIONS


def test_raise_when_recommend_u2i_from_not_fitted() -> None:
    model: ModelBase[ModelConfig] = ModelBase()
    with pytest.raises(NotFittedError):
        model.recommend(
            users=np.array([]),
            dataset=DATASET,
            k=5,
            filter_viewed=False,
        )


def test_raise_when_recommend_i2i_from_not_fitted() -> None:
    model: ModelBase[ModelConfig] = ModelBase()
    with pytest.raises(NotFittedError):
        model.recommend_to_items(
            target_items=np.array([]),
            dataset=DATASET,
            k=5,
        )


@pytest.mark.parametrize("k", (-4, 0))
def test_raise_when_k_is_not_positive_u2i(k: int) -> None:
    model: ModelBase[ModelConfig] = ModelBase()
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
    model: ModelBase[ModelConfig] = ModelBase()
    model.is_fitted = True
    with pytest.raises(ValueError):
        model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=DATASET,
            k=k,
        )


class TestHotWarmCold:
    def setup_method(self) -> None:
        class HotModel(ModelBase):
            recommends_for_cold = False
            recommends_for_warm = False

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
            recommends_for_warm = True

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
            recommends_for_cold = True

            def _recommend_cold(
                self,
                target_ids: np.ndarray,
                dataset: Dataset,
                k: int,
                sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
            ) -> SemiInternalRecoTriplet:
                return (
                    np.repeat(target_ids, k),
                    np.tile(np.arange(k), len(target_ids)),
                    np.tile(np.arange(1, k + 1) * 0.1 + 2, len(target_ids)),
                )

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

    def _get_reco(
        self,
        targets: ExternalIds,
        model_key: str,
        dataset_key: str,
        kind: str,
        on_unsupported_targets: ErrorBehaviour = "raise",
    ) -> pd.DataFrame:
        model = self.models[model_key]
        if kind == "u2i":
            reco = model.recommend(
                users=targets,
                dataset=self.datasets[dataset_key],
                k=2,
                filter_viewed=False,
                add_rank_col=False,
                on_unsupported_targets=on_unsupported_targets,
            )
            reco.rename(columns={Columns.User: "target"}, inplace=True)
        elif kind == "i2i":
            reco = model.recommend_to_items(
                target_items=targets,
                dataset=self.datasets[dataset_key],
                k=2,
                add_rank_col=False,
                filter_itself=False,
                on_unsupported_targets=on_unsupported_targets,
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
    def test_not_cold_models_with_cold_targets_raise(self, dataset_key: str, kind: str, model_key: str) -> None:
        targets = self.colds[kind] + self.hots[kind]
        with pytest.raises(ValueError, match="doesn't support recommendations for cold"):
            self._get_reco(targets, model_key, dataset_key, kind, on_unsupported_targets="raise")

    @pytest.mark.parametrize("dataset_key", ("no_features", "with_features"))
    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot", "hot_warm"))
    def test_not_cold_models_with_cold_targets_ignore(self, dataset_key: str, kind: str, model_key: str) -> None:
        targets = self.colds[kind] + self.hots[kind]
        actual = self._get_reco(targets, model_key, dataset_key, kind, on_unsupported_targets="ignore")
        expected_targets = self.hots[kind]
        expected = self._get_reco(expected_targets, model_key, dataset_key, kind)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("dataset_key", ("no_features", "with_features"))
    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot", "hot_warm"))
    def test_not_cold_models_with_cold_targets_warn(self, dataset_key: str, kind: str, model_key: str) -> None:
        targets = self.colds[kind] + self.hots[kind]
        with warnings.catch_warnings(record=True) as w:
            self._get_reco(targets, model_key, dataset_key, kind, on_unsupported_targets="warn")
            assert len(w) == 1
            for phrase in ("support", "cold"):
                assert phrase in str(w[-1].message)
            assert "warm" not in str(w[-1].message)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_warm_only_model_with_warm_targets_without_features_raise(self, kind: str) -> None:
        targets = self.warms[kind] + self.hots[kind]
        with pytest.raises(ValueError, match="doesn't support recommendations for cold"):
            self._get_reco(targets, "hot_warm", "no_features", kind, on_unsupported_targets="raise")

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_warm_only_model_with_warm_targets_without_features_ignore(self, kind: str) -> None:
        targets = self.warms[kind] + self.hots[kind]

        # ignore
        actual = self._get_reco(targets, "hot_warm", "no_features", kind, on_unsupported_targets="ignore")
        expected_targets = self.hots[kind]
        expected = self._get_reco(expected_targets, "hot_warm", "no_features", kind)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_warm_only_model_with_warm_targets_without_features_warn(self, kind: str) -> None:
        targets = self.warms[kind] + self.hots[kind]
        with warnings.catch_warnings(record=True) as w:
            self._get_reco(targets, "hot_warm", "no_features", kind, on_unsupported_targets="warn")
            assert len(w) == 1
            for phrase in ("support", "cold"):
                assert phrase in str(w[-1].message)
            assert "warm" not in str(w[-1].message)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_hot_only_model_with_warm_targets_raise(self, kind: str) -> None:
        targets = self.warms[kind] + self.hots[kind]
        with pytest.raises(ValueError, match="doesn't support recommendations for warm"):
            self._get_reco(targets, "hot", "with_features", kind, on_unsupported_targets="raise")

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_hot_only_model_with_warm_targets_ignore(self, kind: str) -> None:
        targets = self.warms[kind] + self.hots[kind]
        actual = self._get_reco(targets, "hot", "with_features", kind, on_unsupported_targets="ignore")
        expected_targets = self.hots[kind]
        expected = self._get_reco(expected_targets, "hot", "with_features", kind)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    def test_hot_only_model_with_warm_targets_warn(self, kind: str) -> None:
        targets = self.warms[kind] + self.hots[kind]
        with warnings.catch_warnings(record=True) as w:
            self._get_reco(targets, "hot", "with_features", kind, on_unsupported_targets="warn")
            assert len(w) == 1
            for phrase in ("support", "cold", "warm"):
                assert phrase in str(w[-1].message)

    @pytest.mark.parametrize("dataset_key", ("no_features", "with_features"))
    @pytest.mark.parametrize("kind", ("u2i", "i2i"))
    @pytest.mark.parametrize("model_key", ("hot_cold", "hot_warm_cold"))
    def test_raises_on_incorrect_cold_targets_type(self, dataset_key: str, kind: str, model_key: str) -> None:
        with pytest.raises(TypeError):
            self._get_reco(["some_id"], model_key, dataset_key, kind)


class TestConfiguration:

    def setup_method(self) -> None:
        class SomeModelSubConfig(BaseConfig):
            td: timedelta

        class SomeModelConfig(ModelConfig):
            x: int
            sc: tp.Optional[SomeModelSubConfig] = None

        class SomeModel(ModelBase[SomeModelConfig]):
            config_class = SomeModelConfig

            def __init__(self, x: int, td: tp.Optional[timedelta] = None, verbose: int = 0):
                super().__init__(verbose=verbose)
                self.x = x
                self.td = td

            def _get_config(self) -> SomeModelConfig:
                sc = None if self.td is None else SomeModelSubConfig(td=self.td)
                return SomeModelConfig(x=self.x, sc=sc, verbose=self.verbose)

            @classmethod
            def _from_config(cls, config: SomeModelConfig) -> tpe.Self:
                td = None if config.sc is None else config.sc.td
                return cls(x=config.x, td=td, verbose=config.verbose)

        self.config_class = SomeModelConfig
        self.model_class = SomeModel

    def test_from_pydantic_config(self) -> None:
        config = self.config_class(x=10, verbose=1)
        model = self.model_class.from_config(config)
        assert model.x == 10
        assert model.td is None
        assert model.verbose == 1

    @pytest.mark.parametrize("td", (timedelta(days=2, hours=3), "P2DT3H"))
    def test_from_config_dict(self, td: tp.Union[timedelta, str]) -> None:
        config = {"x": 10, "verbose": 1, "sc": {"td": td}}
        model = self.model_class.from_config(config)
        assert model.x == 10
        assert model.td == timedelta(days=2, hours=3)
        assert model.verbose == 1

    def test_from_config_dict_with_missing_keys(self) -> None:
        config = {"verbose": 1}
        with pytest.raises(ValueError, match="1 validation error for SomeModelConfig\nx\n  Field required"):
            self.model_class.from_config(config)

    def test_from_config_dict_with_extra_keys(self) -> None:
        config = {"x": 10, "extra": "extra"}
        with pytest.raises(
            ValueError, match="1 validation error for SomeModelConfig\nextra\n  Extra inputs are not permitted"
        ):
            self.model_class.from_config(config)

    def test_get_config_pydantic(self) -> None:
        model = self.model_class(x=10, verbose=1)
        config = model.get_config(mode="pydantic")
        assert config == self.config_class(x=10, verbose=1)

    def test_raises_on_pydantic_with_simple_types(self) -> None:
        model = self.model_class(x=10, verbose=1)
        with pytest.raises(ValueError, match="`simple_types` is not compatible with `mode='pydantic'"):
            model.get_config(mode="pydantic", simple_types=True)

    @pytest.mark.parametrize("simple_types, expected_td", ((False, timedelta(days=2, hours=3)), (True, "P2DT3H")))
    def test_get_config_dict(self, simple_types: bool, expected_td: tp.Union[timedelta, str]) -> None:
        model = self.model_class(x=10, verbose=1, td=timedelta(days=2, hours=3))
        config = model.get_config(mode="dict", simple_types=simple_types)
        assert config == {"x": 10, "verbose": 1, "sc": {"td": expected_td}}

    def test_raises_on_incorrect_format(self) -> None:
        model = self.model_class(x=10, verbose=1)
        with pytest.raises(ValueError, match="Unknown mode:"):
            model.get_config(mode="incorrect_mode")  # type: ignore[call-overload]

    @pytest.mark.parametrize("simple_types, expected_td", ((False, timedelta(days=2, hours=3)), (True, "P2DT3H")))
    def test_get_params(self, simple_types: bool, expected_td: tp.Union[timedelta, str]) -> None:
        model = self.model_class(x=10, verbose=1, td=timedelta(days=2, hours=3))
        config = model.get_params(simple_types=simple_types)
        assert config == {"x": 10, "verbose": 1, "sc.td": expected_td}

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_params_with_empty_subconfig(self, simple_types: bool) -> None:
        model = self.model_class(x=10, verbose=1, td=None)
        config = model.get_params(simple_types=simple_types)
        assert config == {"x": 10, "verbose": 1, "sc": None}


class TestFixedColdRecoModelMixin:
    def test_cold_reco_works(self) -> None:
        class ColdRecoModel(FixedColdRecoModelMixin, ModelBase):
            def _get_cold_reco(
                self, dataset: Dataset, k: int, sorted_item_ids_to_recommend: tp.Optional[np.ndarray]
            ) -> tp.Tuple[InternalIds, Scores]:
                return np.arange(k), np.arange(1, k + 1) * 0.1 + 2

        model = ColdRecoModel()

        reco = model._recommend_cold(np.array([10, 11]), DATASET, 2, None)  # pylint: disable=protected-access
        np.testing.assert_array_equal(list(reco[0]), [10, 10, 11, 11])
        np.testing.assert_array_equal(reco[1], [0, 1, 0, 1])
        np.testing.assert_array_equal(reco[2], [2.1, 2.2, 2.1, 2.2])
