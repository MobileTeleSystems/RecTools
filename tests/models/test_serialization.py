#  Copyright 2024 MTS (Mobile Telesystems)
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

import sys
import typing as tp
from tempfile import NamedTemporaryFile

import pytest
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import ItemItemRecommender
from pydantic import ValidationError

try:
    from lightfm import LightFM
except ImportError:
    LightFM = object  # it's ok in case we're skipping the tests


from rectools.metrics import NDCG
from rectools.models import (
    DSSMModel,
    EASEModel,
    ImplicitALSWrapperModel,
    ImplicitBPRWrapperModel,
    ImplicitItemKNNWrapperModel,
    LightFMWrapperModel,
    PopularInCategoryModel,
    PopularModel,
    load_model,
    model_from_config,
)
from rectools.models.base import ModelBase, ModelConfig
from rectools.models.vector import VectorModel

from .utils import get_successors

INTERMEDIATE_MODEL_CLASSES = (VectorModel,)

EXPOSABLE_MODEL_CLASSES = tuple(
    cls
    for cls in get_successors(ModelBase)
    if (cls.__module__.startswith("rectools.models") and cls not in INTERMEDIATE_MODEL_CLASSES)
)
CONFIGURABLE_MODEL_CLASSES = tuple(cls for cls in EXPOSABLE_MODEL_CLASSES if cls not in (DSSMModel,))


def init_default_model(model_cls: tp.Type[ModelBase]) -> ModelBase:
    mandatory_params = {
        ImplicitItemKNNWrapperModel: {"model": ItemItemRecommender()},
        ImplicitALSWrapperModel: {"model": AlternatingLeastSquares()},
        ImplicitBPRWrapperModel: {"model": BayesianPersonalizedRanking()},
        LightFMWrapperModel: {"model": LightFM()},
        PopularInCategoryModel: {"category_feature": "some_feature"},
    }
    params = mandatory_params.get(model_cls, {})
    model = model_cls(**params)
    return model


@pytest.mark.parametrize("model_cls", EXPOSABLE_MODEL_CLASSES)
def test_load_model(model_cls: tp.Type[ModelBase]) -> None:
    model = init_default_model(model_cls)
    with NamedTemporaryFile() as f:
        model.save(f.name)
        loaded_model = load_model(f.name)
    assert isinstance(loaded_model, model_cls)


class CustomModelConfig(ModelConfig):
    some_param: int = 1


class CustomModel(ModelBase[CustomModelConfig]):
    config_class = CustomModelConfig

    def __init__(self, some_param: int = 1, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.some_param = some_param

    @classmethod
    def _from_config(cls, config: CustomModelConfig) -> "CustomModel":
        return cls(some_param=config.some_param, verbose=config.verbose)


class TestModelFromConfig:

    @pytest.mark.parametrize("mode, simple_types", (("pydantic", False), ("dict", False), ("dict", True)))
    @pytest.mark.parametrize("model_cls", CONFIGURABLE_MODEL_CLASSES)
    def test_standard_model_creation(
        self, model_cls: tp.Type[ModelBase], mode: tp.Literal["pydantic", "dict"], simple_types: bool
    ) -> None:
        model = init_default_model(model_cls)
        config = model.get_config(mode=mode, simple_types=simple_types)

        new_model = model_from_config(config)

        assert isinstance(new_model, model_cls)
        assert new_model.get_config(mode=mode, simple_types=simple_types) == config

    @pytest.mark.parametrize(
        "config",
        (
            CustomModelConfig(cls=CustomModel, some_param=2),
            {"cls": "tests.models.test_serialization.CustomModel", "some_param": 2},
        ),
    )
    def test_custom_model_creation(self, config: tp.Union[dict, CustomModelConfig]) -> None:
        model = model_from_config(config)
        assert isinstance(model, CustomModel)
        assert model.some_param == 2

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_fails_on_missing_cls(self, simple_types: bool) -> None:
        model = PopularModel()
        config = model.get_config(mode="dict", simple_types=simple_types)
        config.pop("cls")
        with pytest.raises(ValueError, match="`cls` must be provided in the config to load the model"):
            model_from_config(config)

    @pytest.mark.parametrize("mode, simple_types", (("pydantic", False), ("dict", False), ("dict", True)))
    def test_fails_on_none_cls(self, mode: tp.Literal["pydantic", "dict"], simple_types: bool) -> None:
        model = PopularModel()
        config = model.get_config(mode=mode, simple_types=simple_types)
        if mode == "pydantic":
            config.cls = None  # type: ignore
        else:
            config["cls"] = None  # type: ignore  # pylint: disable=unsupported-assignment-operation
        with pytest.raises(ValueError, match="`cls` must be provided in the config to load the model"):
            model_from_config(config)

    @pytest.mark.parametrize(
        "model_cls_path, error_cls",
        (
            ("nonexistent_module.SomeModel", ModuleNotFoundError),
            ("rectools.models.NonexistentModel", AttributeError),
        ),
    )
    def test_fails_on_nonexistent_cls(self, model_cls_path: str, error_cls: tp.Type[Exception]) -> None:
        config = {"cls": model_cls_path}
        with pytest.raises(error_cls):
            model_from_config(config)

    @pytest.mark.parametrize("model_cls", ("rectools.metrics.NDCG", NDCG))
    def test_fails_on_non_model_cls(self, model_cls: tp.Any) -> None:
        config = {"cls": model_cls}
        with pytest.raises(ValidationError):
            model_from_config(config)

    @pytest.mark.parametrize("mode, simple_types", (("pydantic", False), ("dict", False), ("dict", True)))
    def test_fails_on_incorrect_model_cls(self, mode: tp.Literal["pydantic", "dict"], simple_types: bool) -> None:
        model = PopularModel()
        config = model.get_config(mode=mode, simple_types=simple_types)
        if mode == "pydantic":
            config.cls = EASEModel  # type: ignore
        else:
            if simple_types:
                # pylint: disable=unsupported-assignment-operation
                config["cls"] = "rectools.models.LightFMWrapperModel"  # type: ignore
            else:
                config["cls"] = EASEModel  # type: ignore  # pylint: disable=unsupported-assignment-operation
        with pytest.raises(ValidationError):
            model_from_config(config)

    @pytest.mark.skipif(sys.version_info >= (3, 13), reason="`torch` is not compatible with Python 3.13")
    @pytest.mark.parametrize("model_cls", ("rectools.models.DSSMModel", DSSMModel))
    def test_fails_on_model_cls_without_from_config_support(self, model_cls: tp.Any) -> None:
        config = {"cls": model_cls}
        with pytest.raises(NotImplementedError, match="`from_config` method is not implemented for `DSSMModel` model"):
            model_from_config(config)
