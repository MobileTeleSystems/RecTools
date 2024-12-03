import sys
import typing as tp
from tempfile import NamedTemporaryFile

from pydantic import ValidationError
import pytest
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender

try:
    from lightfm import LightFM
except ImportError:
    LightFM = object  # it's ok in case we're skipping the tests


from rectools.models import (
    DSSMModel,
    ImplicitALSWrapperModel,
    ImplicitItemKNNWrapperModel,
    LightFMWrapperModel,
    PopularInCategoryModel,
    PopularModel,
    load_model,
    model_from_config,
)
from rectools.models.base import ModelBase

from .utils import get_final_successors

MODEL_CLASSES = [
    cls
    for cls in get_final_successors(ModelBase)
    if cls.__module__.startswith("rectools.models") and not (sys.version_info >= (3, 12) and cls is LightFMWrapperModel)
]
CONFIGURABLE_MODEL_CLASSES = [cls for cls in MODEL_CLASSES if cls not in (DSSMModel,)]


def init_default_model(model_cls: tp.Type[ModelBase]) -> ModelBase:
    mandatory_params = {
        ImplicitItemKNNWrapperModel: {"model": ItemItemRecommender()},
        ImplicitALSWrapperModel: {"model": AlternatingLeastSquares()},
        LightFMWrapperModel: {"model": LightFM()},
        PopularInCategoryModel: {"category_feature": "some_feature"},
    }
    params = mandatory_params.get(model_cls, {})
    model = model_cls(**params)
    return model


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_load_model(model_cls: tp.Type[ModelBase]) -> None:
    model = init_default_model(model_cls)
    with NamedTemporaryFile() as f:
        model.save(f.name)
        loaded_model = load_model(f.name)
    assert isinstance(loaded_model, model_cls)


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

    def test_custom_model_creation(self) -> None:
        pass

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
            config.cls = None
        else:
            config["cls"] = None
        with pytest.raises(ValueError, match="`cls` must be provided in the config to load the model"):
            model_from_config(config)

    def test_fails_on_nonexistent_cls(self, ) -> None:
        model = PopularModel()
        config = model.get_config(mode=mode, simple_types=simple_types)
        if mode == "pydantic":
            config.cls = None
        else:
            config["cls"] = None

    def test_fails_on_non_model_cls(self) -> None:
        pass

    @pytest.mark.parametrize("mode, simple_types", (("pydantic", False), ("dict", False), ("dict", True)))
    def test_fails_on_incorrect_model_cls(self, mode: tp.Literal["pydantic", "dict"], simple_types: bool) -> None:
        model = PopularModel()
        config = model.get_config(mode=mode, simple_types=simple_types)
        if mode == "pydantic":
            config.cls = LightFMWrapperModel
        else:
            if simple_types:
                config["cls"] = "rectools.models.LightFMWrapperModel"
            else:
                config["cls"] = LightFMWrapperModel
        with pytest.raises(ValidationError):
            model_from_config(config)

    def test_fails_on_model_cls_without_from_config_support(self) -> None:
        # DSSM
        pass