import sys
import typing as tp
from tempfile import NamedTemporaryFile

import pytest
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender

try:
    from lightfm import LightFM
except ImportError:
    LightFM = object  # it's ok in case we're skipping the tests


from rectools.models import (
    ImplicitALSWrapperModel,
    ImplicitItemKNNWrapperModel,
    LightFMWrapperModel,
    PopularInCategoryModel,
    load_model,
)
from rectools.models.base import ModelBase

from .utils import get_final_successors

MODEL_CLASSES = [
    cls
    for cls in get_final_successors(ModelBase)
    if cls.__module__.startswith("rectools.models") and not (sys.version_info >= (3, 12) and cls is LightFMWrapperModel)
]


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
