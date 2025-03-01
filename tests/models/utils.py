#  Copyright 2022-2025 MTS (Mobile Telesystems)
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
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from rectools.dataset import Dataset
from rectools.models.base import ModelBase
from rectools.models.serialization import load_model


def _dummy_func() -> None:
    pass


def assert_second_fit_refits_model(
    model: ModelBase, dataset: Dataset, pre_fit_callback: tp.Optional[tp.Callable[[], None]] = None
) -> None:
    pre_fit_callback = pre_fit_callback or _dummy_func

    pre_fit_callback()
    model_1 = deepcopy(model)
    pre_fit_callback()
    model_1.fit(dataset)

    pre_fit_callback()
    model_2 = deepcopy(model)
    pre_fit_callback()
    model_2.fit(dataset)
    pre_fit_callback()
    model_2.fit(dataset)

    k = dataset.item_id_map.external_ids.size

    reco_u2i_1 = model_1.recommend(dataset.user_id_map.external_ids, dataset, k, False)
    reco_u2i_2 = model_2.recommend(dataset.user_id_map.external_ids, dataset, k, False)
    pd.testing.assert_frame_equal(reco_u2i_1, reco_u2i_2, atol=0.001)

    reco_i2i_1 = model_1.recommend_to_items(dataset.item_id_map.external_ids, dataset, k, False)
    reco_i2i_2 = model_2.recommend_to_items(dataset.item_id_map.external_ids, dataset, k, False)
    pd.testing.assert_frame_equal(reco_i2i_1, reco_i2i_2, atol=0.001)


def assert_dumps_loads_do_not_change_model(
    model: ModelBase,
    dataset: Dataset,
    check_configs: bool = True,
) -> None:
    def get_reco(model: ModelBase) -> pd.DataFrame:
        users = dataset.user_id_map.external_ids[:2]
        return model.recommend(users=users, dataset=dataset, k=2, filter_viewed=False)

    dumped = model.dumps()
    recovered_model = model.__class__.loads(dumped)

    original_model_reco = get_reco(model)
    recovered_model_reco = get_reco(recovered_model)
    pd.testing.assert_frame_equal(recovered_model_reco, original_model_reco)

    if check_configs:
        original_model_config = model.get_config()
        recovered_model_config = recovered_model.get_config()
        assert recovered_model_config == original_model_config


def assert_save_load_do_not_change_model(
    model: ModelBase,
    dataset: Dataset,
    check_configs: bool = True,
) -> None:

    def get_reco(model: ModelBase) -> pd.DataFrame:
        users = dataset.user_id_map.external_ids[:2]
        return model.recommend(users=users, dataset=dataset, k=2, filter_viewed=False)

    with NamedTemporaryFile() as f:
        model.save(f.name)
        recovered_model = load_model(f.name)

    assert isinstance(recovered_model, model.__class__)

    original_model_reco = get_reco(model)
    recovered_model_reco = get_reco(recovered_model)
    pd.testing.assert_frame_equal(recovered_model_reco, original_model_reco)

    if check_configs:
        original_model_config = model.get_config()
        recovered_model_config = recovered_model.get_config()
        assert recovered_model_config == original_model_config


def assert_default_config_and_default_model_params_are_the_same(
    model: ModelBase, default_config: tp.Dict[str, tp.Any]
) -> None:
    model_from_config = model.from_config(default_config)
    assert model_from_config.get_config() == model.get_config()


def assert_get_config_and_from_config_compatibility(
    model: tp.Type[ModelBase], dataset: Dataset, initial_config: tp.Dict[str, tp.Any], simple_types: bool
) -> None:
    def get_reco(model: ModelBase) -> pd.DataFrame:
        return model.fit(dataset).recommend(users=np.array([10, 20]), dataset=dataset, k=2, filter_viewed=False)

    model_1 = model.from_config(initial_config)
    reco_1 = get_reco(model_1)
    config_1 = model_1.get_config(simple_types=simple_types)

    model_2 = model.from_config(config_1)
    reco_2 = get_reco(model_2)
    config_2 = model_2.get_config(simple_types=simple_types)

    assert config_1 == config_2
    pd.testing.assert_frame_equal(reco_1, reco_2)


def get_successors(cls: tp.Type) -> tp.List[tp.Type]:
    successors = []
    subclasses = cls.__subclasses__()
    for subclass in subclasses:
        successors.append(subclass)
        successors.extend(get_successors(subclass))
    return successors
