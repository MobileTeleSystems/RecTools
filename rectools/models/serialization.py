#  Copyright 2024-2025 MTS (Mobile Telesystems)
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

import pickle
import typing as tp

from pydantic import TypeAdapter

from rectools.models.base import ModelBase, ModelClass, ModelConfig
from rectools.utils.misc import unflatten_dict
from rectools.utils.serialization import FileLike, read_bytes


def load_model(f: FileLike) -> ModelBase:
    """
    Load model from file.

    Parameters
    ----------
    f : str or Path or file-like object
        Path to file or file-like object.

    Returns
    -------
    model
        Model instance.
    """
    data = read_bytes(f)
    loaded = pickle.loads(data)
    return loaded


def model_from_config(config: tp.Union[dict, ModelConfig]) -> ModelBase:
    """
    Create model from config.

    Parameters
    ----------
    config : dict or ModelConfig
        Model config.

    Returns
    -------
    model
        Model instance.
    """
    if isinstance(config, dict):
        model_cls = config.get("cls")
        model_cls = TypeAdapter(tp.Optional[ModelClass]).validate_python(model_cls)
    else:
        model_cls = config.cls

    if model_cls is None:
        raise ValueError("`cls` must be provided in the config to load the model")

    return model_cls.from_config(config)


def model_from_params(params: dict, sep: str = ".") -> ModelBase:
    """
    Create model from dict of parameters.
    Same as `from_config` but accepts flat dict.

    Parameters
    ----------
    params : dict
        Model parameters as a flat dict with keys separated by `sep`.
    sep : str, default "."
        Separator for nested keys.

    Returns
    -------
    model
        Model instance.
    """
    config_dict = unflatten_dict(params, sep=sep)
    return model_from_config(config_dict)
