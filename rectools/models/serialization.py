import pickle
import typing as tp

from pydantic import TypeAdapter

from rectools.models.base import ModelBase, ModelConfig, ModelClass
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
    config : ModelConfig
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
