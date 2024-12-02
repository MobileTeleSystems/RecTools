import pickle
import typing as tp
from rectools.models.base import ModelBase, ModelConfig, deserialize_model_class
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
    def raise_on_none(model_cls: tp.Any) -> None:
        if model_cls is None:
            raise ValueError("`cls` must be provided in the config to load the model")

    if isinstance(config, dict):
        model_cls = deserialize_model_class(config.get("cls"))
        raise_on_none(model_cls)
        if not issubclass(model_cls, ModelBase):
            raise TypeError("`cls` must be (or refer to) a subclass of `ModelBase`")
    else:
        model_cls = config.cls
        raise_on_none(model_cls)
    
    return model_cls.from_config(config)