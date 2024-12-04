import pickle

from rectools.models.base import ModelBase
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
