import typing as tp
from pathlib import Path

import numpy as np
import typing_extensions as tpe
from pydantic import PlainSerializer

FileLike = tp.Union[str, Path, tp.IO[bytes]]

PICKLE_PROTOCOL = 5


def _serialize_random_state(rs: tp.Optional[tp.Union[None, int, np.random.RandomState]]) -> tp.Union[None, int]:
    if rs is None or isinstance(rs, int):
        return rs

    # NOBUG: We can add serialization using get/set_state, but it's not human readable
    raise TypeError("`random_state` must be ``None`` or have ``int`` type to convert it to simple type")


RandomState = tpe.Annotated[
    tp.Union[None, int, np.random.RandomState],
    PlainSerializer(func=_serialize_random_state, when_used="json"),
]


def read_bytes(f: FileLike) -> bytes:
    """Read bytes from a file."""
    if isinstance(f, (str, Path)):
        data = Path(f).read_bytes()
    else:
        data = f.read()
    return data
