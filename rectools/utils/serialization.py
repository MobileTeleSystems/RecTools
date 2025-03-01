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

import typing as tp
from pathlib import Path

import numpy as np
import typing_extensions as tpe
from pydantic import BeforeValidator, PlainSerializer

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

DType = tpe.Annotated[
    np.dtype, BeforeValidator(func=np.dtype), PlainSerializer(func=lambda dtp: dtp.name, when_used="json")
]


def read_bytes(f: FileLike) -> bytes:
    """Read bytes from a file."""
    if isinstance(f, (str, Path)):
        data = Path(f).read_bytes()
    else:
        data = f.read()
    return data
