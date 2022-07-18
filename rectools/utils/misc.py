#  Copyright 2022 MTS (Mobile Telesystems)
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
from itertools import tee

import numpy as np
from typeguard import check_type

T = tp.TypeVar("T")


def pairwise(iterable: tp.Iterable[T]) -> tp.Iterable[tp.Tuple[T, T]]:
    """
    Make iterator of pairs of neighbours in sequence.

    s -> (s0,s1), (s1,s2), (s2, s3), ...

    Parameters
    ----------
    iterable: iterable
        Any sequence

    Returns
    -------
    iterable
        Sequence of pairs.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def log_at_base(arr: np.ndarray, base: float) -> np.ndarray:
    """
    Calculate logarithm at any base.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array.
    base : float
        Logarithm base.

    Returns
    -------
    np.ndarray
        Logarithms of given numbers.

    """
    return np.log(arr) / np.log(base)


AnyType = tp.Any


def is_instance_of_type(obj: tp.Any, type_: AnyType) -> bool:
    """
    Check that `type_` is type of `obj`.

    Works also for generics.

    Parameters
    ----------
    obj : any
        Any object
    type_ : any
        Any type.

    Returns
    -------
    bool
        Whether `type_` is type of `obj`.
    """
    try:
        check_type("", obj, type_)
        return True
    except TypeError:
        return False


def is_instance(obj: tp.Any, types: tp.Union[AnyType, tp.Tuple[AnyType, ...]]) -> bool:
    """
    Analogue of `isinstance(obj, types)` but also works for generics.

    Parameters
    ----------
    obj : any
        Any object
    types : any | tuple(any, ...)
        Any type or tuple of types.

    Returns
    -------
    bool
        Whether `types` (or some of `types` if `types` is tuple) is type of `obj`.
    """
    if not isinstance(types, tuple):
        types = (types,)

    for type_ in types:
        if is_instance_of_type(obj, type_):
            return True
    return False


def select_by_type(
    objects: tp.Dict[tp.Any, tp.Any],
    types: tp.Union[AnyType, tp.Tuple[AnyType, ...]],
) -> tp.Dict[tp.Any, tp.Any]:
    """
    Select objects from `objects` that type is in `types`.

    Parameters
    ----------
    objects : dict
        Dictionary of objects.
    types: any | tuple(any, ...)
        Any type or tuple of types.

    Returns
    -------
    dict
        Dictionary of objects from `objects` where type of value is `types` (or in `types` if `types` is tuple).
    """
    selected = {k: obj for k, obj in objects.items() if is_instance(obj, types)}
    return selected
