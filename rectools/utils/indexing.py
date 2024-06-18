#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

from typing import Tuple, Union

import numpy as np
import pandas as pd

from rectools import AnySequence


def get_element_ids(elements: np.ndarray, test_elements: np.ndarray) -> np.ndarray:
    """
    Find index of each element of `elements` in `test_elements`.

    Similar to `np.searchsorted` but works for any arrays (not only sorted).

    All `elements` must be present in `test_elements`.

    Parameters
    ----------
    elements : np.ndarray
        Elements that indices you want to get.
    test_elements : np.ndarray
        Array in which you want to find indices.

    Returns
    -------
    np.ndarray
        Integer array with same shape as `elements`.

    Raises
    ------
    ValueError
        If there are elements from `elements` which are not in `test_elements`.

    Examples
    --------
    >>> get_element_ids(np.array([50, 20, 30]), np.array([10, 30, 40, 50, 60, 20]))
    array([3, 5, 1])

    """
    sort_test_element_ids = np.argsort(test_elements)
    sorted_test_elements = test_elements[sort_test_element_ids]
    ids_in_sorted_test_elements = np.searchsorted(sorted_test_elements, elements)
    try:
        ids = sort_test_element_ids[ids_in_sorted_test_elements]
    except IndexError:
        raise ValueError("All `elements` must be in `test_elements`")
    if not (test_elements[ids] == elements).all():
        raise ValueError("All `elements` must be in `test_elements`")
    return ids


def get_from_series_by_index(
    series: pd.Series, ids: AnySequence, strict: bool = True, return_missing: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get values from pd.Series by index.

    Analogue to `s[ids]` but it can process cases when ids are not in index.
    Processing is possible in 2 different ways: raise an error or skip.
    `s[ids]` returns NaN for nonexistent values.

    Parameters
    ----------
    series : pd.Series
        `pd.Series` from which values are extracted.
    ids : sequence(int)
        Sequence of indices.
    strict : bool, default True
        - if True, raise KeyError if at least one element of `ids` not in `s.index`;
        - if False, skip nonexistent `ids` and return values only for existent.
    return_missing : bool, default False
        If True, return a tuple of 2 arrays: values and missing indices.
        Works only if `strict` is False.

    Returns
    -------
    np.ndarray
        Array of values.
    np.ndarray, np.ndarray
        Tuple of 2 arrays: values and missing indices.
        Only if `strict` is False and `return_missing` is True.

    Raises
    ------
    KeyError
        If `strict` is ``True`` and at least one element of `ids` not in `s.index`.
    ValueError
        If `strict` and `return_missing` are both ``True``.

    Examples
    --------
    >>> s = pd.Series([10, 20, 30, 40, 50], index=[1, 2, 3, 4, 5])
    >>> get_from_series_by_index(s, [3, 1, 4])
    array([30, 10, 40])

    >>> get_from_series_by_index(s, [3, 7, 4])
    Traceback (most recent call last):
    ...
    KeyError: 'Some indices do not exist'

    >>> get_from_series_by_index(s, [3, 7, 4], strict=False)
    array([30, 40])

    >>> get_from_series_by_index(s, [3, 7, 4], strict=False, return_missing=True)
    (array([30, 40]), array([7]))
    """
    if strict and return_missing:
        raise ValueError("You can't use `strict` and `return_missing` together")

    r = series.reindex(ids)
    if strict:
        if r.isna().any():
            raise KeyError("Some indices do not exist")
    else:
        if return_missing:
            missing = r[r.isna()].index.values
        r.dropna(inplace=True)
    selected = r.astype(series.dtype).values

    if return_missing:
        return selected, missing
    return selected
