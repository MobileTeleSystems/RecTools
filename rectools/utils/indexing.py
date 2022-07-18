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

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_object_dtype

from rectools import AnySequence


def fast_isin(elements: np.ndarray, test_elements: np.ndarray) -> np.ndarray:
    """
    Effective version of `np.isin` that works well even if arrays have `object` types.

    Parameters
    ----------
    elements : np.ndarray
        Array of elements that you want to check.
    test_elements : np.ndarray
        The values against which to test each value of `elements`.

    Returns
    -------
    np.ndarray
        Boolean array with same shape as `elements`.
    """
    if is_object_dtype(elements) or is_object_dtype(test_elements):
        res = pd.Series(elements.astype("O")).isin(test_elements.astype("O")).values
    else:
        res = np.isin(elements, test_elements)
    return res


def fast_isin_for_sorted_test_elements(
    elements: np.ndarray,
    sorted_test_elements: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    """
    Effective version of `np.isin` for case when array with test elements is sorted.

    Works only with 1d arrays.

    Parameters
    ----------
    elements : np.ndarray
        Array of elements that you want to check.
    sorted_test_elements : np.ndarray
        The values against which to test each value of `elements`.
        Must be sorted (in other cases result will be incorrect, no error will be raised).
    invert : bool, default False
        If True, the values in the returned array are inverted,
        as if calculating *`element` not in `test_elements`*.
        Faster than using negation after getting result.

    Returns
    -------
    np.ndarray
        Boolean array with same shape as `elements`.
    """
    ss_result_left = np.searchsorted(sorted_test_elements, elements, side="left")
    ss_result_right = np.searchsorted(sorted_test_elements, elements, side="right")
    if invert:
        return ss_result_right != ss_result_left + 1
    return ss_result_right == ss_result_left + 1


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


def get_from_series_by_index(series: pd.Series, ids: AnySequence, strict: bool = True) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Array of values.

    Raises
    ------
    KeyError
        If `strict` is ``True`` and at least one element of `ids` not in `s.index`.
    """
    ids_arr = np.asarray(ids)
    exists_mask = fast_isin(ids_arr, series.index.values)

    if strict:
        if not exists_mask.all():
            raise KeyError("Some indices not exists")
        known_ids = ids_arr
    else:
        known_ids = ids_arr[exists_mask]
    selected = series[known_ids].values
    return selected
