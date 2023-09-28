import typing as tp

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_object_dtype
from scipy import sparse


def _consolidate_2d_int_array(arr: np.ndarray) -> np.ndarray:
    dtype = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    consolidated = np.ascontiguousarray(arr).view(dtype).ravel()
    return consolidated


def fast_2d_int_unique(arr: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Return unique rows of 2d numpy array and inverse indices.
    Works only for integer arrays.
    Equivalent of `np.unique(arr, axis=0, return_inverse=True)` but faster.

    Parameters
    ----------
    arr : np.ndarray
        Array of integers with 2 dimensions.

    Returns
    -------
    np.ndarray
        Unique rows of arr, shape (n_unique_rows, m)

    Examples
    --------
    >>> arr = np.array(
    ...     [
    ...         [10, 30, 12],
    ...         [10, 30, 555],
    ...         [10, 30, 12],
    ...         [10, 30, 12],
    ...         [1, 2, 3],
    ...         [1, 2, 3],
    ...     ]
    ... )
    >>> unq, inv_ids = fast_2d_int_unique(arr)
    >>> unq
    array([[  1,   2,   3],
           [ 10,  30,  12],
           [ 10,  30, 555]])
    >>> inv_ids
    array([1, 2, 1, 1, 0, 0])

    Notes
    -----
    Taken from https://github.com/numpy/numpy/issues/11136
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Only integer array is allowed")
    if arr.ndim != 2:
        raise ValueError("Only 2d array is allowed")

    arr_dtype, arr_shape = arr.dtype, arr.shape
    consolidated = _consolidate_2d_int_array(arr)
    unq_consolidated, inv_ids = np.unique(consolidated, return_inverse=True)
    del consolidated
    unq_arr = unq_consolidated.view(arr_dtype).reshape(len(unq_consolidated), arr_shape[1])
    return unq_arr, inv_ids


def fast_2d_2col_int_unique(arr: np.ndarray) -> np.ndarray:
    """
    Return unique rows of 2d numpy array with 2 columns.
    Works only for integer arrays.
    Equivalent of `np.unique(arr, axis=0)` but much faster.

    Parameters
    ----------
    arr : np.ndarray
        Array of integers with shape (n, 2).

    Returns
    -------
    np.ndarray
        Unique rows of arr, shape (n_unique_rows, 2), sorted by 1 then 2 column.

    Examples
    --------
    >>> arr = np.array(
    ...     [
    ...         [10, 30],
    ...         [10, 555],
    ...         [10, 30],
    ...         [10, 30],
    ...         [1, 2],
    ...         [1, 2],
    ...     ]
    ... )
    >>> fast_2d_2col_int_unique(arr)
    array([[  1,   2],
           [ 10,  30],
           [ 10,  555]])
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("Only integer array is allowed")
    if arr.ndim != 2:
        raise ValueError("Only 2d array is allowed")
    if arr.shape[1] != 2:
        raise ValueError("Array must have 2 columns")

    if arr.shape[0] == 0:
        return arr

    csr = sparse.csr_matrix(
        (
            np.ones(len(arr), dtype=bool),
            (
                arr[:, 0],
                arr[:, 1],
            ),
        ),
    )
    coo = csr.tocoo(copy=False)
    res = np.array([coo.row, coo.col], dtype=arr.dtype).T
    return res


def fast_isin(elements: np.ndarray, test_elements: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Effective version of `np.isin` that works well even if arrays have `object` types.

    Parameters
    ----------
    elements : np.ndarray
        Array of elements that you want to check.
    test_elements : np.ndarray
        The values against which to test each value of `elements`.
    invert : bool, default ``False``
        If True, the values in the returned array are inverted, as if
        calculating `element not in test_elements`

    Examples
    --------
    >>> fast_isin(np.array(["a", "b", "c"]), np.array(["c", "x", "a", "y"]))
    array([ True, False,  True])

    Returns
    -------
    np.ndarray
        Boolean array with same shape as `elements`.
    """
    if is_object_dtype(elements) or is_object_dtype(test_elements):
        res = pd.Series(elements.astype("O")).isin(test_elements.astype("O")).values
        if invert:
            res = ~res
    else:
        res = np.isin(elements, test_elements, invert=invert)
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

    Examples
    --------
    >>> fast_isin_for_sorted_test_elements(np.array([5, 2, 3]), np.array([1, 3, 4, 5, 6]))
    array([ True, False,  True])

    Returns
    -------
    np.ndarray
        Boolean array with same shape as `elements`.
    """
    if sorted_test_elements.size == 0:
        if invert:
            return np.ones(elements.size, dtype=bool)
        return np.zeros(elements.size, dtype=bool)

    searched_indices = np.searchsorted(sorted_test_elements, elements, side="left")

    # If there are some values in `elements` bigger than the max value in `sorted_test_elements`,
    # they will get index equal to `sorted_test_elements.size`
    searched_indices = np.minimum(searched_indices, sorted_test_elements.size - 1)

    found_elements = sorted_test_elements[searched_indices]
    if invert:
        return found_elements != elements
    return found_elements == elements


def isin_2d_int(
    elements: np.ndarray,
    test_elements: np.ndarray,
    invert: bool = False,
    assume_unique: bool = False,
) -> np.ndarray:
    """
    Check for every row of `elements` array if it presents in `test_elements`.
    Version of `np.isin` that works with 2d integer arrays.

    Parameters
    ----------
    elements : np.ndarray
        Elements to check. Arrays of integers with 2 dimensions.
    test_elements : np.ndarray
        The values against which to test each value of `element`.
        Arrays of integers with 2 dimensions with same type and same number of columns as `elements`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted, as if
        calculating `element not in test_elements`. Default is False.

    Examples
    --------
    >>> elements = np.array(
    ...     [
    ...         [10, 30],
    ...         [1, 2],
    ...         [21, 23],
    ...     ]
    ... )
    >>> test_elements = np.array(
    ...     [
    ...         [4, 5],
    ...         [21, 23],
    ...         [100, 200],
    ...         [10, 30],
    ...     ]
    ... )
    >>> isin_2d_int(elements, test_elements)
    array([ True, False,  True])

    Returns
    -------
    np.ndarray
        Array with shape (n_rows_in_elements,).
    """
    if elements.dtype != test_elements.dtype:
        raise TypeError("Arrays must be of the same types")
    if not np.issubdtype(elements.dtype, np.integer):
        raise TypeError("Only integer arrays is allowed")
    if elements.ndim != 2 or test_elements.ndim != 2:
        raise ValueError("Only 2d arrays are allowed")
    if elements.shape[1] != test_elements.shape[1]:
        raise ValueError("Arrays must have same columns number")

    consolidated_elements = _consolidate_2d_int_array(elements)
    consolidated_test_elements = _consolidate_2d_int_array(test_elements)
    res = np.isin(consolidated_elements, consolidated_test_elements, invert=invert, assume_unique=assume_unique)
    return res
