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

import typing as tp

import numpy as np
import pytest

from rectools.utils.array_set_ops import (
    fast_2d_2col_int_unique,
    fast_2d_int_unique,
    fast_isin,
    fast_isin_for_sorted_test_elements,
    isin_2d_int,
)


class TestFast2dIntUnique:
    @pytest.mark.parametrize(
        "arr",
        (
            np.array([], dtype=int).reshape((0, 2)),
            np.array([[1, 10]]),
            np.array([[1, 10], [2, 20]]),
            np.array([[1, 10], [1, 10]]),
            np.array([[1, 10], [2, 20], [1, 10], [2, 20]]),
            np.array([[1], [2], [1]]),
            np.array([[1, 2, 3], [1, 2, 3], [10, 20, 30]]),
        ),
    )
    def test_fast_2d_int_unique(self, arr: np.ndarray) -> None:
        actual_unq, actual_inv = fast_2d_int_unique(arr)
        expected_unq, expected_inv = np.unique(arr, axis=0, return_inverse=True)
        np.testing.assert_equal(actual_unq, expected_unq)
        np.testing.assert_equal(actual_inv, expected_inv)

    @pytest.mark.parametrize(
        "arr,expected_error_type,expected_error_text",
        (
            (np.array([[1.0, 10.0], [1, 10]]), TypeError, "integer"),
            (np.array([[[1, 10], [1, 10]]]), ValueError, "2d"),
        ),
    )
    def test_with_incorrect_array(
        self,
        arr: np.ndarray,
        expected_error_type: tp.Type[Exception],
        expected_error_text: str,
    ) -> None:
        with pytest.raises(expected_error_type, match=expected_error_text):
            fast_2d_int_unique(arr)


class TestFast2d2colIntUnique:
    @pytest.mark.parametrize(
        "arr",
        (
            np.array([], dtype=int).reshape((0, 2)),
            np.array([[1, 10]]),
            np.array([[1, 10], [2, 20]]),
            np.array([[1, 10], [1, 10]]),
            np.array([[1, 10], [2, 20], [1, 10], [2, 20]]),
        ),
    )
    def test_correct(self, arr: np.ndarray) -> None:
        actual = fast_2d_2col_int_unique(arr)
        expected = np.unique(arr, axis=0)
        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "arr,expected_error_type,expected_error_text",
        (
            (np.array([[1.0, 10.0], [1, 10]]), TypeError, "integer"),
            (np.array([[[1, 10], [1, 10]]]), ValueError, "2d"),
            (np.array([[1, 10, 100], [1, 10, 10]]), ValueError, "2 columns"),
        ),
    )
    def test_with_incorrect_array(
        self,
        arr: np.ndarray,
        expected_error_type: tp.Type[Exception],
        expected_error_text: str,
    ) -> None:
        with pytest.raises(expected_error_type, match=expected_error_text):
            fast_2d_2col_int_unique(arr)


@pytest.mark.parametrize(
    "elements,test_elements,expected",
    (
        (np.array([2, 6, 4]), np.array([2, 5, 4, 1]), np.array([True, False, True])),
        (np.array(["2", "6", "4"]), np.array(["2", "5", "4", "1"]), np.array([True, False, True])),
        (np.array([2, 6, 4], dtype="O"), np.array([2, 5, 4, 1], dtype="O"), np.array([True, False, True])),
        (np.array([2, 6, 4]), np.array([2, 5, 4, 1], dtype="O"), np.array([True, False, True])),
        (np.array([2, 6, 4], dtype="O"), np.array([2, 5, 4, 1]), np.array([True, False, True])),
        (np.array([2, 6, 4]), np.array(["2", "5", "4", "1"]), np.array([False, False, False])),
        (np.array(["2", "6", "4"]), np.array([2, 5, 4, 1]), np.array([False, False, False])),
        (np.array([2, 6, 4]), np.array(["2", "5", "4", "1"], dtype="O"), np.array([False, False, False])),
        (np.array([2, 6, 4], dtype="O"), np.array(["2", "5", "4", "1"], dtype="O"), np.array([False, False, False])),
        (np.array(["2", "6", "4"], dtype="O"), np.array([2, 5, 4, 1]), np.array([False, False, False])),
        (np.array([]), np.array([]), np.array([], dtype=bool)),
        (np.array([]), np.array([2, 5, 4]), np.array([], dtype=bool)),
        (np.array([2, 6, 4]), np.array([]), np.array([False, False, False])),
    ),
)
@pytest.mark.parametrize("invert", (True, False))
@pytest.mark.filterwarnings("ignore:elementwise comparison failed")
def test_fast_isin(elements: np.ndarray, test_elements: np.ndarray, expected: np.ndarray, invert: bool) -> None:
    actual = fast_isin(elements, test_elements, invert=invert)
    if invert:
        expected = ~expected
    np.testing.assert_array_equal(actual, expected)


class TestFastIsinForSortedTestElements:
    @pytest.mark.parametrize("invert", (True, False))
    def test_when_arrays_not_empty(self, invert: bool) -> None:
        actual = fast_isin_for_sorted_test_elements(
            np.array([10, 7, 3, 8, 4, 12]),
            np.array([4, 6, 8, 10]),
            invert=invert,
        )
        expected = np.array([True, False, False, True, True, False])
        if invert:
            expected = ~expected
        np.testing.assert_array_equal(actual, expected)

    def test_for_empty_elements(self) -> None:
        actual = fast_isin_for_sorted_test_elements(np.array([]), np.array([4, 8, 10]))
        np.testing.assert_array_equal(actual, np.array([]))

    @pytest.mark.parametrize("invert", (True, False))
    def test_output_for_empty_test_elements(self, invert: bool) -> None:
        actual = fast_isin_for_sorted_test_elements(np.array([10, 6]), np.array([]), invert=invert)
        expected = np.array([False, False])
        if invert:
            expected = ~expected
        np.testing.assert_array_equal(actual, expected)


class TestIsin2dInt:
    @pytest.mark.parametrize(
        "ar1,ar2,expected",
        (
            (np.array([], dtype=int).reshape(0, 2), np.array([], dtype=int).reshape(0, 2), np.array([], dtype=bool)),
            (np.array([[1, 10]]), np.array([], dtype=int).reshape(0, 2), np.array([False])),
            (np.array([], dtype=int).reshape(0, 2), np.array([[1, 10]]), np.array([], dtype=bool)),
            (np.array([[2, 20]]), np.array([[1, 10]]), np.array([False])),
            (np.array([[1, 10]]), np.array([[1, 10]]), np.array([True])),
            (
                np.array([[1, 10], [3, 30], [2, 20], [1, 10], [3, 30], [4, 40]]),
                np.array([[2, 10], [2, 20], [1, 10], [2, 20], [3, 10], [5, 50]]),
                np.array([True, False, True, True, False, False]),
            ),
            (np.array([[1], [2]]), np.array([[1], [3]]), np.array([True, False])),
            (np.array([[1, 10, 100], [2, 20, 200]]), np.array([[1, 10, 100], [3, 30, 300]]), np.array([True, False])),
        ),
    )
    @pytest.mark.parametrize("invert", (True, False))
    @pytest.mark.parametrize("assume_unique", (True, False))
    def test_correct(
        self,
        ar1: np.ndarray,
        ar2: np.ndarray,
        invert: bool,
        assume_unique: bool,
        expected: np.ndarray,
    ) -> None:
        if assume_unique:
            ar1, unq_idx = np.unique(ar1, axis=0, return_index=True)
            expected = expected[unq_idx]
            ar2 = np.unique(ar2, axis=0)
        actual = isin_2d_int(ar1, ar2, invert=invert, assume_unique=assume_unique)
        if invert:
            expected = ~expected
        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "elements,test_elements,expected_error_type,expected_error_text",
        (
            (np.array([[1, 10]]), np.array([[1.0, 10.0]]), TypeError, "same types"),
            (np.array([[1.0, 10.0]]), np.array([[1.0, 10.0]]), TypeError, "integer"),
            (np.array([[[1, 10]]]), np.array([[[1, 10]]]), ValueError, "2d"),
            (np.array([[1, 10, 100]]), np.array([[1, 10]]), ValueError, "same columns number"),
        ),
    )
    def test_with_incorrect_arrays(
        self,
        elements: np.ndarray,
        test_elements: np.ndarray,
        expected_error_type: tp.Type[Exception],
        expected_error_text: str,
    ) -> None:
        with pytest.raises(expected_error_type, match=expected_error_text):
            isin_2d_int(elements, test_elements)
