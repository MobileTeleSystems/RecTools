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
import pytest

from rectools.utils import fast_isin, fast_isin_for_sorted_test_elements, get_element_ids, get_from_series_by_index


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

    def test_output_for_empty_test_elements(self) -> None:
        actual = fast_isin_for_sorted_test_elements(np.array([10, 6]), np.array([]))
        np.testing.assert_array_equal(actual, np.array([False, False]))


class TestGetElementIds:
    def test_when_elements_present(self) -> None:
        actual = get_element_ids(np.array([2, 5, 3, 8]), np.array([2, 3, 4, 8, 1, 5]))
        np.testing.assert_equal(actual, np.array([0, 5, 1, 3]))

    def test_raises_when_elements_not_present(self) -> None:
        with pytest.raises(ValueError):
            get_element_ids(np.array([2, 5, 3, 8]), np.array([3, 4, 8, 1, 5]))

    def test_when_elements_empty(self) -> None:
        actual = get_element_ids(np.array([]), np.array([2, 3, 4, 8, 1, 5]))
        np.testing.assert_equal(actual, np.array([]))

    def test_raises_when_test_elements_empty(self) -> None:
        with pytest.raises(ValueError):
            get_element_ids(np.array([2, 5, 3, 8]), np.array([]))


@pytest.mark.parametrize("index_type", ("int64", "str"))
@pytest.mark.parametrize("value_type", ("int64", "str"))
class TestGetFromSeriesByIndex:
    def test_normal(self, index_type: str, value_type: str) -> None:
        s = pd.Series([40, 20, 40, 10, 30], index=np.array([4, 2, 1, 3, 0], dtype=index_type), dtype=value_type)
        ids = np.array([1, 3, 4], dtype=index_type)
        actual = get_from_series_by_index(s, ids)
        expected = np.array([40, 10, 40], dtype=value_type)
        np.testing.assert_equal(actual, expected)

    def test_raises_when_unknown_object(self, index_type: str, value_type: str) -> None:
        s = pd.Series([40, 20], index=np.array([4, 2], dtype=index_type), dtype=value_type)
        ids = np.array([1, 2, 4], dtype=index_type)
        with pytest.raises(KeyError):
            get_from_series_by_index(s, ids)

    def test_selects_known_objects(self, index_type: str, value_type: str) -> None:
        s = pd.Series([40, 20], index=np.array([4, 2], dtype=index_type), dtype=value_type)
        ids = np.array([2, 4, 1], dtype=index_type)
        actual = get_from_series_by_index(s, ids, strict=False)
        expected = np.array([20, 40], dtype=value_type)
        np.testing.assert_equal(actual, expected)
