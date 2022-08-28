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

from rectools.utils import get_element_ids, get_from_series_by_index


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


class TestGetFromSeriesByIndex:
    def test_normal(self) -> None:
        s = pd.Series([40, 20, 40, 10, 30], index=[4, 2, 1, 3, 0])
        actual = get_from_series_by_index(s, [1, 3, 4])
        np.testing.assert_equal(actual, np.array([40, 10, 40]))

    def test_raises_when_unknown_object(self) -> None:
        s = pd.Series([40, 20], index=[4, 2])
        with pytest.raises(KeyError):
            get_from_series_by_index(s, [1, 2, 4])

    def test_selects_known_objects(self) -> None:
        s = pd.Series([40, 20], index=[4, 2])
        actual = get_from_series_by_index(s, [2, 4, 1], strict=False)
        np.testing.assert_equal(actual, np.array([20, 40]))
