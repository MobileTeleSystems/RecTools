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

# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools.dataset import IdMap


class TestIdMap:
    def setup(self) -> None:
        self.to_internal = pd.Series([1, 0, 2], index=["b", "c", "a"])

    def test_creation(self) -> None:
        id_map = IdMap(self.to_internal)
        pd.testing.assert_series_equal(id_map.to_internal, self.to_internal)

    def test_internal_validation(self) -> None:
        to_internal = pd.Series([1, 3, 2], index=["b", "c", "a"])
        with pytest.raises(ValueError):
            IdMap(to_internal)

    def test_external_validation(self) -> None:
        to_internal = pd.Series([1, 0, 2], index=["b", "b", "a"])
        with pytest.raises(ValueError):
            IdMap(to_internal)

    def test_from_values_creation(self) -> None:
        values = ["b", "a", "a", "c"]
        id_map = IdMap.from_values(values)
        expected = pd.Series([0, 1, 2], index=["a", "b", "c"])
        pd.testing.assert_series_equal(id_map.to_internal, expected)

    def test_from_dict_creation(self) -> None:
        existing_mapping: tp.Dict[tp.Hashable, int] = {"a": 0, "b": 1, "c": 3, "e": 2}
        id_map = IdMap.from_dict(existing_mapping)
        expected = pd.Series([0, 1, 3, 2], index=["a", "b", "c", "e"])
        pd.testing.assert_series_equal(id_map.to_internal, expected)

    def test_to_external(self) -> None:
        id_map = IdMap(self.to_internal)
        expected = pd.Series(["b", "c", "a"], index=[1, 0, 2])
        pd.testing.assert_series_equal(id_map.to_external, expected)

    def test_internal_ids(self) -> None:
        id_map = IdMap(self.to_internal)
        expected = np.array([1, 0, 2])
        np.testing.assert_equal(id_map.internal_ids, expected)

    def test_external_ids(self) -> None:
        id_map = IdMap(self.to_internal)
        expected = np.array(["b", "c", "a"])
        np.testing.assert_equal(id_map.external_ids, expected)

    def test_get_sorted_inner(self) -> None:
        id_map = IdMap(self.to_internal)
        expected = np.array([0, 1, 2])
        np.testing.assert_equal(id_map.get_sorted_internal(), expected)

    def test_get_extern_sorted_by_inner(self) -> None:
        id_map = IdMap(self.to_internal)
        expected = np.array(["c", "b", "a"])
        np.testing.assert_equal(id_map.get_external_sorted_by_internal(), expected)
