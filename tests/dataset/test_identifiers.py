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
        self.external_ids = np.array(["b", "c", "a"])
        self.id_map = IdMap(self.external_ids)

    def test_creation(self) -> None:
        np.testing.assert_equal(self.id_map.external_ids, self.external_ids)

    def test_from_values_creation(self) -> None:
        values = ["b", "c", "c", "a"]
        id_map = IdMap.from_values(values)
        np.testing.assert_equal(id_map.external_ids, self.external_ids)

    def test_from_dict_creation(self) -> None:
        existing_mapping: tp.Dict[tp.Hashable, int] = {"a": 2, "b": 0, "c": 1}
        id_map = IdMap.from_dict(existing_mapping)
        np.testing.assert_equal(id_map.external_ids, self.external_ids)

    @pytest.mark.parametrize("existing_mapping", ({"a": "0", "b": "1"}, {"a": 1, "b": 2}, {"a": 0, "b": 2}))
    def test_from_dict_creation_with_incorrect_internal_ids(self, existing_mapping: tp.Dict[tp.Hashable, int]) -> None:
        with pytest.raises(ValueError):
            IdMap.from_dict(existing_mapping)

    def test_size(self) -> None:
        assert self.id_map.size == 3

    @pytest.mark.parametrize("external_ids", (np.array(["a", "b"]), np.array([1, 2]), np.array([1, 2], dtype="O")))
    def test_external_dtype(self, external_ids: np.ndarray) -> None:
        id_map = IdMap(external_ids)
        assert id_map.external_dtype == external_ids.dtype

        id_map = IdMap.from_values(external_ids)
        assert id_map.external_dtype == external_ids.dtype

    def test_to_internal(self) -> None:
        actual = self.id_map.to_internal
        expected = pd.Series([0, 1, 2], index=self.external_ids)
        pd.testing.assert_series_equal(actual, expected)

    def test_to_external(self) -> None:
        actual = self.id_map.to_external
        expected = pd.Series(self.external_ids, index=pd.RangeIndex(0, 3))
        pd.testing.assert_series_equal(actual, expected, check_index_type=True)

    def test_internal_ids(self) -> None:
        actual = self.id_map.internal_ids
        expected = np.array([0, 1, 2])
        np.testing.assert_equal(actual, expected)

    def test_get_sorted_inner(self) -> None:
        actual = self.id_map.get_sorted_internal()
        expected = np.array([0, 1, 2])
        np.testing.assert_equal(actual, expected)

    def test_get_extern_sorted_by_inner(self) -> None:
        actual = self.id_map.get_external_sorted_by_internal()
        np.testing.assert_equal(actual, self.external_ids)

    def test_convert_to_internal(self) -> None:
        with pytest.raises(KeyError):
            self.id_map.convert_to_internal(["b", "a", "e", "a"])

    def test_convert_to_internal_not_strict(self) -> None:
        actual = self.id_map.convert_to_internal(["b", "a", "e", "a"], strict=False)
        expected = np.array([0, 2, 2])
        np.testing.assert_equal(actual, expected)

    def test_convert_to_internal_with_return_missing(self) -> None:
        # pylint: disable=unpacking-non-sequence
        values, missing = self.id_map.convert_to_internal(["b", "a", "e", "a"], strict=False, return_missing=True)
        np.testing.assert_equal(values, np.array([0, 2, 2]))
        np.testing.assert_equal(missing, np.array(["e"]))

    def test_convert_to_external(self) -> None:
        with pytest.raises(KeyError):
            self.id_map.convert_to_external([0, 2, 4, 2])

    def test_convert_to_external_not_strict(self) -> None:
        actual = self.id_map.convert_to_external([0, 2, 4, 2], strict=False)
        expected = np.array(["b", "a", "a"])
        np.testing.assert_equal(actual, expected)

    def test_convert_to_external_with_return_missing(self) -> None:
        # pylint: disable=unpacking-non-sequence
        values, missing = self.id_map.convert_to_external([0, 2, 4, 2], strict=False, return_missing=True)
        np.testing.assert_equal(values, np.array(["b", "a", "a"]))
        np.testing.assert_equal(missing, np.array([4]))

    def test_add_ids(self) -> None:
        new_id_map = self.id_map.add_ids(["d", "e", "c", "d"])
        actual = new_id_map.external_ids
        expected = np.array(["b", "c", "a", "d", "e"])
        np.testing.assert_equal(actual, expected)

    def test_add_ids_with_raising_on_repeating_ids(self) -> None:
        with pytest.raises(ValueError):
            self.id_map.add_ids(["d", "e", "c", "d"], raise_if_already_present=True)
