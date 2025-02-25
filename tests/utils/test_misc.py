#  Copyright 2025 MTS (Mobile Telesystems)
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

from rectools.utils.misc import unflatten_dict


class TestUnflattenDict:
    def test_empty(self) -> None:
        assert unflatten_dict({}) == {}

    def test_complex(self) -> None:
        flattened = {
            "a.b": 1,
            "a.c": 2,
            "d": 3,
            "a.e.f": [10, 20],
        }
        excepted = {
            "a": {"b": 1, "c": 2, "e": {"f": [10, 20]}},
            "d": 3,
        }
        assert unflatten_dict(flattened) == excepted

    def test_simple(self) -> None:
        flattened = {
            "a": 1,
            "b": 2,
        }
        excepted = {
            "a": 1,
            "b": 2,
        }
        assert unflatten_dict(flattened) == excepted

    def test_non_default_sep(self) -> None:
        flattened = {
            "a_b": 1,
            "a_c": 2,
            "d": 3,
        }
        excepted = {
            "a": {"b": 1, "c": 2},
            "d": 3,
        }
        assert unflatten_dict(flattened, sep="_") == excepted
