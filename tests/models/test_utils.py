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

import numpy as np
import pytest
from scipy import sparse

from rectools.models.utils import get_viewed_item_ids, recommend_from_scores

_ui = [
    [0, 1, 3],
    [1, 0, 1],
    [0, 0, 0],
    [3, 1, 2],
]


@pytest.mark.parametrize(
    "user_items,user_id,expected",
    (
        (_ui, 0, [1, 2]),
        (_ui, 1, [0, 2]),
        (_ui, 2, []),
        (_ui, 3, [0, 1, 2]),
    ),
)
def test_get_viewed_item_ids(user_items: tp.List[tp.List[int]], user_id: int, expected: tp.List[int]) -> None:
    actual = get_viewed_item_ids(sparse.csr_matrix(user_items), user_id)
    np.testing.assert_equal(expected, actual)


@pytest.mark.parametrize(
    "blacklist,whitelist,all_expected_ids",
    (
        (None, None, np.array([6, 0, 2, 4, 1, 3, 5])),
        (np.array([0, 1, 5, 6]), None, np.array([2, 4, 3])),
        (None, np.array([0, 2, 5, 6]), np.array([6, 0, 2, 5])),
        (np.array([0, 1, 5, 6]), np.array([0, 2, 5, 6]), np.array([2])),
        (np.array([0, 1, 2, 3]), np.array([1, 2, 3]), np.array([], dtype=int)),
    ),
)
@pytest.mark.parametrize("ascending", (True, False))
def test_recommend_from_scores(
    blacklist: np.ndarray, whitelist: np.ndarray, all_expected_ids: np.ndarray, ascending: bool
) -> None:
    if ascending:
        all_expected_ids = all_expected_ids[::-1]
    expected_ids = all_expected_ids[:5]
    input_scores = np.array([10.5, 2, 7, 0, 5, -3, 100])
    actual_ids, actual_scores = recommend_from_scores(input_scores, 5, blacklist, whitelist, ascending)
    np.testing.assert_equal(actual_ids, expected_ids)
    expected_scores = input_scores[expected_ids]
    np.testing.assert_equal(actual_scores, expected_scores)
