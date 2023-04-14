#  Copyright 2023 MTS (Mobile Telesystems)
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

from rectools.model_selection.utils import get_not_seen_mask


class TestGetNotSeenMask:
    @pytest.mark.parametrize(
        "train_users,train_items,test_users,test_items,expected",
        (
            ([], [], [], [], []),
            ([1, 2], [10, 20], [], [], []),
            ([], [], [1, 2], [10, 20], [True, True]),
            ([1, 2, 3, 4, 2, 3], [10, 20, 30, 40, 22, 30], [1, 2, 3, 2], [10, 20, 33, 20], [False, False, True, False]),
        ),
    )
    def test_correct(
        self,
        train_users: tp.List[int],
        train_items: tp.List[int],
        test_users: tp.List[int],
        test_items: tp.List[int],
        expected: tp.List[bool],
    ) -> None:
        actual = get_not_seen_mask(*(np.array(a) for a in (train_users, train_items, test_users, test_items)))
        np.testing.assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "train_users,train_items,test_users,test_items,expected_error_type",
        (
            ([1], [10, 20], [1], [10], ValueError),
            ([1], [10], [1, 2], [10], ValueError),
        ),
    )
    def test_with_incorrect_arrays(
        self,
        train_users: tp.List[int],
        train_items: tp.List[int],
        test_users: tp.List[int],
        test_items: tp.List[int],
        expected_error_type: tp.Type[Exception],
    ) -> None:
        with pytest.raises(expected_error_type):
            get_not_seen_mask(*(np.array(a) for a in (train_users, train_items, test_users, test_items)))
