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

import pandas as pd

from rectools.semantic.data_handling.loo_split import loo_split


class TestLooSplit:
    def test_basic_split(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "item_id": [10, 20, 30, 40, 50, 60],
                "timestamp": [1, 2, 3, 4, 5, 6],
            }
        )
        train, val, test = loo_split(df)
        # Train: all but last 2 per user
        assert len(train) == 2  # 1 per user
        # Val: all but last 1 per user
        assert len(val) == 4  # 2 per user
        # Test: all interactions
        assert len(test) == 6

    def test_users_with_fewer_than_3_interactions_dropped(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2],
                "item_id": [10, 20, 30, 40, 50],
                "timestamp": [1, 2, 3, 4, 5],
            }
        )
        train, val, test = loo_split(df)
        # User 2 has only 2 interactions, should be dropped
        assert 2 not in train["user_id"].values
        assert 2 not in val["user_id"].values
        assert 2 not in test["user_id"].values

    def test_sorts_by_timestamp(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "item_id": [30, 10, 20],
                "timestamp": [3, 1, 2],
            }
        )
        train, val, test = loo_split(df)
        # After sorting by timestamp: items are [10, 20, 30]
        # Train = first item, val = first 2 items, test = all 3
        assert train["item_id"].tolist() == [10]
        assert set(val["item_id"].tolist()) == {10, 20}
        assert set(test["item_id"].tolist()) == {10, 20, 30}

    def test_no_timestamp_column(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "item_id": [10, 20, 30],
            }
        )
        train, val, test = loo_split(df)
        assert len(train) == 1
        assert len(val) == 2
        assert len(test) == 3

    def test_all_users_dropped(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [10, 20, 30, 40],
                "timestamp": [1, 2, 3, 4],
            }
        )
        train, val, test = loo_split(df)
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0
