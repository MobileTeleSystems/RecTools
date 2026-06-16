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
import pytest

from rectools.semantic.data_handling.k_core import k_core


class TestKCore:
    @pytest.fixture
    def interactions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                "item_id": [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30],
            }
        )

    def test_no_filtering_needed(self, interactions: pd.DataFrame) -> None:
        result = k_core(interactions, user_min_interactions=3, item_min_interactions=3)
        assert len(result) == len(interactions)

    def test_filters_users_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2],
                "item_id": [10, 20, 30, 10],
            }
        )
        result = k_core(df, user_min_interactions=2, item_min_interactions=1)
        assert set(result["user_id"].unique()) == {1}

    def test_filters_items_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "item_id": [10, 20, 10, 20, 10, 30],
            }
        )
        result = k_core(df, user_min_interactions=1, item_min_interactions=2)
        assert 30 not in result["item_id"].values

    def test_iterative_filtering(self) -> None:
        # User 3 has only item 30, but item 30 only has user 3
        # Removing item 30 removes user 3 interactions, then user 3 has 0
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "item_id": [10, 20, 10, 20, 30],
            }
        )
        result = k_core(df, user_min_interactions=2, item_min_interactions=2)
        assert 3 not in result["user_id"].values
        assert 30 not in result["item_id"].values
        assert len(result) == 4

    def test_custom_column_names(self) -> None:
        df = pd.DataFrame(
            {
                "uid": [1, 1, 2, 2],
                "iid": [10, 20, 10, 20],
            }
        )
        result = k_core(df, user_min_interactions=2, item_min_interactions=2, user_col="uid", item_col="iid")
        assert len(result) == 4

    def test_empty_result(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "item_id": [10, 20, 30],
            }
        )
        result = k_core(df, user_min_interactions=2, item_min_interactions=2)
        assert len(result) == 0
