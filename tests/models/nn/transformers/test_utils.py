#  Copyright 2022-2025 MTS (Mobile Telesystems)
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
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Interactions
from rectools.models.nn.transformers.utils import leave_one_out_mask


class TestLeaveOneOutMask:
    def setup_method(self) -> None:
        np.random.seed(32)

    @pytest.fixture
    def interactions(self) -> Interactions:
        df = pd.DataFrame(
            [
                [1, 1, 1, "2021-09-01"],  # 0
                [1, 2, 1, "2021-09-02"],  # 1
                [1, 1, 1, "2021-09-03"],  # 2
                [1, 2, 1, "2021-09-04"],  # 3
                [1, 3, 1, "2021-09-05"],  # 4
                [2, 3, 1, "2021-09-06"],  # 5
                [2, 2, 1, "2021-08-20"],  # 6
                [2, 2, 1, "2021-09-06"],  # 7
                [3, 1, 1, "2021-09-05"],  # 8
                [1, 6, 1, "2021-09-05"],  # 9
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]"})
        return Interactions(df)

    @pytest.mark.parametrize(
        "swap_interactions,expected_val_index, expected_val_item, val_users",
        (
            ([9, 9], [7, 8, 9], 6, None),
            ([9, 9], [7, 8, 9], 6, 3),
            ([9, 9], [8, 9], 6, 2),
            ([4, 9], [7, 8, 9], 3, None),
            ([4, 9], [7, 8, 9], 3, 3),
            ([4, 9], [8, 9], 3, 2),
            ([7, 7], [7, 8], 2, [2, 3]),
            ([5, 7], [7, 8], 3, [2, 3]),
            ([8, 8], [8], 1, [3]),
        ),
    )
    def test_correct_last_interactions(
        self,
        interactions: Interactions,
        swap_interactions: tuple,
        expected_val_index: tp.List[int],
        expected_val_item: int,
        val_users: tp.Optional[tp.List[int]],
    ) -> None:
        interactions_df = interactions.df
        swap_revert = swap_interactions[::-1]
        interactions_df.iloc[swap_interactions] = interactions_df.iloc[swap_revert]
        val_mask = leave_one_out_mask(interactions_df, val_users)
        val_interactions = interactions_df[val_mask]
        last_index = max(swap_interactions)

        assert list(val_interactions.index) == expected_val_index
        assert val_interactions.loc[last_index, [Columns.Item]].values[0] == expected_val_item
