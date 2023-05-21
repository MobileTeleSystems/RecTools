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

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.sar import SarWrapper
from .data import DATASET

class TestSarWrapper :
    @pytest.fixture
    def dataset(self) -> Dataset:
        return DATASET
    
    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [11, 17, 11, 17]
                    }
                )
            ),
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [13, 17, 15, 17]
                    }
                )
            )
        )
    )
    def test_recomend(
        self,
        dataset: Dataset,
        filter_viewed: bool,
        expected: pd.DataFrame) -> None:
        sar = SarWrapper()
        sar.fit(dataset)
        actual = sar.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
            add_rank_col=False
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)

    @pytest.mark.parametrize(
        "expected",
        (
            (
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [12, 14, 11, 14]
                    }
                )
            )
        )
    )
    def test_i2i(
        self, dataset: Dataset, expected: pd.DataFrame) -> None:
        model = SarWrapper().fit(dataset)
        actual = model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=dataset,
            k=2,
            filter_itself=False,
            add_rank_col=False
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)