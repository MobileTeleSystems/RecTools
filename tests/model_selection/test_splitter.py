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
from rectools.dataset import Interactions
from rectools.model_selection import Splitter


class TestSplitter:
    @pytest.fixture
    def interactions(self) -> Interactions:
        df = pd.DataFrame(
            [
                [1, 1, 1, "2021-09-01"],
                [1, 2, 1, "2021-09-02"],
                [2, 1, 1, "2021-09-02"],
                [2, 2, 1, "2021-09-03"],
                [3, 2, 1, "2021-09-03"],
                [3, 3, 1, "2021-09-03"],
                [3, 4, 1, "2021-09-04"],
                [1, 2, 1, "2021-09-04"],
                [3, 1, 1, "2021-09-05"],
                [4, 2, 1, "2021-09-05"],
                [3, 3, 1, "2021-09-06"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        ).astype({Columns.Datetime: "datetime64[ns]"})
        return Interactions(df)

    def test_not_implemented(self, interactions: Interactions) -> None:
        s = Splitter()
        with pytest.raises(NotImplementedError):
            for _, _, _ in s.split(interactions):
                pass

    @pytest.mark.parametrize("collect_fold_stats", [False, True])
    def test_not_defined_fields(self, interactions: Interactions, collect_fold_stats: bool) -> None:
        s = Splitter()
        train_idx = np.array([1, 2, 3, 5, 7, 8])
        test_idx = np.array([4, 6, 9, 10])
        fold_info = {"info_from_split": 123}
        train_idx_new, test_idx_new, _ = s.filter(interactions, collect_fold_stats, train_idx, test_idx, fold_info)

        assert np.array_equal(train_idx, train_idx_new)
        assert sorted(test_idx_new) == [4]
