#  Copyright 2024 MTS (Mobile Telesystems)
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

import pandas as pd

from rectools import Columns
from rectools.metrics import Completeness

RECO = pd.DataFrame(
    {
        Columns.User: [1, 1, 1, 2, 2, 3, 4],
        Columns.Item: [1, 2, 3, 1, 2, 1, 1],
        Columns.Rank: [1, 2, 3, 1, 2, 3, 1],
    }
)

class TestCompleteness:
    def setup_method(self) -> None:
        self.metric = Completeness(k=2)

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series(
            [1., 1., 0., 0.5],
            index=pd.Series([1, 2, 3, 4], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(RECO), expected_metric_per_user)
        assert self.metric.calc(RECO) == expected_metric_per_user.mean()
