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

# pylint: disable=attribute-defined-outside-init

import numpy as np
import pandas as pd

from rectools import Columns
from rectools.metrics import CatalogCoverage


class TestCatalogCoverage:
    def setup_method(self) -> None:
        self.metric = CatalogCoverage(k=2)
        self.reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 3, 4],
                Columns.Item: [1, 2, 3, 1, 2, 1, 1],
                Columns.Rank: [1, 2, 3, 1, 1, 3, 2],
            }
        )

    def test_calc(self) -> None:
        catalog = np.arange(5)
        expected = 0.4
        assert self.metric.calc(self.reco, catalog) == expected
