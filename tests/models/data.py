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

import pandas as pd

from rectools import Columns
from rectools.dataset import Dataset

INTERACTIONS = pd.DataFrame(
    [
        [10, 11],
        [10, 12],
        [10, 14],
        [20, 11],
        [20, 12],
        [20, 13],
        [30, 11],
        [30, 12],
        [30, 14],
        [30, 15],
        [40, 11],
        [40, 15],
        [40, 17],
    ],
    columns=Columns.UserItem,
)
INTERACTIONS[Columns.Weight] = 1
INTERACTIONS[Columns.Datetime] = "2021-09-09"

DATASET = Dataset.construct(INTERACTIONS)
