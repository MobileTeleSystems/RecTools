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

"""
Model selection tools (:mod:`rectools.model_selection`)
=======================================================

Instruments to validate and compare models.

Splitters
---------
`model_selection.Splitter` - base class for all splitters

`model_selection.KFoldSplitter` - split interactions randomly
`model_selection.LastNSplitter` - split interactions by recent activity
`model_selection.TimeRangeSplit` - split interactions by time

Model selection tools
---------------------
`model_selection.cross_validate` - run cross validation on multiple models with multiple metrics
"""

from .cross_validate import cross_validate
from .last_n_split import LastNSplitter
from .random_split import RandomSplitter
from .splitter import Splitter
from .time_split import TimeRangeSplitter

__all__ = (
    "Splitter",
    "RandomSplitter",
    "LastNSplitter",
    "TimeRangeSplitter",
    "cross_validate",
)
