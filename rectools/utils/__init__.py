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
Utils (:mod:`rectools.utils`)
=============================

Inner helpers.


Tools
-----
`utils.fast_2d_int_unique`
`utils.fast_isin`
`utils.fast_isin_for_sorted_test_elements`
`utils.get_element_ids`
`utils.get_from_series_by_index`
`utils.pairwise`
`utils.log_at_base`
`utils.is_instance`
`utils.select_by_type`
"""

from .array_set_ops import (
    fast_2d_2col_int_unique,
    fast_2d_int_unique,
    fast_isin,
    fast_isin_for_sorted_test_elements,
    isin_2d_int,
)
from .indexing import get_element_ids, get_from_series_by_index
from .misc import is_instance, log_at_base, pairwise, select_by_type

__all__ = (
    "fast_2d_int_unique",
    "fast_2d_2col_int_unique",
    "fast_isin",
    "fast_isin_for_sorted_test_elements",
    "isin_2d_int",
    "get_element_ids",
    "get_from_series_by_index",
    "pairwise",
    "log_at_base",
    "is_instance",
    "select_by_type",
)
