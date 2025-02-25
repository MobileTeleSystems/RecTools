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

# pylint: disable=wrong-import-position

"""
Recommendation models (:mod:`rectools.models.rank`)
==============================================

Rankers to build recs from embeddings.


Rankers
------
`rank.ImplicitRanker`
`rank.TorchRanker`
"""

try:
    from .rank_torch import TorchRanker
except ImportError:  # pragma: no cover
    from .compat import TorchRanker  # type: ignore

from rectools.models.rank.rank import Distance, Ranker
from rectools.models.rank.rank_implicit import ImplicitRanker

__all__ = [
    "TorchRanker",
    "ImplicitRanker",
    "Distance",
    "Ranker",
]
