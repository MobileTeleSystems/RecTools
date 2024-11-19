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

# pylint: disable=wrong-import-position

"""
Two-stage ranking Recommendation models (:mod:`rectools.models.ranking`)
==============================================

`CandidateRankingModel` and helper classes.


Models
------
`models.ranking.CandidateRankingModel`
`models.ranking.CandidateGenerator`
`models.ranking.CandidateFeatureCollector`
`models.ranking.Reranker`
`models.ranking.CatBoostReranker`
"""

from .candidate_ranking import CandidateRankingModel, CandidateGenerator, CandidateFeatureCollector, Reranker

try:
    from .catboost_reranker import CatBoostReranker
except ImportError:  # pragma: no cover
    from ...compat import CatBoostReranker  # type: ignore


__all__ = (
    "CatBoostReranker",
    "Reranker",
    "CandidateRankingModel",
    "CandidateGenerator",
    "CandidateFeatureCollector",
    "Reranker"
)

    
