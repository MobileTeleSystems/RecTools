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
Metrics calculation tools (:mod:`rectools.metrics`).
===============================================================

Tools for fast and convenient calculation of
different recommendation metrics.

Metrics
-------
`metrics.Precision`
`metrics.Recall`
`metrics.F1Beta`
`metrics.Accuracy`
`metrics.MCC`
`metrics.MAP`
`metrics.NDCG`
`metrics.MRR`
`metrics.MeanInvUserFreq`
`metrics.IntraListDiversity`
`metrics.AvgRecPopularity`
`metrics.Serendipity`

Tools
-----
`metrics.calc_metrics` - calculate a set of metrics efficiently
`metrics.PairwiseDistanceCalculator`
`metrics.PairwiseHammingDistanceCalculator`
`metrics.SparsePairwiseHammingDistanceCalculator`
"""

from .classification import MCC, Accuracy, F1Beta, Precision, Recall
from .distances import (
    PairwiseDistanceCalculator,
    PairwiseHammingDistanceCalculator,
    SparsePairwiseHammingDistanceCalculator,
)
from .diversity import IntraListDiversity
from .novelty import MeanInvUserFreq
from .popularity import AvgRecPopularity
from .ranking import MAP, MRR, NDCG
from .scoring import calc_metrics
from .serendipity import Serendipity

__all__ = (
    "Precision",
    "Recall",
    "F1Beta",
    "Accuracy",
    "MCC",
    "MAP",
    "NDCG",
    "MRR",
    "MeanInvUserFreq",
    "IntraListDiversity",
    "AvgRecPopularity",
    "Serendipity",
    "calc_metrics",
    "PairwiseDistanceCalculator",
    "PairwiseHammingDistanceCalculator",
    "SparsePairwiseHammingDistanceCalculator",
)
