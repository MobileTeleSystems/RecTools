#  Copyright 2022-2024 MTS (Mobile Telesystems)
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
`metrics.MAP`
`metrics.NDCG`
`metrics.MRR`
`metrics.HitRate`
`metrics.PartialAUC`
`metrics.PAP`
`metrics.F1Beta`
`metrics.Accuracy`
`metrics.MCC`
`metrics.MeanInvUserFreq`
`metrics.IntraListDiversity`
`metrics.AvgRecPopularity`
`metrics.Serendipity`
`metrics.Intersection`
`metrics.SufficientReco`
`metrics.UnrepeatedReco`
`metrics.CoveredUsers`

Tools
-----
`metrics.calc_metrics` - calculate a set of metrics efficiently
`metrics.PairwiseDistanceCalculator`
`metrics.PairwiseHammingDistanceCalculator`
`metrics.SparsePairwiseHammingDistanceCalculator`
"""

from .auc import PAP, PartialAUC
from .classification import MCC, Accuracy, F1Beta, HitRate, Precision, Recall
from .distances import (
    PairwiseDistanceCalculator,
    PairwiseHammingDistanceCalculator,
    SparsePairwiseHammingDistanceCalculator,
)
from .diversity import IntraListDiversity
from .dq import CoveredUsers, SufficientReco, UnrepeatedReco
from .intersection import Intersection
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
    "HitRate",
    "MAP",
    "NDCG",
    "PartialAUC",
    "PAP",
    "MRR",
    "MeanInvUserFreq",
    "IntraListDiversity",
    "AvgRecPopularity",
    "Serendipity",
    "calc_metrics",
    "PairwiseDistanceCalculator",
    "PairwiseHammingDistanceCalculator",
    "SparsePairwiseHammingDistanceCalculator",
    "Intersection",
    "SufficientReco",
    "UnrepeatedReco",
    "CoveredUsers",
)
