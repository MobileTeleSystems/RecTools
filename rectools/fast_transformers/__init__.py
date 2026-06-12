#  Copyright 2026 MTS (Mobile Telesystems)
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

"""Fast Transformers: flat sequential recommenders without ItemNet hierarchy."""

from .metrics import compute_metrics, hitrate_at_k, mrr_at_k, ndcg_at_k
from .net import FlatSASRecNet, SASRecBlock
from .preprocessing import SequenceBatchDataset, align_embeddings, build_sequences
from .unisrec import UniSRecLightning, UniSRecModel, UniSRecNet
from .unisrec.net import FeedForward

__all__ = [
    "build_sequences",
    "align_embeddings",
    "SequenceBatchDataset",
    "FlatSASRecNet",
    "SASRecBlock",
    "UniSRecNet",
    "FeedForward",
    "UniSRecLightning",
    "UniSRecModel",
    "hitrate_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "compute_metrics",
]
