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
Data conversion tools (:mod:`rectools.dataset`).
==========================================================

Data and identifiers conversion tools for future working with
models.


Data Containers
---------------
`dataset.IdMap` - Mapping between external and internal identifiers.
`dataset.DenseFeatures` - Container for dense features.
`dataset.SparseFeatures` - Container for sparse features.
`dataset.Interactions` - Container for interactions.
`dataset.Dataset` - Container for all data.

"""


from .dataset import Dataset
from .features import DenseFeatures, Features, SparseFeatures
from .identifiers import IdMap
from .interactions import Interactions

__all__ = (
    "Dataset",
    "DenseFeatures",
    "SparseFeatures",
    "Features",
    "IdMap",
    "Interactions",
)
