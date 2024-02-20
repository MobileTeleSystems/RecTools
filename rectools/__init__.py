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
MTS Advanced Recommender Systems package for Python
===================================================

RecTools provides a convenient wrappers for popular recommendation
algorithms (ItemKNN, ALS, LightFM, etc.) and offers its own
realisations and optimizations. It also provides tools
for metrics computation, easy data conversion, and preparing
models for production-ready systems.

See https://rectools.readthedocs.io for complete documentation.

Subpackages
-----------
    dataset - Data and  identifiers conversion
    metrics - Metrics calculation
    model_selection - Cross-validation
    models - Recommendation models
    tools - Useful instruments
    visuals - Visualization apps
"""

from .columns import Columns
from .types import AnyIds, AnySequence, ExternalId, ExternalIds, InternalId, InternalIds
from .version import VERSION

__version__ = VERSION

__all__ = (
    "Columns",
    "AnyIds",
    "AnySequence",
    "ExternalId",
    "ExternalIds",
    "InternalId",
    "InternalIds",
    "__version__",
)
