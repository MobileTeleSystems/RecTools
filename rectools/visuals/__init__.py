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
Visualization tools (:mod:`rectools.visuals`)
=======================================================

Instruments to visualize recommender models performance

Recos visualization
---------
`visuals.VisualApp` - Jupyter app for visual comparison of recommendations
"""

try:
    from .visual_app import VisualApp
except ImportError:  # pragma: no cover
    from ..compat import VisualApp  # type: ignore

__all__ = ("VisualApp",)
