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

import typing as tp

import numpy as np

ExternalId = tp.Hashable
ExternalIds = tp.Union[tp.Sequence[ExternalId], np.ndarray]
InternalId = int
InternalIds = tp.Union[tp.Sequence[InternalId], np.ndarray]
AnyIds = tp.Union[ExternalIds, InternalIds]
AnySequence = tp.Union[tp.Sequence[tp.Any], np.ndarray]
