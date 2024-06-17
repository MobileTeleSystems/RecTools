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

import typing as tp

import pytest

from rectools.compat import (
    DSSMModel,
    ItemToItemAnnRecommender,
    ItemToItemVisualApp,
    LightFMWrapperModel,
    MetricsApp,
    UserToItemAnnRecommender,
    VisualApp,
)


@pytest.mark.parametrize(
    "model",
    (
        DSSMModel,
        ItemToItemAnnRecommender,
        UserToItemAnnRecommender,
        LightFMWrapperModel,
        VisualApp,
        ItemToItemVisualApp,
        MetricsApp,
    ),
)
def test_raise_when_model_not_available(
    model: tp.Any,
) -> None:
    with pytest.raises(ImportError):
        model()
