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

import numpy as np
import pytest

from rectools.exceptions import NotFittedError
from rectools.models.base import ModelBase

from .data import DATASET


def test_raise_when_recommend_u2i_from_not_fitted() -> None:
    model = ModelBase()
    with pytest.raises(NotFittedError):
        model.recommend(
            users=np.array([]),
            dataset=DATASET,
            k=5,
            filter_viewed=False,
        )


def test_raise_when_recommend_i2i_from_not_fitted() -> None:
    model = ModelBase()
    with pytest.raises(NotFittedError):
        model.recommend_to_items(
            target_items=np.array([]),
            dataset=DATASET,
            k=5,
        )


def test_raise_when_recommend_for_nonexistent_user() -> None:
    model = ModelBase()
    model.is_fitted = True
    with pytest.raises(KeyError):
        model.recommend(
            users=np.array([10, 90]),
            dataset=DATASET,
            k=5,
            filter_viewed=False,
        )


def test_raise_when_recommend_for_nonexistent_item() -> None:
    model = ModelBase()
    model.is_fitted = True
    with pytest.raises(KeyError):
        model.recommend_to_items(
            target_items=np.array([11, 19]),
            dataset=DATASET,
            k=5,
        )


@pytest.mark.parametrize("k", (-4, 0))
def test_raise_when_k_is_not_positive_u2i(k: int) -> None:
    model = ModelBase()
    model.is_fitted = True
    with pytest.raises(ValueError):
        model.recommend(
            users=np.array([10, 20]),
            dataset=DATASET,
            k=k,
            filter_viewed=True,
        )


@pytest.mark.parametrize("k", (-4, 0))
def test_raise_when_k_is_not_positive_i2i(k: int) -> None:
    model = ModelBase()
    model.is_fitted = True
    with pytest.raises(ValueError):
        model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=DATASET,
            k=k,
        )
