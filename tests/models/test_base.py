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
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from rectools import Columns
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.base import ModelBase, Scores
from rectools.types import AnyIds, InternalIds

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


class SomeModel(ModelBase):
    def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
        pass

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        return [0, 0, 1], [0, 1, 2], [0.1, 0.2, 0.3]

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        return [0, 0, 1], [0, 1, 2], [0.1, 0.2, 0.3]


def test_recommend_from_internal_ids(mocker: MockerFixture) -> None:
    model = SomeModel().fit(DATASET)
    users = [0, 1, 2]
    items_to_recommend = [0, 1, 2]

    spy = mocker.spy(model, "_recommend_u2i")
    model.recommend(
        users=users,
        dataset=DATASET,
        k=2,
        filter_viewed=False,
        items_to_recommend=items_to_recommend,
        assume_internal_ids=True,
    )

    assert list(spy.call_args.args[0]) == users
    assert list(spy.call_args.args[4]) == items_to_recommend


@pytest.mark.parametrize(
    "users, items_to_recommend",
    (
        (["u1", "u2"], [0, 1]),
        ([0, 1], ["i1", "i2"]),
        (["u1", "u2"], ["i1", "i2"]),
    ),
)
def test_recommend_from_internal_ids_fails_when_not_integer_ids(users: AnyIds, items_to_recommend: AnyIds) -> None:
    model = SomeModel().fit(DATASET)

    with pytest.raises(TypeError):
        model.recommend(
            users=users,
            dataset=DATASET,
            k=2,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
            assume_internal_ids=True,
        )


def test_recommend_returns_internal_ids() -> None:
    model = SomeModel().fit(DATASET)

    reco = model.recommend(
        users=[10, 20],
        dataset=DATASET,
        k=2,
        filter_viewed=False,
        items_to_recommend=[11, 12],
        add_rank_col=False,
        return_internal_ids=True,
    )

    excepted = pd.DataFrame(
        {
            Columns.User: [0, 0, 1],
            Columns.Item: [0, 1, 2],
            Columns.Score: [0.1, 0.2, 0.3],
        }
    )

    pd.testing.assert_frame_equal(reco, excepted)


def test_recommend_to_items_from_internal_ids(mocker: MockerFixture) -> None:
    model = SomeModel().fit(DATASET)
    target_items = [0, 1, 2]
    items_to_recommend = [0, 1, 2]

    spy = mocker.spy(model, "_recommend_i2i")
    model.recommend_to_items(
        target_items=target_items,
        dataset=DATASET,
        k=2,
        items_to_recommend=items_to_recommend,
        assume_internal_ids=True,
    )

    assert list(spy.call_args.args[0]) == target_items
    assert list(spy.call_args.args[3]) == items_to_recommend


@pytest.mark.parametrize(
    "target_items, items_to_recommend",
    (
        (["i1", "i2"], [0, 1]),
        ([0, 1], ["i1", "i2"]),
        (["i1", "i2"], ["i1", "i2"]),
    ),
)
def test_recommend_to_items_from_internal_ids_fails_when_not_integer_ids(
    target_items: AnyIds, items_to_recommend: AnyIds
) -> None:
    model = SomeModel().fit(DATASET)

    with pytest.raises(TypeError):
        model.recommend_to_items(
            target_items=target_items,
            dataset=DATASET,
            k=2,
            items_to_recommend=items_to_recommend,
            assume_internal_ids=True,
        )


def test_recommend_to_items_returns_internal_ids() -> None:
    model = SomeModel().fit(DATASET)

    reco = model.recommend_to_items(
        target_items=[11, 12],
        dataset=DATASET,
        k=2,
        items_to_recommend=[11, 12],
        filter_itself=False,
        add_rank_col=False,
        return_internal_ids=True,
    )

    excepted = pd.DataFrame(
        {
            Columns.TargetItem: [0, 0, 1],
            Columns.Item: [0, 1, 2],
            Columns.Score: [0.1, 0.2, 0.3],
        }
    )

    pd.testing.assert_frame_equal(reco, excepted)
