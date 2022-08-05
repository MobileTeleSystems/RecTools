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

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import RandomModel

from .data import DATASET, INTERACTIONS
from .utils import assert_second_fit_refits_model


class TestRandomModel:
    @pytest.fixture
    def dataset(self) -> Dataset:
        return DATASET

    @pytest.mark.parametrize("items_to_recommend", (None, [11, 12, 13]))
    def test_basic(self, dataset: Dataset, items_to_recommend: tp.Optional[tp.List[tp.Any]]) -> None:
        model = RandomModel().fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
        )
        assert actual.columns.tolist() == Columns.Recommendations
        assert actual[Columns.User].tolist() == [10, 10, 20, 20]
        assert actual[Columns.Rank].tolist() == [1, 2, 1, 2]
        assert actual[Columns.Score].tolist() == [2, 1, 2, 1]
        assert set(actual[Columns.Item]) <= set(items_to_recommend or INTERACTIONS[Columns.Item])

    @pytest.mark.parametrize("items_to_recommend", (None, [11, 12, 13]))
    def test_when_required_more_items_than_exists(
        self,
        dataset: Dataset,
        items_to_recommend: tp.Optional[tp.List[tp.Any]],
    ) -> None:
        model = RandomModel().fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=10,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
        )
        item_set = set(items_to_recommend or INTERACTIONS[Columns.Item])
        n_items = len(item_set)
        assert actual.columns.tolist() == Columns.Recommendations
        assert actual[Columns.User].tolist() == np.repeat([10, 20], n_items).tolist()
        assert actual[Columns.Rank].tolist() == np.tile(np.arange(n_items) + 1, 2).tolist()
        assert actual[Columns.Score].tolist() == np.tile(np.arange(n_items, 0, -1), 2).tolist()
        for user in [10, 20]:
            assert set(actual.loc[actual[Columns.User] == user, Columns.Item].tolist()) == item_set

    @pytest.mark.parametrize("items_to_recommend", (None, [11, 12, 15, 17]))
    def test_viewed_filtering(self, dataset: Dataset, items_to_recommend: tp.Optional[tp.List[tp.Any]]) -> None:
        model = RandomModel().fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=True,
            items_to_recommend=items_to_recommend,
        )
        assert actual.columns.tolist() == Columns.Recommendations
        assert actual[Columns.User].tolist() == [10, 10, 20, 20]
        assert actual[Columns.Rank].tolist() == [1, 2, 1, 2]
        assert actual[Columns.Score].tolist() == [2, 1, 2, 1]
        assert set(actual.loc[actual[Columns.User] == 10, Columns.Item].tolist()) <= {13, 15, 17}
        assert set(actual.loc[actual[Columns.User] == 20, Columns.Item].tolist()) <= {14, 15, 17}

    @pytest.mark.parametrize("n_items", (10, 10000))
    def test_random_state_works(self, n_items: int) -> None:
        interactions = pd.DataFrame(
            {
                Columns.User: np.random.choice([10, 20], n_items * 2),
                Columns.Item: np.tile(np.arange(n_items), 2),
                Columns.Weight: 1,
                Columns.Datetime: 1,
            }
        )
        dataset = Dataset.construct(interactions)
        model = RandomModel(random_state=42).fit(dataset)
        reco_1 = model.recommend(users=np.array([10, 20]), dataset=dataset, k=5, filter_viewed=False)
        reco_2 = model.recommend(users=np.array([10, 20]), dataset=dataset, k=5, filter_viewed=False)
        pd.testing.assert_frame_equal(reco_1, reco_2)

    @pytest.mark.parametrize("filter_itself", (True, False))
    @pytest.mark.parametrize("whitelist", (None, np.array([11, 12, 13])))
    def test_i2i(self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray]) -> None:
        model = RandomModel().fit(dataset)
        actual = model.recommend_to_items(
            target_items=[11],
            dataset=dataset,
            k=10,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        assert actual.columns.tolist() == Columns.RecommendationsI2I
        if whitelist is None:
            expected_reco_set = set(dataset.item_id_map.external_ids)
        else:
            expected_reco_set = set(whitelist)
        if filter_itself:
            expected_reco_set -= {11}
        assert actual[Columns.TargetItem].tolist() == [11] * len(expected_reco_set)
        assert actual[Columns.Rank].tolist() == list(range(1, len(expected_reco_set) + 1))
        assert set(actual[Columns.Score]) <= set(np.arange(dataset.item_id_map.external_ids.size) + 1)
        assert set(actual[Columns.Item]) == expected_reco_set

    def test_second_fit_refits_model(self, dataset: Dataset) -> None:
        model = RandomModel(random_state=1)
        assert_second_fit_refits_model(model, dataset)
