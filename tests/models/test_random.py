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

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import RandomModel
from rectools.models.random import _RandomGen, _RandomSampler

from .data import DATASET, INTERACTIONS
from .utils import (
    assert_default_config_and_default_model_params_are_the_same,
    assert_dumps_loads_do_not_change_model,
    assert_get_config_and_from_config_compatibility,
    assert_second_fit_refits_model,
)


class TestRandomSampler:

    def test_sample_small_n(self) -> None:
        gen = _RandomGen(42)
        sampler = _RandomSampler(np.arange(10), gen)
        sampled = sampler.sample(5)
        np.testing.assert_array_equal(sampled, [1, 0, 4, 9, 6])

    def test_sample_big_n(self) -> None:
        gen = _RandomGen(42)
        sampler = _RandomSampler(np.arange(100), gen)
        sampled = sampler.sample(30)
        np.testing.assert_array_equal(
            sampled,
            [
                95, 42, 74, 65, 6, 7, 16, 97, 67, 54, 11, 15, 80, 44, 88,
                94, 34, 61, 39, 32, 99, 53, 40, 45, 55, 87, 60, 47, 76, 63,
            ],
        )  # fmt: skip

    @pytest.mark.parametrize("n", (10, 50))
    def test_different_results_after_sequential_inits_with_same_gen(self, n: int) -> None:
        gen = _RandomGen(42)
        values = np.arange(100)
        sampled_1 = _RandomSampler(values, gen).sample(n)
        sampled_2 = _RandomSampler(values, gen).sample(n)
        assert not np.array_equal(sampled_1, sampled_2)


class TestRandomModel:
    @pytest.fixture
    def dataset(self) -> Dataset:
        return DATASET

    @pytest.mark.parametrize("items_to_recommend", (None, [11, 12, 13]))
    def test_basic(self, items_to_recommend: tp.Optional[tp.List[tp.Any]]) -> None:
        user_features = pd.DataFrame(
            {
                "id": [10, 50],
                "feature": ["f1", "f1"],
                "value": [1, 1],
            }
        )
        dataset = Dataset.construct(INTERACTIONS, user_features_df=user_features)
        model = RandomModel().fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20, 50, 60]),
            dataset=dataset,
            k=2,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
        )
        assert actual.columns.tolist() == Columns.Recommendations
        assert actual[Columns.User].tolist() == [10, 10, 20, 20, 50, 50, 60, 60]
        assert actual[Columns.Rank].tolist() == [1, 2, 1, 2, 1, 2, 1, 2]
        assert actual[Columns.Score].tolist() == [2, 1, 2, 1, 2, 1, 2, 1]
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

        model_1 = RandomModel(random_state=42).fit(dataset)
        reco_1 = model_1.recommend(users=np.array([10, 20]), dataset=dataset, k=5, filter_viewed=False)
        model_2 = RandomModel(random_state=42).fit(dataset)
        reco_2 = model_2.recommend(users=np.array([10, 20]), dataset=dataset, k=5, filter_viewed=False)
        pd.testing.assert_frame_equal(reco_1, reco_2)

    @pytest.mark.parametrize("filter_itself", (True, False))
    @pytest.mark.parametrize("whitelist", (None, [11, 12, 13]))
    def test_i2i(self, filter_itself: bool, whitelist: tp.Optional[tp.List[tp.Any]]) -> None:
        item_features = pd.DataFrame(
            {
                "id": [11, 16],
                "feature": ["f1", "f1"],
                "value": [1, 1],
            }
        )
        dataset = Dataset.construct(INTERACTIONS, item_features_df=item_features)
        model = RandomModel().fit(dataset)
        actual = model.recommend_to_items(
            target_items=[11, 12, 16, 18],
            dataset=dataset,
            k=2,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        assert actual.columns.tolist() == Columns.RecommendationsI2I

        assert actual[Columns.TargetItem].tolist() == [11, 11, 12, 12, 16, 16, 18, 18]
        assert actual[Columns.Rank].tolist() == [1, 2, 1, 2, 1, 2, 1, 2]
        # Items that aren't present in interactions but have features can also be used
        assert set(actual[Columns.Item]) <= set(whitelist or dataset.item_id_map.external_ids.tolist())

        if not filter_itself:
            assert actual[Columns.Score].tolist() == [2, 1, 2, 1, 2, 1, 2, 1]
        else:
            # We give scores first then filter items itself, so there can be score k+1
            assert set(actual[Columns.Score]) <= {1, 2, 3}
            assert (actual[Columns.TargetItem] != actual[Columns.Item]).all()

    def test_second_fit_refits_model(self, dataset: Dataset) -> None:
        model = RandomModel(random_state=1)
        assert_second_fit_refits_model(model, dataset)

    def test_dumps_loads(self, dataset: Dataset):
        model = RandomModel()
        model.fit(dataset)
        assert_dumps_loads_do_not_change_model(model, dataset)


class TestRandomModelConfiguration:
    def test_from_config(self) -> None:
        config = {
            "random_state": 32,
            "verbose": 0,
        }
        model = RandomModel.from_config(config)
        assert model.random_state == 32
        assert model.verbose == 0

    @pytest.mark.parametrize("random_state", (None, 42))
    def test_get_config(self, random_state: tp.Optional[int]) -> None:
        model = RandomModel(
            random_state=random_state,
            verbose=1,
        )
        config = model.get_config()
        expected = {
            "random_state": random_state,
            "verbose": 1,
        }
        assert config == expected

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(self, simple_types: bool) -> None:
        initial_config = {
            "random_state": 32,
            "verbose": 0,
        }
        assert_get_config_and_from_config_compatibility(RandomModel, DATASET, initial_config, simple_types)

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        default_config: tp.Dict[str, int] = {}
        model = RandomModel()
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
