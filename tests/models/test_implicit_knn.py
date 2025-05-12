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
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, ItemItemRecommender, TFIDFRecommender

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel

from .data import DATASET, INTERACTIONS
from .utils import (
    assert_default_config_and_default_model_params_are_the_same,
    assert_dumps_loads_do_not_change_model,
    assert_get_config_and_from_config_compatibility,
    assert_second_fit_refits_model,
)


class TestImplicitItemKNNWrapperModel:
    @pytest.fixture
    def dataset(self) -> Dataset:
        return DATASET

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [15, 13, 14, 15],
                        Columns.Score: [0.905, 0.674, 1.352, 0.737],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [12, 11, 12, 11],
                        Columns.Score: [2.568, 2.442, 2.503, 2.388],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_basic(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        base_model = TFIDFRecommender(K=5, num_threads=2)
        model = ImplicitItemKNNWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
        )
        expected = expected.astype({Columns.Score: np.float32})
        pd.testing.assert_frame_equal(actual, expected, atol=0.001)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [15, 17, 15, 17],
                        Columns.Score: [0.905, 0.559, 0.737, 0.559],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [11, 15, 11, 15],
                        Columns.Score: [2.442, 0.905, 2.388, 0.737],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        base_model = TFIDFRecommender(K=5, num_threads=2)
        model = ImplicitItemKNNWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([11, 15, 17]),
        )
        expected = expected.astype({Columns.Score: np.float32})
        pd.testing.assert_frame_equal(actual, expected, atol=0.001)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 12, 12, 11],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [12, 14, 11, 14],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                np.array([11, 13, 14]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 14, 11, 14],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_i2i(self, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame) -> None:
        base_model = TFIDFRecommender(K=5, num_threads=2)
        # Recreate dataset to prevent same co-occurrence count between (11, 14) and (11, 15)
        # which leads to different results in the test in Python 3.13
        # This is because numpy.argpartition behavior was changed.
        # See also: https://github.com/MobileTeleSystems/RecTools/pull/227#discussion_r1941872699
        interactions = pd.DataFrame(
            [
                [10, 11],
                [10, 12],
                [10, 14],
                [20, 11],
                [20, 12],
                [20, 13],
                [30, 11],
                [30, 12],
                [30, 14],
                [40, 11],
                [40, 15],
                [40, 17],
            ],
            columns=Columns.UserItem,
        )
        interactions[Columns.Weight] = 1
        interactions[Columns.Datetime] = "2021-09-09"
        dataset = Dataset.construct(interactions)
        model = ImplicitItemKNNWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend_to_items(
            target_items=np.array([11, 12]),
            dataset=dataset,
            k=2,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, dataset: Dataset) -> None:
        base_model = TFIDFRecommender(K=5, num_threads=2)
        model = ImplicitItemKNNWrapperModel(model=base_model)
        assert_second_fit_refits_model(model, dataset)

    @pytest.mark.parametrize(
        "user_features, error_match",
        (
            (None, "doesn't support recommendations for cold users"),
            (
                pd.DataFrame(
                    {
                        "id": [10, 50],
                        "feature": ["f1", "f1"],
                        "value": [1, 1],
                    }
                ),
                "doesn't support recommendations for warm and cold users",
            ),
        ),
    )
    def test_u2i_with_warm_and_cold_users(self, user_features: tp.Optional[pd.DataFrame], error_match: str) -> None:
        dataset = Dataset.construct(INTERACTIONS, user_features_df=user_features)
        base_model = TFIDFRecommender(K=5, num_threads=2)
        model = ImplicitItemKNNWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(ValueError, match=error_match):
            model.recommend(
                users=[10, 20, 50],
                dataset=dataset,
                k=2,
                filter_viewed=False,
            )

    @pytest.mark.parametrize(
        "item_features, error_match",
        (
            (None, "doesn't support recommendations for cold items"),
            (
                pd.DataFrame(
                    {
                        "id": [11, 16],
                        "feature": ["f1", "f1"],
                        "value": [1, 1],
                    }
                ),
                "doesn't support recommendations for warm and cold items",
            ),
        ),
    )
    def test_i2i_with_warm_and_cold_items(self, item_features: tp.Optional[pd.DataFrame], error_match: str) -> None:
        dataset = Dataset.construct(INTERACTIONS, item_features_df=item_features)
        base_model = TFIDFRecommender(K=5, num_threads=2)
        model = ImplicitItemKNNWrapperModel(model=base_model).fit(dataset)
        with pytest.raises(ValueError, match=error_match):
            model.recommend_to_items(
                target_items=[11, 12, 16],
                dataset=dataset,
                k=2,
            )

    def test_base_class(self, dataset: Dataset) -> None:
        # Base class ItemItemRecommender didn't work due to implicit dtype conversion to np.float64
        base_model = ItemItemRecommender(K=5, num_threads=2)
        model = ImplicitItemKNNWrapperModel(model=base_model).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=False,
        )
        expected = pd.DataFrame(
            {
                Columns.User: [10, 10, 20, 20],
                Columns.Item: [11, 12, 11, 12],
                Columns.Score: [9.0, 8.0, 8.0, 7.0],
                Columns.Rank: [1, 2, 1, 2],
            }
        ).astype({Columns.Score: np.float32})
        pd.testing.assert_frame_equal(actual, expected, atol=0.001)

    def test_dumps_loads(self, dataset: Dataset) -> None:
        model = ImplicitItemKNNWrapperModel(model=TFIDFRecommender())
        model.fit(dataset)
        assert_dumps_loads_do_not_change_model(model, dataset)


class CustomKNN(ItemItemRecommender):
    pass


class TestImplicitItemKNNWrapperModelConfiguration:

    @pytest.mark.parametrize(
        "model_class",
        (
            TFIDFRecommender,  # class object
            "ItemItemRecommender",  # keyword
            "TFIDFRecommender",  # keyword
            "CosineRecommender",  # keyword
            "BM25Recommender",  # keyword
            "tests.models.test_implicit_knn.CustomKNN",  # custom class
        ),
    )
    def test_from_config(self, model_class: tp.Union[tp.Type[ItemItemRecommender], str]) -> None:
        inner_model_config: tp.Dict[str, tp.Any] = {"cls": model_class, "K": 5}
        if model_class == "BM25Recommender":
            inner_model_config.update({"K1": 0.33})
        config = {
            "model": inner_model_config,
            "verbose": 1,
        }
        model = ImplicitItemKNNWrapperModel.from_config(config)
        assert model.verbose == 1
        inner_model = model._model  # pylint: disable=protected-access
        assert inner_model.K == 5
        assert inner_model.num_threads == 0
        if model_class == "BM25Recommender":
            assert inner_model.K1 == 0.33
        if isinstance(model_class, str):
            assert inner_model.__class__.__name__ == model_class.split(".")[-1]
        else:
            assert inner_model.__class__ is model_class

    @pytest.mark.parametrize("simple_types", (False, True))
    @pytest.mark.parametrize(
        "model_class, model_class_str",
        (
            (ItemItemRecommender, "ItemItemRecommender"),
            (TFIDFRecommender, "TFIDFRecommender"),
            (CosineRecommender, "CosineRecommender"),
            (BM25Recommender, "BM25Recommender"),
            (CustomKNN, "tests.models.test_implicit_knn.CustomKNN"),
        ),
    )
    def test_to_config(
        self, simple_types: bool, model_class: tp.Type[ItemItemRecommender], model_class_str: str
    ) -> None:
        model = ImplicitItemKNNWrapperModel(
            model=model_class(K=5),
            verbose=1,
        )
        config = model.get_config(simple_types=simple_types)
        expected_inner_model_config: tp.Dict[str, tp.Any] = {
            "cls": model_class if not simple_types else model_class_str,
            "K": 5,
            "num_threads": 0,
        }
        if model_class is BM25Recommender:
            expected_inner_model_config.update(
                {
                    "K1": 1.2,
                    "B": 0.75,
                }
            )
        expected = {
            "cls": "ImplicitItemKNNWrapperModel" if simple_types else ImplicitItemKNNWrapperModel,
            "model": expected_inner_model_config,
            "verbose": 1,
        }
        assert config == expected

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(self, simple_types: bool) -> None:
        initial_config = {
            "model": {
                "cls": TFIDFRecommender,
                "K": 3,
            },
            "verbose": 1,
        }
        assert_get_config_and_from_config_compatibility(
            ImplicitItemKNNWrapperModel, DATASET, initial_config, simple_types
        )

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        default_config: tp.Dict[str, tp.Any] = {"model": {"cls": ItemItemRecommender}}
        model = ImplicitItemKNNWrapperModel(model=ItemItemRecommender())
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
