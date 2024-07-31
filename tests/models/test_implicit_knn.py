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
from .utils import assert_second_fit_refits_model


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
                        Columns.Item: [12, 15, 11, 14],
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
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        base_model = TFIDFRecommender(K=5, num_threads=2)
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
        params = {"K": 5}
        if model_class == "BM25Recommender":
            params |= {"K1": 0.33}
        config = {
            "model": {
                "cls": model_class,
                "params": params,
            },
            "verbose": 1,
        }
        model = ImplicitItemKNNWrapperModel.from_config(config)
        assert model.verbose == 1
        inner_model = model._model
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
        expected_model_params = {
            "K": 5,
            "num_threads": 0,
        }
        if model_class is BM25Recommender:
            expected_model_params |= {
                "K1": 1.2,
                "B": 0.75,
            }
        expected = {
            "model": {
                "cls": model_class if not simple_types else model_class_str,
                "params": expected_model_params,
            },
            "verbose": 1,
        }
        assert config == expected

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(self, simple_types: bool) -> None:
        def get_reco(model: ImplicitItemKNNWrapperModel):
            return model.fit(DATASET).recommend(users=np.array([10, 20]), dataset=DATASET, k=2, filter_viewed=False)

        initial_config = {
            "model": {
                "cls": TFIDFRecommender,
                "params": {"K": 3},
            },
            "verbose": 1,
        }

        model_1 = ImplicitItemKNNWrapperModel.from_config(initial_config)
        reco_1 = get_reco(model_1)
        config_1 = model_1.get_config(simple_types=simple_types)

        model_2 = ImplicitItemKNNWrapperModel.from_config(config_1)
        reco_2 = get_reco(model_2)
        config_2 = model_2.get_config(simple_types=simple_types)

        assert config_1 == config_2
        pd.testing.assert_frame_equal(reco_1, reco_2)

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        model_from_config = ImplicitItemKNNWrapperModel.from_config({"model": {}})
        model_from_params = ImplicitItemKNNWrapperModel(model=ItemItemRecommender())
        assert model_from_config.get_config() == model_from_params.get_config()
