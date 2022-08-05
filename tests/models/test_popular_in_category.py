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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import PopularInCategoryModel
from tests.models.utils import assert_second_fit_refits_model


@pytest.mark.filterwarnings("ignore")
class TestPopularInCategoryModel:
    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [70, 11, 1, "2021-11-30"],
                [70, 12, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-30"],
                [10, 12, 1, "2021-11-29"],
                [10, 13, 9, "2021-11-28"],
                [20, 11, 1, "2021-11-27"],
                [20, 14, 2, "2021-11-26"],
                [20, 14, 1, "2021-11-25"],
                [20, 14, 1, "2021-11-25"],
                [20, 14, 1, "2021-11-25"],
                [20, 14, 1, "2021-11-25"],
                [20, 14, 1, "2021-11-25"],
                [30, 11, 1, "2021-11-24"],
                [30, 12, 1, "2021-11-23"],
                [30, 14, 1, "2021-11-23"],
                [30, 15, 5, "2021-11-21"],
                [30, 15, 5, "2021-11-21"],
                [40, 11, 1, "2021-11-20"],
                [40, 12, 1, "2021-11-19"],
                [50, 12, 1, "2021-11-19"],
                [60, 12, 1, "2021-11-19"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.fixture
    def item_features_df(self) -> pd.DataFrame:
        item_features_df = pd.DataFrame(
            {
                "id": [11, 11, 12, 12, 13, 13, 14, 14, 14],
                "feature": ["f1", "f2", "f1", "f2", "f1", "f2", "f1", "f2", "f3"],
                "value": [100, "a", 100, "b", 100, "b", 200, "c", 1],
            }
        )
        return item_features_df

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame, item_features_df: pd.DataFrame) -> Dataset:
        dataset = Dataset.construct(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            cat_item_features=["f2", "f1"],
        )
        return dataset

    @classmethod
    def assert_reco(
        cls,
        expected_items: tp.List[tp.List[int]],
        expected_scores: tp.Union[tp.List[tp.List[int]], tp.List[tp.List[float]]],
        users: tp.List[int],
        actual: pd.DataFrame,
    ) -> None:
        assert actual.columns.tolist() == Columns.Recommendations

        expected_users: tp.List[int] = sum([[u] * len(u_reco) for u, u_reco in zip(users, expected_items)], [])
        assert actual[Columns.User].tolist() == expected_users

        expected_ranks: tp.List[int] = sum([list(range(1, len(u_reco) + 1)) for u_reco in expected_items], [])
        assert actual[Columns.Rank].tolist() == expected_ranks

        assert actual[Columns.Item].tolist() == sum(expected_items, [])
        np.testing.assert_almost_equal(actual[Columns.Score].values, sum(expected_scores, []))

    def test_raises_when_incorrect_popularity(self) -> None:
        with pytest.raises(ValueError):
            PopularInCategoryModel(popularity="strange", category_feature="f2")

    def test_raises_when_incorrect_n_categories(self) -> None:
        with pytest.raises(ValueError):
            PopularInCategoryModel(category_feature="f2", n_categories=-1)

    def test_raises_when_incorrect_mixing_strategy(self) -> None:
        with pytest.raises(ValueError):
            PopularInCategoryModel(mixing_strategy="strange", category_feature="f2")

    def test_raises_when_incorrect_ratio_strategy(self) -> None:
        with pytest.raises(ValueError):
            PopularInCategoryModel(ratio_strategy="strange", category_feature="f2")

    def test_raises_when_dense_features(self, interactions_df: pd.DataFrame) -> None:
        item_idx = interactions_df[Columns.Item].unique()
        features_dense = pd.DataFrame({"id": item_idx, "f2": [1] * len(item_idx)})
        dataset_w_dense_features = Dataset.construct(
            interactions_df=interactions_df,
            item_features_df=features_dense,
            cat_item_features=["f2"],
            make_dense_item_features=True,
        )
        model = PopularInCategoryModel(category_feature="f2")
        with pytest.raises(TypeError):
            model.fit(dataset_w_dense_features)

    def test_raises_when_no_item_features(self, interactions_df: pd.DataFrame) -> None:
        dataset_w_no_features = Dataset.construct(interactions_df)
        model = PopularInCategoryModel(category_feature="f2")
        with pytest.raises(ValueError):
            model.fit(dataset_w_no_features)

    def test_raises_when_category_feature_not_in_item_features(self, dataset: Dataset) -> None:
        model = PopularInCategoryModel(category_feature="strange")
        with pytest.raises(ValueError):
            model.fit(dataset)

    def test_raises_when_category_feature_not_in_category_features(self, dataset: Dataset) -> None:
        model = PopularInCategoryModel(category_feature="f3")
        with pytest.raises(ValueError):
            model.fit(dataset)

    def test_raises_when_both_period_and_begin_from_are_set(self) -> None:
        with pytest.raises(ValueError):
            PopularInCategoryModel(period=timedelta(days=1), begin_from=datetime(2021, 11, 30), category_feature="f2")

    @pytest.mark.parametrize(
        "model,expected_category_scores",
        (
            (PopularInCategoryModel(category_feature="f2"), pd.Series({2: 6, 1: 5, 3: 2})),
            (PopularInCategoryModel(category_feature="f2", popularity="n_interactions"), pd.Series({2: 7, 3: 7, 1: 5})),
            (
                PopularInCategoryModel(category_feature="f2", popularity="sum_weight", n_categories=2),
                pd.Series({2: 15, 3: 8}),
            ),
            (
                PopularInCategoryModel(category_feature="f2", popularity="mean_weight", n_categories=4),
                pd.Series({2: 2.142857, 3: 1.142857, 1: 1.0}),
            ),
        ),
    )
    def test_popularity_scores_after_fitting(
        self, dataset: Dataset, model: PopularInCategoryModel, expected_category_scores: pd.Series
    ) -> None:
        model.fit(dataset)
        assert np.allclose(model.category_scores.sort_index(), expected_category_scores.sort_index())

    @pytest.mark.parametrize(
        "model,k,expected_num_recs",
        (
            (PopularInCategoryModel(category_feature="f2"), 13, pd.Series({2: 6, 1: 5, 3: 2})),
            (PopularInCategoryModel(category_feature="f2"), 10, pd.Series({2: 5, 1: 4, 3: 1})),
            (PopularInCategoryModel(category_feature="f2"), 3, pd.Series({2: 1, 1: 1, 3: 1})),
            (PopularInCategoryModel(category_feature="f2"), 2, pd.Series({2: 1, 1: 1, 3: 0})),
            (PopularInCategoryModel(category_feature="f2", ratio_strategy="equal"), 13, pd.Series({2: 5, 1: 4, 3: 4})),
            (PopularInCategoryModel(category_feature="f2", ratio_strategy="equal"), 3, pd.Series({2: 1, 1: 1, 3: 1})),
        ),
    )
    def test_num_recs_after_fitting(
        self, dataset: Dataset, model: PopularInCategoryModel, k: int, expected_num_recs: pd.Series
    ) -> None:
        model.fit(dataset)
        actual = model._get_num_recs_for_each_category(k)  # pylint: disable=protected-access
        assert np.allclose(actual, expected_num_recs)

    @pytest.mark.parametrize(
        "model,expected_items,expected_scores",
        (
            (
                PopularInCategoryModel(
                    category_feature="f2",
                    popularity="mean_weight",
                    mixing_strategy="group",
                    ratio_strategy="proportional",
                ),
                [[13, 12, 14, 11], [13, 12, 14, 11]],
                [[9, 1, 1.142857, 1], [9, 1, 1.142857, 1]],
            ),
            (
                PopularInCategoryModel(
                    category_feature="f2",
                    popularity="mean_weight",
                    mixing_strategy="rotate",
                    ratio_strategy="proportional",
                ),
                [[13, 14, 11, 12], [13, 14, 11, 12]],
                [[9, 1.142857, 1, 1], [9, 1.142857, 1, 1]],
            ),
        ),
    )
    def test_without_filtering_viewed(
        self,
        dataset: Dataset,
        model: PopularInCategoryModel,
        expected_items: tp.List[tp.List[tp.Any]],
        expected_scores: tp.List[tp.List[tp.Any]],
    ) -> None:
        model.fit(dataset)
        actual = model.recommend(users=np.array([10, 30]), dataset=dataset, k=4, filter_viewed=False)
        self.assert_reco(expected_items, expected_scores, [10, 30], actual)

    @pytest.mark.parametrize(
        "model,k,expected_items,expected_scores",
        (
            (
                PopularInCategoryModel(
                    category_feature="f2",
                    popularity="n_interactions",
                    mixing_strategy="group",
                    ratio_strategy="equal",
                    begin_from=datetime(year=2021, month=11, day=28),
                ),
                2,
                [[], [12, 13], [13], [13, 11]],
                [[], [2, 1], [1], [1, 2]],
            ),
            (
                PopularInCategoryModel(
                    category_feature="f2",
                    popularity="n_interactions",
                    mixing_strategy="group",
                    ratio_strategy="equal",
                    period=timedelta(days=2),
                ),
                2,
                [[], [12, 13], [13], [13, 11]],
                [[], [2, 1], [1], [1, 2]],
            ),
            (
                PopularInCategoryModel(
                    category_feature="f2",
                    popularity="n_users",
                    mixing_strategy="group",
                    ratio_strategy="equal",
                ),
                4,
                [[14], [12, 13], [13, 14], [13, 11, 14]],
                [[2], [6, 1], [1, 2], [1, 5, 2]],
            ),
            (
                PopularInCategoryModel(
                    category_feature="f2",
                    popularity="n_users",
                    mixing_strategy="group",
                    ratio_strategy="equal",
                ),
                1,
                [[14], [12], [13], [13]],
                [[2], [6], [1], [1]],
            ),
        ),
    )
    def test_with_filtering_viewed(
        self,
        dataset: Dataset,
        model: PopularInCategoryModel,
        k: int,
        expected_items: tp.List[tp.List[tp.Any]],
        expected_scores: tp.List[tp.List[tp.Any]],
    ) -> None:
        model.fit(dataset)
        actual = model.recommend(users=np.array([10, 20, 40, 50]), dataset=dataset, k=k, filter_viewed=True)
        self.assert_reco(expected_items, expected_scores, [10, 20, 40, 50], actual)

    def test_with_items_white_list(self, dataset: Dataset) -> None:
        model = PopularInCategoryModel(
            category_feature="f2",
            popularity="n_users",
            mixing_strategy="group",
            ratio_strategy="equal",
        )
        model.fit(dataset)
        actual = model.recommend(
            users=[10, 20, 40, 50], dataset=dataset, k=2, items_to_recommend=[12, 13], filter_viewed=True
        )
        expected_items = [[12, 13], [13], [13]]
        expected_scores = [[6, 1], [1], [1]]
        self.assert_reco(expected_items, expected_scores, [20, 40, 50], actual)

    @pytest.mark.parametrize(
        "item_features_df,expected_n_categories",
        (
            (
                pd.DataFrame(
                    {
                        "id": [11, 12, 13, 14],
                        "feature": ["f2", "f2", "f2", "f2"],
                        "value": ["a", "a", "b", 1],
                    }
                ),
                3,
            ),
            (
                pd.DataFrame(
                    {
                        "id": [11, 12, 13, 14],
                        "feature": ["f2", "f2", "f2", "f2"],
                        "value": [0, 1, 2, None],
                    }
                ),
                4,
            ),
            (
                pd.DataFrame(
                    {
                        "id": [11, 12, 13, 14],
                        "feature": ["f2", "f2", "f2", "f2"],
                        "value": [True, True, False, False],
                    }
                ),
                2,
            ),
        ),
    )
    def test_n_effective_categories(
        self, interactions_df: pd.DataFrame, item_features_df: pd.DataFrame, expected_n_categories: int
    ) -> None:
        dataset = Dataset.construct(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            cat_item_features=["f2", "f1"],
        )
        model = PopularInCategoryModel(category_feature="f2", n_categories=100)
        model.fit(dataset)
        assert model.n_effective_categories == expected_n_categories

    def test_fallback_reco_correctness(self, interactions_df: pd.DataFrame) -> None:
        item_features_df = pd.DataFrame(
            {
                "id": [11, 12, 12, 13, 14, 15],
                "feature": ["f2", "f2", "f2", "f2", "f2", "f2"],
                "value": ["a", "a", "b", "b", "c", "a"],
            }
        )
        dataset = Dataset.construct(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            cat_item_features=["f2"],
        )
        model = PopularInCategoryModel(category_feature="f2", ratio_strategy="equal")
        model.fit(dataset)
        actual = model.recommend(users=[10], dataset=dataset, k=6, filter_viewed=False)
        expected_items = [[12, 13, 14, 11, 15]]
        expected_scores = [[6, 1, 2, 5, 1]]
        self.assert_reco(expected_items, expected_scores, [10], actual)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 13, 13],
                        Columns.Item: [13, 14, 13, 14],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 13, 13],
                        Columns.Item: [13, 14, 14, 11],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                np.array([12, 13, 11]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 13, 13],
                        Columns.Item: [13, 11, 13, 11],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        model = PopularInCategoryModel(
            category_feature="f2",
            popularity="mean_weight",
            mixing_strategy="group",
            ratio_strategy="proportional",
        ).fit(dataset)
        actual = model.recommend_to_items(
            target_items=np.array([11, 13]),
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
        model = PopularInCategoryModel(
            category_feature="f2",
            popularity="mean_weight",
            mixing_strategy="group",
            ratio_strategy="proportional",
        )
        assert_second_fit_refits_model(model, dataset)
