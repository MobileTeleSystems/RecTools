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
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.models import PopularModel
from tests.models.utils import assert_second_fit_refits_model


class TestPopularModel:
    @pytest.fixture
    def dataset(self) -> Dataset:
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
        user_id_map = IdMap.from_values([10, 20, 30, 40, 50, 60, 70, 80])
        item_id_map = IdMap.from_values([11, 12, 13, 14, 15, 16])
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map)
        return Dataset(user_id_map, item_id_map, interactions)

    @classmethod
    def assert_reco(
        cls,
        expected_items: tp.List[tp.List[int]],
        expected_scores: tp.Union[tp.List[tp.List[int]], tp.List[tp.List[float]]],
        targets: tp.List[int],
        target_col: str,
        actual: pd.DataFrame,
    ) -> None:
        assert actual.columns.tolist() == Columns.Recommendations

        expected_targets: tp.List[int] = sum([[u] * len(u_reco) for u, u_reco in zip(targets, expected_items)], [])
        assert actual[target_col].tolist() == expected_targets

        expected_ranks: tp.List[int] = sum([list(range(1, len(u_reco) + 1)) for u_reco in expected_items], [])
        assert actual[Columns.Rank].tolist() == expected_ranks

        assert actual[Columns.Item].tolist() == sum(expected_items, [])

        np.testing.assert_almost_equal(actual[Columns.Score].values, sum(expected_scores, []))

    @pytest.mark.parametrize(
        "model,expected_items,expected_scores",
        (
            (PopularModel(), [[14, 15], [12, 11, 14]], [[2, 1], [6, 5, 2]]),
            (PopularModel(popularity="n_interactions"), [[14, 15], [14, 12, 11]], [[7, 2], [7, 6, 5]]),
            (PopularModel(popularity="mean_weight"), [[15, 14], [13, 15, 14]], [[5, 8 / 7], [9, 5, 8 / 7]]),
            (PopularModel(popularity="sum_weight"), [[15, 14], [15, 13, 14]], [[10, 8], [10, 9, 8]]),
            (PopularModel(period=timedelta(days=7)), [[14], [11, 12, 14]], [[2], [4, 3, 2]]),
            (PopularModel(begin_from=datetime(2021, 11, 23)), [[14], [11, 12, 14]], [[2], [4, 3, 2]]),
            (PopularModel(add_cold=True), [[14, 15, 16], [12, 11, 14]], [[2, 1, 0], [6, 5, 2]]),
            (
                PopularModel(period=timedelta(days=7), add_cold=True),
                [[14, 15, 16], [11, 12, 14]],
                [[2, 0, 0], [4, 3, 2]],
            ),
            (PopularModel(inverse=True, period=timedelta(days=7)), [[14], [13, 14, 12]], [[2], [1, 2, 3]]),
            (
                PopularModel(add_cold=True, inverse=True, period=timedelta(days=7)),
                [[16, 15, 14], [16, 15, 13]],
                [[0, 0, 2], [0, 0, 1]],
            ),
        ),
    )
    def test_with_filtering_viewed(
        self,
        dataset: Dataset,
        model: PopularModel,
        expected_items: tp.List[tp.List[tp.Any]],
        expected_scores: tp.List[tp.List[float]],
    ) -> None:
        model.fit(dataset)
        actual = model.recommend(
            users=np.array([10, 80]),
            dataset=dataset,
            k=3,
            filter_viewed=True,
        )
        self.assert_reco(expected_items, expected_scores, [10, 80], Columns.User, actual)

    def test_without_filtering_viewed(self, dataset: Dataset) -> None:
        model = PopularModel().fit(dataset)
        actual = model.recommend(
            users=np.array([10, 80]),
            dataset=dataset,
            k=3,
            filter_viewed=False,
        )
        expected_items = [[12, 11, 14], [12, 11, 14]]
        expected_scores = [[6, 5, 2], [6, 5, 2]]
        self.assert_reco(expected_items, expected_scores, [10, 80], Columns.User, actual)

    def test_with_items_whitelist(self, dataset: Dataset) -> None:
        model = PopularModel().fit(dataset)
        actual = model.recommend(
            users=np.array([10, 80]), dataset=dataset, k=3, filter_viewed=True, items_to_recommend=[11, 15, 14]
        )
        expected_items = [[14, 15], [11, 14, 15]]
        expected_scores = [[2, 1], [5, 2, 1]]
        self.assert_reco(expected_items, expected_scores, [10, 80], Columns.User, actual)

    def test_raises_when_incorrect_popularity(self) -> None:
        with pytest.raises(ValueError):
            PopularModel(popularity="strange")

    def test_raises_when_both_period_and_begin_from_are_set(self) -> None:
        with pytest.raises(ValueError):
            PopularModel(period=timedelta(days=1), begin_from=datetime(2021, 11, 30))

    def test_raises_when_incorrect_popularity_in_fit(self, dataset: Dataset) -> None:
        model = PopularModel()
        model.popularity = "strange"  # type: ignore
        with pytest.raises(ValueError):
            model.fit(dataset)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [12, 11, 12, 11],
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
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        model = PopularModel().fit(dataset)
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
        model = PopularModel()
        assert_second_fit_refits_model(model, dataset)
