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
from rectools.exceptions import NotFittedError
from rectools.models.pure_svd import PureSVDModel
from rectools.models.utils import recommend_from_scores

from .data import DATASET, INTERACTIONS
from .utils import assert_second_fit_refits_model


class TestPureSVDModel:
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
                        Columns.Item: [13, 15, 14, 15],
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
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_basic(
        self,
        dataset: Dataset,
        filter_viewed: bool,
        expected: pd.DataFrame,
    ) -> None:
        model = PureSVDModel(factors=2).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [15, 17, 15, 17],
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
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = PureSVDModel(factors=2).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([11, 15, 17]),
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_get_vectors(self, dataset: Dataset) -> None:
        model = PureSVDModel(factors=2).fit(dataset)
        user_embeddings, item_embeddings = model.get_vectors()
        predictions = user_embeddings @ item_embeddings.T
        vectors_predictions = [recommend_from_scores(predictions[i], k=5) for i in range(4)]
        vectors_reco = np.array([vp[0] for vp in vectors_predictions]).ravel()
        vectors_scores = np.array([vp[1] for vp in vectors_predictions]).ravel()
        _, reco_item_ids, reco_scores = model._recommend_u2i(  # pylint: disable=protected-access
            user_ids=dataset.user_id_map.convert_to_internal(np.array([10, 20, 30, 40])),
            dataset=dataset,
            k=5,
            filter_viewed=False,
            sorted_item_ids_to_recommend=None,
        )
        np.testing.assert_equal(vectors_reco, reco_item_ids)
        np.testing.assert_almost_equal(vectors_scores, reco_scores, decimal=5)

    def test_raises_when_get_vectors_from_not_fitted(self, dataset: Dataset) -> None:
        model = PureSVDModel(factors=2)
        with pytest.raises(NotFittedError):
            model.get_vectors()

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 14, 12, 14],
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
                        Columns.Item: [14, 12, 14, 13],
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
                        Columns.Item: [11, 14, 14, 13],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        model = PureSVDModel(factors=2).fit(dataset)
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
        model = PureSVDModel(factors=3)
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
        model = PureSVDModel(factors=2).fit(dataset)
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
        model = PureSVDModel(factors=2).fit(dataset)
        with pytest.raises(ValueError, match=error_match):
            model.recommend_to_items(
                target_items=[11, 12, 16],
                dataset=dataset,
                k=2,
            )
