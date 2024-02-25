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
from lightfm import LightFM

from rectools import Columns
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models import LightFMWrapperModel
from rectools.models.utils import recommend_from_scores
from tests.models.utils import assert_second_fit_refits_model


# pylint: disable=attribute-defined-outside-init
class DeterministicLightFM(LightFM):
    def _initialize(self, no_components: int, no_item_features: int, no_user_features: int) -> None:
        super()._initialize(no_components, no_item_features, no_user_features)

        self.item_embeddings = (
            np.linspace(-0.5, 0.5, no_item_features * no_components, dtype=np.float32).reshape(
                no_item_features, no_components
            )
            / no_components
        )
        self.user_embeddings = (
            np.linspace(-0.5, 0.5, no_user_features * no_components, dtype=np.float32).reshape(
                no_user_features, no_components
            )
            / no_components
        )


class TestLightFMWrapperModel:
    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        data = [
            [10, 11],
            [10, 12],
            [10, 13],
            [10, 14],
            [20, 11],
            [20, 12],
            [20, 15],
            [30, 11],
            [30, 12],
            [30, 13],
            [30, 15],
        ]
        data += [[40 + i, iid] for i in range(2) for iid in (11, 12, 13)]
        data += [[50 + i, iid] for i in range(4) for iid in (11, 12)]
        data += [[60 + i, iid] for i in range(50) for iid in (11,)]
        interactions = pd.DataFrame(data, columns=Columns.UserItem)
        interactions[Columns.Weight] = 1
        interactions[Columns.Datetime] = "2021-09-09"
        return interactions

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def dataset_with_features(self, interactions_df: pd.DataFrame) -> Dataset:
        user_features_df = pd.DataFrame({"id": [10], "feature": ["f1"], "value": [2]})
        item_features_df = pd.DataFrame(
            {
                "id": [11, 11, 12, 12, 14, 14],
                "feature": ["f1", "f2", "f1", "f2", "f1", "f2"],
                "value": [100, "a", 100, "a", 100, "a"],
            }
        )

        dataset = Dataset.construct(
            interactions_df=interactions_df,
            user_features_df=user_features_df,
            item_features_df=item_features_df,
            cat_item_features=["f1", "f2"],
        )
        return dataset

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 20, 20],
                        Columns.Item: [15, 13, 14],
                        Columns.Rank: [1, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [11, 12, 11, 12],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_without_features(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        base_model = DeterministicLightFM(no_components=2, loss="logistic")
        model = LightFMWrapperModel(model=base_model, epochs=50).fit(dataset)
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
                        Columns.User: [20],
                        Columns.Item: [14],
                        Columns.Rank: [1],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20],
                        Columns.Item: [11, 15],
                        Columns.Rank: [1, 2],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        base_model = DeterministicLightFM(no_components=2, loss="logistic")
        model = LightFMWrapperModel(model=base_model, epochs=50).fit(dataset)
        actual = model.recommend(
            users=np.array([20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([11, 14, 15]),
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_with_features(self, dataset_with_features: Dataset) -> None:
        base_model = DeterministicLightFM(no_components=2, loss="logistic")
        model = LightFMWrapperModel(model=base_model, epochs=50).fit(dataset_with_features)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset_with_features,
            k=2,
            filter_viewed=True,
        )

        expected = pd.DataFrame(
            {
                Columns.User: [10, 20, 20],
                Columns.Item: [15, 14, 13],
                Columns.Rank: [1, 1, 2],
            }
        )
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_with_weights(self, interactions_df: pd.DataFrame) -> None:
        interactions_df.loc[interactions_df[Columns.Item] == 14, Columns.Weight] = 100
        dataset = Dataset.construct(interactions_df)
        base_model = DeterministicLightFM(no_components=2, loss="logistic")
        model = LightFMWrapperModel(model=base_model, epochs=50).fit(dataset)
        actual = model.recommend(
            users=np.array([20]),
            dataset=dataset,
            k=2,
            filter_viewed=True,
        )
        assert actual[Columns.Item].tolist() == [14, 13]
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_get_vectors(self, dataset_with_features: Dataset) -> None:
        base_model = LightFM(no_components=2, loss="logistic")
        model = LightFMWrapperModel(model=base_model).fit(dataset_with_features)
        user_embeddings, item_embeddings = model.get_vectors(dataset_with_features)
        predictions = user_embeddings @ item_embeddings.T
        vectors_predictions = [recommend_from_scores(predictions[i], k=5) for i in range(4)]
        vectors_reco = np.array([vp[0] for vp in vectors_predictions]).ravel()
        vectors_scores = np.array([vp[1] for vp in vectors_predictions]).ravel()
        _, reco_item_ids, reco_scores = model._recommend_u2i(  # pylint: disable=protected-access
            user_ids=dataset_with_features.user_id_map.convert_to_internal(np.array([10, 20, 30, 40])),
            dataset=dataset_with_features,
            k=5,
            filter_viewed=False,
            sorted_item_ids_to_recommend=None,
        )
        np.testing.assert_equal(vectors_reco, reco_item_ids)
        np.testing.assert_almost_equal(vectors_scores, reco_scores, decimal=5)

    def test_raises_when_get_vectors_from_not_fitted(self, dataset: Dataset) -> None:
        model = LightFMWrapperModel(model=LightFM())
        with pytest.raises(NotFittedError):
            model.get_vectors(dataset)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 15, 12, 13],
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
                        Columns.Item: [15, 14, 13, 14],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                np.array([11, 15, 14]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 15, 14, 15],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        base_model = DeterministicLightFM(no_components=2, loss="logistic")
        model = LightFMWrapperModel(model=base_model, epochs=100).fit(dataset)
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
        base_model = LightFM(no_components=2, loss="logistic", random_state=1)
        model = LightFMWrapperModel(model=base_model, epochs=5, num_threads=1)
        assert_second_fit_refits_model(model, dataset)
