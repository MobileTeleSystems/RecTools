#  Copyright 2024 MTS (Mobile Telesystems)
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
from rectools.models import EASEModel

from .data import DATASET, INTERACTIONS
from .utils import (
    assert_default_config_and_default_model_params_are_the_same,
    assert_dumps_loads_do_not_change_model,
    assert_get_config_and_from_config_compatibility,
    assert_second_fit_refits_model,
)


class TestEASEModel:
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
                        Columns.Score: np.array([0.00788948, 0.0039526, 0.00789337, 0.00590922], dtype=np.float32),
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
                        Columns.Score: np.array([0.00988546, 0.00986199, 0.00791307, 0.00789747], dtype=np.float32),
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_basic(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = EASEModel(regularization=500).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
        )
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
                        Columns.Score: np.array([0.00788948, 0.00196058, 0.00590922, 0.00196845], dtype=np.float32),
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
                        Columns.Score: np.array([0.00986199, 0.00788948, 0.00789747, 0.00590922], dtype=np.float32),
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = EASEModel(regularization=500).fit(dataset)
        actual = model.recommend(
            users=np.array([10, 20]),
            dataset=dataset,
            k=2,
            filter_viewed=filter_viewed,
            items_to_recommend=np.array([11, 15, 17]),
        )
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
                        Columns.Item: [12, 15, 11, 14],
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
                        Columns.Item: [14, 13, 11, 14],
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        model = EASEModel(regularization=500).fit(dataset)
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
        model = EASEModel()
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
    @pytest.mark.parametrize("filter_viewed", (True, False))
    def test_u2i_with_warm_and_cold_users(
        self, filter_viewed: bool, user_features: tp.Optional[pd.DataFrame], error_match: str
    ) -> None:
        dataset = Dataset.construct(INTERACTIONS, user_features_df=user_features)
        model = EASEModel(regularization=500).fit(dataset)
        with pytest.raises(ValueError, match=error_match):
            model.recommend(
                users=[10, 20, 50],
                dataset=dataset,
                k=2,
                filter_viewed=filter_viewed,
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
        model = EASEModel(regularization=500).fit(dataset)
        with pytest.raises(ValueError, match=error_match):
            model.recommend_to_items(
                target_items=[11, 12, 16],
                dataset=dataset,
                k=2,
            )

    def test_dumps_loads(self, dataset: Dataset):
        model = EASEModel()
        model.fit(dataset)
        assert_dumps_loads_do_not_change_model(model, dataset)


class TestEASEModelConfiguration:
    def test_from_config(self) -> None:
        config = {
            "regularization": 500,
            "num_threads": 1,
            "verbose": 1,
        }
        model = EASEModel.from_config(config)
        assert model.num_threads == 1
        assert model.verbose == 1
        assert model.regularization == 500

    def test_get_config(self) -> None:
        model = EASEModel(
            regularization=500,
            num_threads=1,
            verbose=1,
        )
        config = model.get_config()
        expected = {
            "regularization": 500,
            "num_threads": 1,
            "verbose": 1,
        }
        assert config == expected

    @pytest.mark.parametrize("simple_types", (False, True))
    def test_get_config_and_from_config_compatibility(self, simple_types: bool) -> None:
        initial_config = {
            "regularization": 500,
            "num_threads": 1,
            "verbose": 1,
        }
        assert_get_config_and_from_config_compatibility(EASEModel, DATASET, initial_config, simple_types)

    def test_default_config_and_default_model_params_are_the_same(self) -> None:
        default_config: tp.Dict[str, int] = {}
        model = EASEModel()
        assert_default_config_and_default_model_params_are_the_same(model, default_config)
