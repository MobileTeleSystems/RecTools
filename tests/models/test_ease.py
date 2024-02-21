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
from rectools.models import EASEModel

from .data import DATASET
from .utils import assert_second_fit_refits_model


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
                        Columns.Item: [17, 13, 17, 15],
                        Columns.Score: np.array([-503.96243, -1006.93286, -503.96243, -1510.8953], dtype=np.float32),
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [17, 13, 17, 13],
                        Columns.Score: np.array([-503.96243, -1006.93286, -503.96243, -1006.93286], dtype=np.float32),
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
        tol_kwargs: tp.Dict[str, float] = {"check_less_precise": 3} if pd.__version__ < "1" else {"atol": 0.001}
        pd.testing.assert_frame_equal(actual, expected, **tol_kwargs)  # pylint: disable = unexpected-keyword-arg

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [17, 15, 17, 15],
                        Columns.Score: np.array([-503.96243, -2012.8776, -503.96243, -1510.8953], dtype=np.float32),
                        Columns.Rank: [1, 2, 1, 2],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20],
                        Columns.Item: [17, 15, 17, 15],
                        Columns.Score: np.array([-503.96243, -2012.8776, -503.96243, -1510.8953], dtype=np.float32),
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
        tol_kwargs: tp.Dict[str, float] = {"check_less_precise": 3} if pd.__version__ < "1" else {"atol": 0.001}
        pd.testing.assert_frame_equal(actual, expected, **tol_kwargs)  # pylint: disable = unexpected-keyword-arg

    @pytest.mark.parametrize("filter_viewed", (True, False))
    def test_raises_when_new_user(self, dataset: Dataset, filter_viewed: bool) -> None:
        model = EASEModel(regularization=500).fit(dataset)
        with pytest.raises(KeyError):
            model.recommend(
                users=np.array([10, 50]),
                dataset=dataset,
                k=2,
                filter_viewed=filter_viewed,
            )

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [11, 11, 12, 12],
                        Columns.Item: [11, 13, 17, 12],
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
                        Columns.Item: [13, 17, 17, 13],
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
                        Columns.Item: [11, 13, 13, 14],
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
