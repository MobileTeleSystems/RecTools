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
from rectools.models.rank import Distance
from rectools.models.vector import Factors, VectorModel

T = tp.TypeVar("T")

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")


class TestVectorModel:  # pylint: disable=protected-access, attribute-defined-outside-init
    def setup_method(self) -> None:
        stub_interactions = pd.DataFrame([], columns=Columns.Interactions)
        self.stub_dataset = Dataset.construct(stub_interactions)
        user_embeddings = np.array([[-4, 0, 3], [0, 1, 2]])
        item_embeddings = np.array(
            [
                [-4, 0, 3],
                [0, 1, 2],
                [1, 10, 100],
            ]
        )
        user_biases = np.array([0, 1])
        item_biases = np.array([0, 1, 3])
        self.user_factors = Factors(user_embeddings)
        self.item_factors = Factors(item_embeddings)
        self.user_biased_factors = Factors(user_embeddings, user_biases)
        self.item_biased_factors = Factors(item_embeddings, item_biases)

    @staticmethod
    def make_model(
        user_factors: tp.Optional[Factors] = None,
        item_factors: tp.Optional[Factors] = None,
        u2i_distance: Distance = Distance.DOT,
        i2i_distance: Distance = Distance.COSINE,
    ) -> VectorModel:
        class SomeVectorModel(VectorModel):

            u2i_dist = u2i_distance
            i2i_dist = i2i_distance

            def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
                pass

            def _get_users_factors(self, dataset: Dataset) -> Factors:
                return user_factors if user_factors is not None else Factors(np.array([]))

            def _get_items_factors(self, dataset: Dataset) -> Factors:
                return item_factors if item_factors is not None else Factors(np.array([]))

        model = SomeVectorModel()
        return model

    @pytest.mark.parametrize(
        "distance,expected_reco,expected_scores",
        (
            (Distance.DOT, [[2, 0, 1], [2, 0, 1]], [[296.0, 25.0, 6.0], [210.0, 6.0, 5.0]]),
            (Distance.COSINE, [[0, 2, 1], [1, 2, 0]], [[1.0, 0.58903, 0.53666], [1.0, 0.93444, 0.53666]]),
            (Distance.EUCLIDEAN, [[0, 1, 2], [1, 0, 2]], [[0.0, 4.24264, 97.6422], [0.0, 4.24264, 98.41748]]),
        ),
    )
    @pytest.mark.parametrize("method", ("u2i", "i2i"))
    def test_without_biases(
        self,
        distance: Distance,
        expected_reco: tp.List[tp.List[int]],
        expected_scores: tp.List[tp.List[float]],
        method: str,
    ) -> None:
        model = self.make_model(self.user_factors, self.item_factors, u2i_distance=distance, i2i_distance=distance)
        if method == "u2i":
            _, reco, scores = model._recommend_u2i(np.array([0, 1]), self.stub_dataset, 5, False, None)
        else:  # i2i
            _, reco, scores = model._recommend_i2i(np.array([0, 1]), self.stub_dataset, 5, None)
        assert list(reco) == sum(expected_reco, [])
        np.testing.assert_almost_equal(scores, np.array(expected_scores).ravel(), decimal=5)

    @pytest.mark.parametrize(
        "distance,expected_reco,expected_scores",
        (
            (Distance.DOT, [[2, 0, 1], [2, 1, 0]], [[299.0, 25.0, 7.0], [214.0, 7.0, 7.0]]),
            (Distance.COSINE, [[0, 2, 1], [1, 2, 0]], [[1.0, 0.58877, 0.4899], [1.0, 0.86483, 0.4899]]),
            (Distance.EUCLIDEAN, [[0, 1, 2], [1, 0, 2]], [[0.0, 4.3589, 97.68828], [0.0, 4.3589, 98.4378]]),
        ),
    )
    @pytest.mark.parametrize("method", ("u2i", "i2i"))
    def test_with_biases(
        self,
        distance: Distance,
        expected_reco: tp.List[tp.List[int]],
        expected_scores: tp.List[tp.List[float]],
        method: str,
    ) -> None:
        model = self.make_model(
            self.user_biased_factors, self.item_biased_factors, u2i_distance=distance, i2i_distance=distance
        )
        if method == "u2i":
            _, reco, scores = model._recommend_u2i(np.array([0, 1]), self.stub_dataset, 5, False, None)
        else:  # i2i
            _, reco, scores = model._recommend_i2i(np.array([0, 1]), self.stub_dataset, 5, None)
        assert list(reco) == sum(expected_reco, [])
        np.testing.assert_almost_equal(scores, np.array(expected_scores).ravel(), decimal=5)

    @pytest.mark.parametrize("method", ("u2i", "i2i"))
    def test_with_incorrect_distance(self, method: str) -> None:
        with pytest.raises(ValueError):
            if method == "u2i":
                m = self.make_model(self.user_biased_factors, self.item_biased_factors, u2i_distance=7)  # type: ignore
                m._get_u2i_vectors(self.stub_dataset)
            else:
                m = self.make_model(self.user_biased_factors, self.item_biased_factors, i2i_distance=7)  # type: ignore
                m._get_i2i_vectors(self.stub_dataset)
