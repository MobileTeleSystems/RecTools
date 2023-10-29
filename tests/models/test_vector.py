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

import implicit.cpu
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.vector import Distance, Factors, ImplicitRanker, VectorModel

T = tp.TypeVar("T")

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")


class TestImplicitRanker:  # pylint: disable=protected-access
    @pytest.fixture
    def subject_factors(self) -> np.ndarray:
        return np.array([[-4, 0, 3], [0, 0, 0]])

    @pytest.fixture
    def object_factors(self) -> np.ndarray:
        return np.array(
            [
                [-4, 0, 3],
                [0, 0, 0],
                [1, 1, 1],
            ]
        )

    def test_neginf_score(self, subject_factors: np.ndarray, object_factors: np.ndarray) -> None:
        implicit_ranker = ImplicitRanker(Distance.DOT, subjects_factors=subject_factors, objects_factors=object_factors)
        dummy_factors = np.array([[1, 2]], dtype=np.float32)
        neginf = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
            items=dummy_factors,
            query=dummy_factors,
            k=1,
            filter_items=np.array([0]),
        )[1][0][0]
        assert neginf == implicit_ranker._get_neginf_score()

    def test_mask_for_correct_scores(self, subject_factors: np.ndarray, object_factors: np.ndarray) -> None:
        implicit_ranker = ImplicitRanker(Distance.DOT, subjects_factors=subject_factors, objects_factors=object_factors)
        neginf = implicit_ranker._get_neginf_score()
        scores = np.array([7, 6, 0, 0], dtype=np.float32)

        actual = implicit_ranker._get_mask_for_correct_scores(scores)
        assert actual == [True] * 4

        actual = implicit_ranker._get_mask_for_correct_scores(np.append(scores, [neginf] * 2))
        assert actual == [True] * 4 + [False] * 2

        actual = implicit_ranker._get_mask_for_correct_scores(np.append(scores, [neginf * 0.99] * 2))
        assert actual == [True] * 6

        actual = implicit_ranker._get_mask_for_correct_scores(np.insert(scores, 0, neginf))
        assert actual == [True] * 5

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores",
        (
            (Distance.DOT, [0, 1, 2, 2, 1, 0], [25, 0, -1, 0, 0, 0]),
            (Distance.COSINE, [0, 1, 2, 2, 1, 0], [1, 0, -1 / (5 * 3**0.5), 0, 0, 0]),
            (Distance.EUCLIDEAN, [0, 1, 2, 1, 2, 0], [0, 5, 30**0.5, 0, 3**0.5, 5]),
        ),
    )
    def test_rank(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
    ) -> None:
        ranker = ImplicitRanker(distance, subject_factors, object_factors)
        _, actoal_recs, actual_scores = ranker.rank(subject_ids=[0, 1], k=3)
        np.testing.assert_equal(actoal_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores",
        (
            (Distance.DOT, [0, 2, 2, 1, 0], [25, -1, 0, 0, 0]),
            (Distance.COSINE, [0, 2, 2, 1, 0], [1, -1 / (5 * 3**0.5), 0, 0, 0]),
            (Distance.EUCLIDEAN, [0, 2, 1, 2, 0], [0, 30**0.5, 0, 3**0.5, 5]),
        ),
    )
    def test_rank_with_filtering_viewed_items(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
    ) -> None:
        ui_csr = sparse.csr_matrix(
            [
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        ranker = ImplicitRanker(distance, subject_factors, object_factors)
        _, actoal_recs, actual_scores = ranker.rank(subject_ids=[0, 1], k=3, filter_so_csr=ui_csr)
        np.testing.assert_equal(actoal_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores",
        (
            (Distance.DOT, [0, 2, 2, 0], [25, -1, 0, 0]),
            (Distance.COSINE, [0, 2, 2, 0], [1, -1 / (5 * 3**0.5), 0, 0]),
            (Distance.EUCLIDEAN, [0, 2, 2, 0], [0, 30**0.5, 3**0.5, 5]),
        ),
    )
    def test_rank_with_objects_whitelist(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
    ) -> None:
        ranker = ImplicitRanker(distance, subject_factors, object_factors)
        _, actoal_recs, actual_scores = ranker.rank(subject_ids=[0, 1], k=3, sorted_object_whitelist=np.array([0, 2]))
        np.testing.assert_equal(actoal_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores",
        (
            (Distance.DOT, [2, 2, 0], [-1, 0, 0]),
            (Distance.COSINE, [2, 2, 0], [-1 / (5 * 3**0.5), 0, 0]),
            (Distance.EUCLIDEAN, [2, 2, 0], [30**0.5, 3**0.5, 5]),
        ),
    )
    def test_rank_with_objects_whitelist_and_filtering_viewed_items(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
    ) -> None:
        ui_csr = sparse.csr_matrix(
            [
                [1, 1, 0],
                [0, 0, 0],
            ]
        )
        ranker = ImplicitRanker(distance, subject_factors, object_factors)
        _, actoal_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1], k=3, sorted_object_whitelist=np.array([0, 2]), filter_so_csr=ui_csr
        )
        np.testing.assert_equal(actoal_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)


class TestVectorModel:  # pylint: disable=protected-access, attribute-defined-outside-init
    def setup(self) -> None:
        stub_interactions = pd.DataFrame([], columns=Columns.Interactions)
        self.stub_dataset = Dataset.construct(stub_interactions)
        user_embeddings = np.array([[-4, 0, 3], [0, 0, 0]])
        item_embeddings = np.array(
            [
                [-4, 0, 3],
                [0, 0, 0],
                [1, 1, 1],
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
            (Distance.DOT, [[0, 1, 2], [2, 1, 0]], [[25, 0, -1], [0, 0, 0]]),
            (Distance.COSINE, [[0, 1, 2], [2, 1, 0]], [[1, 0, -1 / (5 * 3**0.5)], [0, 0, 0]]),
            (Distance.EUCLIDEAN, [[0, 1, 2], [1, 2, 0]], [[0, 5, 30**0.5], [0, 3**0.5, 5]]),
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
            (Distance.DOT, [[0, 2, 1], [2, 1, 0]], [[25, 2, 1], [4, 2, 1]]),
            (Distance.COSINE, [[0, 1, 2], [1, 2, 0]], [[1, 0, -1 / (5 * 12**0.5)], [1, 3 / (1 * 12**0.5), 0]]),
            (Distance.EUCLIDEAN, [[0, 1, 2], [1, 2, 0]], [[0, 26**0.5, 39**0.5], [0, 7**0.5, 26**0.5]]),
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
