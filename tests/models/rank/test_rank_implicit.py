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

import implicit.cpu
import numpy as np
import pytest
from scipy import sparse

from rectools.models.rank import Distance, ImplicitRanker

T = tp.TypeVar("T")

pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")


class TestImplicitRanker:  # pylint: disable=protected-access
    @pytest.fixture
    def subject_factors(self) -> np.ndarray:
        return np.array([[-4, 0, 3], [0, 1, 2]])

    @pytest.fixture
    def object_factors(self) -> np.ndarray:
        return np.array(
            [
                [-4, 0, 3],
                [0, 2, 4],
                [1, 10, 100],
            ]
        )

    @pytest.mark.parametrize(
        "dense",
        (
            (True),
            (False),
        ),
    )
    def test_neginf_score(
        self,
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)
        implicit_ranker = ImplicitRanker(
            Distance.DOT,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )
        dummy_factors: np.ndarray = np.array([[1, 2]], dtype=np.float32)
        neginf = implicit.cpu.topk.topk(  # pylint: disable=c-extension-no-member
            items=dummy_factors,
            query=dummy_factors,
            k=1,
            filter_items=np.array([0]),
        )[1][0][0]
        assert neginf <= implicit_ranker._get_neginf_score() <= -1e38

    @pytest.mark.parametrize(
        "dense",
        (
            (True),
            (False),
        ),
    )
    def test_mask_for_correct_scores(
        self, subject_factors: np.ndarray, object_factors: np.ndarray, dense: bool
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        implicit_ranker = ImplicitRanker(
            Distance.DOT,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )
        neginf = implicit_ranker._get_neginf_score()
        scores: np.ndarray = np.array([7, 6, 0, 0], dtype=np.float32)

        actual = implicit_ranker._get_mask_for_correct_scores(scores)
        assert actual == [True] * 4

        actual = implicit_ranker._get_mask_for_correct_scores(np.append(scores, [neginf] * 2))
        assert actual == [True] * 4 + [False] * 2

        actual = implicit_ranker._get_mask_for_correct_scores(np.append(scores, [neginf * 0.99] * 2))
        assert actual == [True] * 6

        actual = implicit_ranker._get_mask_for_correct_scores(np.insert(scores, 0, neginf))
        assert actual == [True] * 5

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores, dense",
        (
            (Distance.DOT, [2, 0, 1, 2, 1, 0], [296, 25, 12, 210, 10, 6], True),
            (
                Distance.COSINE,
                [0, 2, 1, 1, 2, 0],
                [1, 0.5890328, 0.5366563, 1, 0.9344414, 0.5366563],
                True,
            ),
            (
                Distance.EUCLIDEAN,
                [0, 1, 2, 1, 0, 2],
                [0, 4.58257569, 97.64220399, 2.23606798, 4.24264069, 98.41747812],
                True,
            ),
            (
                Distance.DOT,
                [2, 0, 1, 2, 1, 0],
                [296, 25, 12, 210, 10, 6],
                False,
            ),
        ),
    )
    @pytest.mark.parametrize("use_gpu", (False, True))
    def test_rank(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
        use_gpu: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ranker = ImplicitRanker(
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
            use_gpu=use_gpu,
        )
        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
        )

        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores, dense",
        (
            (Distance.DOT, [2, 0, 2, 1, 0], [296, 25, 210, 10, 6], True),
            (
                Distance.COSINE,
                [0, 2, 1, 2, 0],
                [1, 0.5890328, 1, 0.9344414, 0.5366563],
                True,
            ),
            (
                Distance.EUCLIDEAN,
                [0, 2, 1, 0, 2],
                [0, 97.64220399, 2.23606798, 4.24264069, 98.41747812],
                True,
            ),
            (Distance.DOT, [2, 0, 2, 1, 0], [296, 25, 210, 10, 6], False),
        ),
    )
    @pytest.mark.parametrize("use_gpu", (False, True))
    def test_rank_with_filtering_viewed_items(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
        use_gpu: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ui_csr = sparse.csr_matrix(
            [
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        ranker = ImplicitRanker(
            distance,
            subject_factors,
            object_factors,
            use_gpu=use_gpu,
        )
        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
            filter_pairs_csr=ui_csr,
        )
        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores, dense",
        (
            (Distance.DOT, [2, 0, 2, 0], [296, 25, 210, 6], True),
            (Distance.COSINE, [0, 2, 2, 0], [1, 0.5890328, 0.9344414, 0.5366563], True),
            (
                Distance.EUCLIDEAN,
                [0, 2, 0, 2],
                [0, 97.64220399, 4.24264069, 98.41747812],
                True,
            ),
            (Distance.DOT, [2, 0, 2, 0], [296, 25, 210, 6], False),
        ),
    )
    @pytest.mark.parametrize("use_gpu", (False, True))
    def test_rank_with_objects_whitelist(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
        use_gpu: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ranker = ImplicitRanker(
            distance,
            subject_factors,
            object_factors,
            use_gpu=use_gpu,
        )

        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
            sorted_object_whitelist=np.array([0, 2]),
        )
        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize(
        "distance, expected_recs, expected_scores, dense",
        (
            (Distance.DOT, [2, 2, 0], [296, 210, 6], True),
            (Distance.COSINE, [2, 2, 0], [0.5890328, 0.9344414, 0.5366563], True),
            (
                Distance.EUCLIDEAN,
                [2, 0, 2],
                [97.64220399, 4.24264069, 98.41747812],
                True,
            ),
            (Distance.DOT, [2, 2, 0], [296, 210, 6], False),
        ),
    )
    @pytest.mark.parametrize("use_gpu", (False, True))
    def test_rank_with_objects_whitelist_and_filtering_viewed_items(
        self,
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
        use_gpu: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ui_csr = sparse.csr_matrix(
            [
                [1, 1, 0],
                [0, 0, 0],
            ]
        )
        ranker = ImplicitRanker(
            distance,
            subject_factors,
            object_factors,
            use_gpu=use_gpu,
        )
        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
            sorted_object_whitelist=np.array([0, 2]),
            filter_pairs_csr=ui_csr,
        )
        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(actual_scores, expected_scores)

    @pytest.mark.parametrize("distance", (Distance.COSINE, Distance.EUCLIDEAN))
    def test_raises(
        self,
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        distance: Distance,
    ) -> None:
        subject_factors = sparse.csr_matrix(subject_factors)
        with pytest.raises(ValueError):
            ImplicitRanker(
                distance=distance,
                subjects_factors=subject_factors,
                objects_factors=object_factors,
            )
