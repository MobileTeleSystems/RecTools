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
from itertools import product

import numpy as np
import pytest
from scipy import sparse

from rectools.models.rank import Distance, ImplicitRanker
from rectools.models.rank_torch import Ranker, TorchRanker

T = tp.TypeVar("T")
EPS_DIGITS = 5
pytestmark = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in true_divide"
)


def gen_rankers() -> tp.List[tp.Tuple[tp.Any, tp.Dict[str, tp.Any]]]:
    keys = ["device", "batch_size"]
    vals = list(
        product(
            ["cpu", "cuda:0"],
            [128, 1],
        )
    )
    torch_ranker_args = [(TorchRanker, dict(zip(keys, v))) for v in vals]

    keys = ["use_gpu"]
    vals = list(
        product(
            [False, True],
        )
    )
    implicit_ranker_args = [(ImplicitRanker, dict(zip(keys, v))) for v in vals]

    return [*torch_ranker_args, *implicit_ranker_args]


class TestRanker:  # pylint: disable=protected-access
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
        "distance, expected_recs, expected_scores, dense",
        (
            (
                Distance.DOT,
                [2, 0, 1, 2, 1, 0],
                [296, 25, 12, 210, 10, 6],
                True,
            ),
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
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )

        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
        )

        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )

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
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank_with_filtering_viewed_items(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ui_csr = sparse.csr_matrix(
            [
                [0, 1, 0],
                [0, 0, 0],
            ]
        )
        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )
        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
            filter_pairs_csr=ui_csr,
        )
        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )

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
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank_with_objects_whitelist(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )

        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
            sorted_object_whitelist=np.array([0, 2]),
        )
        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )

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
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank_with_objects_whitelist_and_filtering_viewed_items(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ui_csr = sparse.csr_matrix(
            [
                [1, 1, 0],
                [0, 0, 0],
            ]
        )
        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )
        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=3,
            sorted_object_whitelist=np.array([0, 2]),
            filter_pairs_csr=ui_csr,
        )
        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )

    @pytest.mark.parametrize(
        "distance, k, expected_recs, expected_scores, dense",
        (
            (
                Distance.DOT,
                2,
                [2, 0, 2, 1],
                [296, 25, 210, 10],
                True,
            ),
            (
                Distance.COSINE,
                2,
                [0, 2, 1, 2],
                [1, 0.5890328, 1, 0.9344414],
                True,
            ),
            (
                Distance.EUCLIDEAN,
                2,
                [0, 1, 1, 0],
                [0, 4.58257569, 2.23606798, 4.24264069],
                True,
            ),
            (
                Distance.DOT,
                2,
                [2, 0, 2, 1],
                [296, 25, 210, 10],
                False,
            ),
            (
                Distance.DOT,
                None,
                [2, 0, 1, 2, 1, 0],
                [296, 25, 12, 210, 10, 6],
                True,
            ),
            (
                Distance.COSINE,
                None,
                [0, 2, 1, 1, 2, 0],
                [1, 0.5890328, 0.5366563, 1, 0.9344414, 0.5366563],
                True,
            ),
            (
                Distance.EUCLIDEAN,
                None,
                [0, 1, 2, 1, 0, 2],
                [0, 4.58257569, 97.64220399, 2.23606798, 4.24264069, 98.41747812],
                True,
            ),
        ),
    )
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank_different_k(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        k: int,
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )

        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=[0, 1],
            k=k,
        )

        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )

    @pytest.mark.parametrize(
        "distance, user_ids, expected_recs, expected_scores, dense",
        (
            (
                Distance.DOT,
                [0],
                [2, 0, 1],
                [296, 25, 12],
                True,
            ),
            (
                Distance.COSINE,
                [1],
                [1, 2, 0],
                [1, 0.9344414, 0.5366563],
                True,
            ),
            (
                Distance.EUCLIDEAN,
                [0],
                [0, 1, 2],
                [0, 4.58257569, 97.64220399],
                True,
            ),
            (
                Distance.DOT,
                [1],
                [2, 1, 0],
                [210, 10, 6],
                False,
            ),
        ),
    )
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank_different_user_ids(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        user_ids: tp.List[int],
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )

        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=user_ids,
            k=3,
        )

        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )

    @pytest.mark.parametrize(
        "distance, user_ids, expected_recs, expected_scores, dense",
        (
            (
                Distance.DOT,
                [0],
                [2],
                [296],
                True,
            ),
            (
                Distance.COSINE,
                [1],
                [1, 2, 0],
                [1, 0.9344414, 0.5366563],
                True,
            ),
            (
                Distance.EUCLIDEAN,
                [0],
                [2],
                [97.64220399],
                True,
            ),
            (
                Distance.DOT,
                [1],
                [2, 1, 0],
                [210, 10, 6],
                False,
            ),
        ),
    )
    @pytest.mark.parametrize("ranker_cls, ranker_args", gen_rankers())
    def test_rank_different_user_ids_and_filter_viewed(
        self,
        ranker_cls,
        ranker_args: tp.Dict[str, tp.Any],
        distance: Distance,
        user_ids: tp.List[int],
        expected_recs: tp.List[int],
        expected_scores: tp.List[float],
        subject_factors: np.ndarray,
        object_factors: np.ndarray,
        dense: bool,
    ) -> None:
        if not dense:
            subject_factors = sparse.csr_matrix(subject_factors)

        ui_csr = sparse.csr_matrix(
            [
                [1, 1, 0],
                [0, 0, 0],
            ]
        )

        ranker: Ranker = ranker_cls(
            **ranker_args,
            distance=distance,
            subjects_factors=subject_factors,
            objects_factors=object_factors,
        )

        _, actual_recs, actual_scores = ranker.rank(
            subject_ids=user_ids,
            k=3,
            filter_pairs_csr=ui_csr[user_ids],
        )

        np.testing.assert_equal(actual_recs, expected_recs)
        np.testing.assert_almost_equal(
            actual_scores,
            expected_scores,
            decimal=EPS_DIGITS,
        )
