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

import implicit.cpu
import numpy as np
import pytest
import torch
from scipy import sparse

from rectools.models.rank import Distance, ImplicitRanker
from rectools.models.rank_torch import TorchRanker

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

    return torch_ranker_args


class TestTorchRanker:  # pylint: disable=protected-access
    @pytest.fixture
    def subject_factors(self) -> torch.Tensor:
        return torch.from_numpy(np.array([[-4, 0, 3], [0, 1, 2]]))

    @pytest.fixture
    def object_factors(self) -> torch.Tensor:
        return torch.from_numpy(
            np.array(
                [
                    [-4, 0, 3],
                    [0, 2, 4],
                    [1, 10, 100],
                ]
            )
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
