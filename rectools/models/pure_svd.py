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

"""SVD Model."""

import typing as tp

import numpy as np
from scipy.sparse.linalg import svds

from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.rank import Distance
from rectools.models.vector import Factors, VectorModel


class PureSVDModel(VectorModel):
    """
    PureSVD matrix factorization model.

    See https://dl.acm.org/doi/10.1145/1864708.1864721

    Parameters
    ----------
    factors : int, default ``10``
        The number of latent factors to compute.
    tol : float, default 0
        Tolerance for singular values. Zero means machine precision.
    maxiter : int, optional, default ``None``
        Maximum number of iterations.
    random_state : int, optional, default ``None``
        Pseudorandom number generator state used to generate resamples.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    recommends_for_warm = False
    recommends_for_cold = False

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    def __init__(
        self,
        factors: int = 10,
        tol: float = 0,
        maxiter: tp.Optional[int] = None,
        random_state: tp.Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.factors = factors
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state

        self.user_factors: np.ndarray
        self.item_factors: np.ndarray

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        u, sigma, vt = svds(ui_csr, k=self.factors, tol=self.tol, maxiter=self.maxiter, random_state=self.random_state)

        self.user_factors = u
        self.item_factors = (np.diag(sigma) @ vt).T

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        return Factors(self.user_factors)

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        return Factors(self.item_factors)

    def get_vectors(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Return user and item vector representations from fitted model.

        Returns
        -------
        (np.ndarray, np.ndarray)
            User and item embeddings.
            Shapes are (n_users, n_factors) and (n_items, n_factors).
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self.user_factors, self.item_factors
