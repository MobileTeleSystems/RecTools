#  Copyright 2022-2025 MTS (Mobile Telesystems)
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
import warnings

import numpy as np
import typing_extensions as tpe
from scipy.sparse.linalg import svds

from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.base import ModelConfig
from rectools.models.rank import Distance
from rectools.models.vector import Factors, VectorModel

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse.linalg import svds as cupy_svds
except ImportError:  # pragma: no cover
    cupy_svds = None
    cp_csr_matrix = None
    cp = None


class PureSVDModelConfig(ModelConfig):
    """Config for `PureSVD` model."""

    factors: int = 10
    tol: float = 0
    maxiter: tp.Optional[int] = None
    random_state: tp.Optional[int] = None
    use_gpu: tp.Optional[bool] = False
    recommend_n_threads: int = 0
    recommend_use_gpu_ranking: bool = True


class PureSVDModel(VectorModel[PureSVDModelConfig]):
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
        Pseudorandom number generator state used to generate resamples. Omitted if use_gpu is True.
    use_gpu : bool, default ``False``
        If ``True``, `cupyx.scipy.sparse.linalg.svds()` is used instead of SciPy. CuPy is required.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    recommend_n_threads: int, default 0
        Number of threads to use for recommendation ranking on CPU.
        Specifying ``0`` means to default to the number of cores on the machine.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_n_threads` attribute.
    recommend_use_gpu_ranking: bool, default ``True``
        Flag to use GPU for recommendation ranking. Please note that GPU and CPU ranking may provide
        different ordering of items with identical scores in recommendation table.
        If ``True``, `implicit.gpu.HAS_CUDA` will also be checked before ranking.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_use_gpu_ranking` attribute.
    """

    recommends_for_warm = False
    recommends_for_cold = False

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    config_class = PureSVDModelConfig

    def __init__(
        self,
        factors: int = 10,
        tol: float = 0,
        maxiter: tp.Optional[int] = None,
        random_state: tp.Optional[int] = None,
        use_gpu: tp.Optional[bool] = False,
        verbose: int = 0,
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
    ):
        super().__init__(verbose=verbose)

        self.factors = factors
        self.tol = tol
        self.maxiter = maxiter
        self.random_state = random_state
        self._use_gpu = use_gpu  # for making a config
        if use_gpu:  # pragma: no cover
            if not cp:
                warnings.warn("Forced to use CPU. CuPy is not available.")
                use_gpu = False
            elif not cp.cuda.is_available():
                warnings.warn("Forced to use CPU. GPU is not available.")
                use_gpu = False

        self.use_gpu = use_gpu
        self.recommend_n_threads = recommend_n_threads
        self.recommend_use_gpu_ranking = recommend_use_gpu_ranking

        self.user_factors: np.ndarray
        self.item_factors: np.ndarray

    def _get_config(self) -> PureSVDModelConfig:
        return PureSVDModelConfig(
            cls=self.__class__,
            factors=self.factors,
            tol=self.tol,
            maxiter=self.maxiter,
            random_state=self.random_state,
            use_gpu=self._use_gpu,
            verbose=self.verbose,
            recommend_n_threads=self.recommend_n_threads,
            recommend_use_gpu_ranking=self.recommend_use_gpu_ranking,
        )

    @classmethod
    def _from_config(cls, config: PureSVDModelConfig) -> tpe.Self:
        return cls(
            factors=config.factors,
            tol=config.tol,
            maxiter=config.maxiter,
            random_state=config.random_state,
            use_gpu=config.use_gpu,
            verbose=config.verbose,
            recommend_n_threads=config.recommend_n_threads,
            recommend_use_gpu_ranking=config.recommend_use_gpu_ranking,
        )

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        if self.use_gpu:  # pragma: no cover
            ui_csr = cp_csr_matrix(ui_csr)
            # To prevent IndexError, we need to subtract 1 from factors
            u, sigma, vt = cupy_svds(ui_csr.toarray(), k=self.factors - 1, tol=self.tol, maxiter=self.maxiter)
            u = u.get()
            self.item_factors = (cp.diag(sigma) @ vt).T.get()
        else:
            u, sigma, vt = svds(
                ui_csr, k=self.factors, tol=self.tol, maxiter=self.maxiter, random_state=self.random_state
            )
            self.item_factors = (np.diag(sigma) @ vt).T

        self.user_factors = u

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
