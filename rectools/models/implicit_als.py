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
import warnings
from copy import deepcopy

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.utils import check_random_state
from scipy import sparse
from tqdm.auto import tqdm

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError

from .vector import Distance, Factors, VectorModel

MAX_GPU_FACTORS = 1024
AVAILABLE_RECOMMEND_METHODS = ("loop",)


class ImplicitALSWrapperModel(VectorModel):
    """
    Wrapper for `implicit.als.AlternatingLeastSquares`
    with possibility to use explicit features and GPU support.

    See https://implicit.readthedocs.io/en/latest/als.html for details of base model.

    Parameters
    ----------
    model : AlternatingLeastSquares
        Base model that will be used.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    fit_features_together: bool, default False
        Whether fit explicit features together with latent features or not.
        Used only if explicit features are present in dataset.
        See documentations linked above for details.
    """

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    def __init__(self, model: AlternatingLeastSquares, verbose: int = 0, fit_features_together: bool = False):
        super().__init__(verbose=verbose)

        if model.use_gpu and model.factors > MAX_GPU_FACTORS:  # pragma: no cover
            raise ValueError(f"When using GPU max number of factors is {MAX_GPU_FACTORS}")
        self.model: AlternatingLeastSquares
        self._model = model  # for refit; TODO: try to do it better

        self.fit_features_together = fit_features_together

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.model = deepcopy(self._model)
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        if self.fit_features_together:
            user_factors, item_factors = fit_als_with_features_together(
                self.model,
                ui_csr,
                dataset.user_features,
                dataset.item_features,
                self.verbose,
            )
        else:
            user_factors, item_factors = fit_als_with_features_separately(
                self.model,
                ui_csr,
                dataset.user_features,
                dataset.item_features,
                self.verbose,
            )

        self.model.user_factors = user_factors
        self.model.item_factors = item_factors

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        return Factors(self.model.user_factors)

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        return Factors(self.model.item_factors)

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
        return self.model.user_factors, self.model.item_factors


def fit_als_with_features_separately(
    model: AlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    verbose: int = 0,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Fit ALS model with explicit features, explicit features fit separately from latent.

    Parameters
    ----------
    model: AlternatingLeastSquares
        Base model to fit.
    ui_csr : sparse.csr_matrix
        Matrix of interactions.
    user_features : (SparseFeatures | DenseFeatures), optional
        Explicit user features.
    item_features : (SparseFeatures | DenseFeatures), optional
        Explicit item features.
    verbose : int
         Whether to print output.

    Returns
    -------
    user_factors : np.ndarray
        Combined latent and explicit user factors.
    item_factors : np.ndarray
        Combined latent and explicit user factors.
    """
    iu_csr = ui_csr.T.tocsr(copy=False)
    model.fit(iu_csr, show_progress=verbose > 0)

    user_factors_chunks = [model.user_factors]
    item_factors_chunks = [model.item_factors]

    if user_features is not None:
        user_feature_factors = user_features.get_dense()
        item_factors_paired_to_user_features = _fit_paired_factors(model, iu_csr, user_feature_factors)
        user_factors_chunks.append(user_feature_factors)
        item_factors_chunks.append(item_factors_paired_to_user_features)

    if item_features is not None:
        item_feature_factors = item_features.get_dense()
        user_factors_paired_to_item_features = _fit_paired_factors(model, ui_csr, item_feature_factors)
        item_factors_chunks.append(item_feature_factors)
        user_factors_chunks.append(user_factors_paired_to_item_features)

    user_factors = np.hstack(user_factors_chunks)
    item_factors = np.hstack(item_factors_chunks)
    return user_factors, item_factors


def _fit_paired_factors(model: AlternatingLeastSquares, xy_csr: sparse.csr_matrix, y_factors: np.ndarray) -> np.ndarray:
    if model.use_gpu:  # pragma: no cover
        paired_factors = _fit_paired_factors_on_gpu(model, xy_csr, y_factors)
    else:
        paired_factors = _fit_paired_factors_on_cpu(model, xy_csr, y_factors)
    return paired_factors


def _fit_paired_factors_on_cpu(
    model: AlternatingLeastSquares,
    xy_csr: sparse.csr_matrix,
    y_factors: np.ndarray,
) -> np.ndarray:
    x_factors = np.zeros(shape=(xy_csr.shape[0], y_factors.shape[1]), dtype=y_factors.dtype)
    model.solver(
        xy_csr,
        x_factors,
        y_factors,
        model.regularization,
        model.num_threads,
    )
    return x_factors


def _fit_paired_factors_on_gpu(
    model: AlternatingLeastSquares,
    xy_csr: sparse.csr_matrix,
    y_factors: np.ndarray,
) -> np.ndarray:  # pragma: no cover
    try:
        from implicit.cuda import (  # pylint: disable=import-outside-toplevel
            CuCSRMatrix,
            CuDenseMatrix,
            CuLeastSquaresSolver,
        )
    except ImportError:
        raise RuntimeError("implicit.cuda is not available")

    n_factors = y_factors.shape[1]
    if n_factors > MAX_GPU_FACTORS:
        raise ValueError(f"When using GPU max number of factors is {MAX_GPU_FACTORS}, here is {n_factors} factors")

    x_factors = np.zeros(shape=(xy_csr.shape[0], n_factors), dtype=y_factors.dtype)
    x_cuda = CuDenseMatrix(x_factors)
    y_cuda = CuDenseMatrix(y_factors)
    xy_csr_cuda = CuCSRMatrix(xy_csr)

    solver = CuLeastSquaresSolver(n_factors)
    solver.least_squares(xy_csr_cuda, x_cuda, y_cuda, model.regularization, model.cg_steps)

    x_cuda.to_host(x_factors)
    return x_factors


def fit_als_with_features_together(
    model: AlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    verbose: int = 0,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Fit ALS model with explicit features, explicit features fit together with latent.

    Parameters
    ----------
    model: AlternatingLeastSquares
        Base model to fit.
    ui_csr : sparse.csr_matrix
        Matrix of interactions.
    user_features : (SparseFeatures | DenseFeatures), optional
        Explicit user features.
    item_features : (SparseFeatures | DenseFeatures), optional
        Explicit item features.
    verbose : int
         Whether to print output.

    Returns
    -------
    user_factors : np.ndarray
        Combined latent and explicit user factors.
    item_factors : np.ndarray
        Combined latent and explicit user factors.
    """
    n_users, n_items = ui_csr.shape

    # Prepare explicit factors
    user_explicit_factors: np.ndarray
    if user_features is None:
        user_explicit_factors = np.array([]).reshape((n_users, 0))
    else:
        user_explicit_factors = user_features.get_dense()
    n_user_explicit_factors = user_explicit_factors.shape[1]

    item_explicit_factors: np.ndarray
    if item_features is None:
        item_explicit_factors = np.array([]).reshape((n_items, 0))
    else:
        item_explicit_factors = item_features.get_dense()
    n_item_explicit_factors = item_explicit_factors.shape[1]

    # Fix number of factors
    n_factors_all = model.factors + n_user_explicit_factors + n_item_explicit_factors
    if model.use_gpu and n_factors_all % 32:  # pragma: no cover
        padding = 32 - n_factors_all % 32
        warnings.warn(
            "GPU training requires number of factors to be a multiple of 32."
            f" Increasing factors from {n_factors_all} to {n_factors_all + padding}"
            f" (increasing latent factors from {model.factors} to {model.factors + padding})"
        )
        n_latent_factors = model.factors + padding
    else:
        n_latent_factors = model.factors
    n_factors_all = n_latent_factors + n_user_explicit_factors + n_item_explicit_factors
    model.factors = n_factors_all

    # Prepare latent factors
    random_state = check_random_state(model.random_state)
    user_latent_factors = random_state.rand(n_users, n_latent_factors) * 0.01
    item_latent_factors = random_state.rand(n_items, n_latent_factors) * 0.01

    # Prepare paired factors
    user_factors_paired_to_items = np.zeros((n_users, n_item_explicit_factors))
    item_factors_paired_to_users = np.zeros((n_items, n_user_explicit_factors))

    # Make full factors
    user_factors = np.hstack(
        (
            user_explicit_factors,
            user_latent_factors,
            user_factors_paired_to_items,
        )
    ).astype(model.dtype)
    item_factors = np.hstack(
        (
            item_factors_paired_to_users,
            item_latent_factors,
            item_explicit_factors,
        )
    ).astype(model.dtype)

    ui_csr = ui_csr.astype(np.float32)
    if model.use_gpu:  # pragma: no cover
        _fit_combined_factors_on_gpu_inplace(
            model,
            ui_csr,
            user_factors,
            item_factors,
            n_user_explicit_factors,
            n_item_explicit_factors,
            verbose,
        )
    else:
        _fit_combined_factors_on_cpu_inplace(
            model,
            ui_csr,
            user_factors,
            item_factors,
            n_user_explicit_factors,
            n_item_explicit_factors,
            verbose,
        )

    return user_factors, item_factors


def _fit_combined_factors_on_cpu_inplace(
    model: AlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_user_explicit_factors: int,
    n_item_explicit_factors: int,
    verbose: int,
) -> None:
    n_factors = user_factors.shape[1]
    user_explicit_factors = user_factors[:, :n_user_explicit_factors].copy()
    item_explicit_factors = item_factors[:, n_factors - n_item_explicit_factors :].copy()
    iu_csr = ui_csr.T.tocsr(copy=False)

    for _ in tqdm(range(model.iterations), disable=verbose == 0):
        model.solver(
            ui_csr,
            user_factors,
            item_factors,
            model.regularization,
            model.num_threads,
        )
        user_factors[:, :n_user_explicit_factors] = user_explicit_factors

        model.solver(
            iu_csr,
            item_factors,
            user_factors,
            model.regularization,
            model.num_threads,
        )
        item_factors[:, n_factors - n_item_explicit_factors :] = item_explicit_factors


def _fit_combined_factors_on_gpu_inplace(
    model: AlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_user_explicit_factors: int,
    n_item_explicit_factors: int,
    verbose: int,
) -> None:  # pragma: no cover
    try:
        from implicit.cuda import (  # pylint: disable=import-outside-toplevel
            CuCSRMatrix,
            CuDenseMatrix,
            CuLeastSquaresSolver,
        )
    except ImportError:
        raise RuntimeError("implicit.cuda is not available")

    n_factors = user_factors.shape[1]
    if n_factors > MAX_GPU_FACTORS:
        raise ValueError(f"When using GPU max number of factors is {MAX_GPU_FACTORS}, here is {n_factors} factors")

    user_explicit_factors = user_factors[:, :n_user_explicit_factors].copy()
    item_explicit_factors = item_factors[:, n_factors - n_item_explicit_factors :].copy()
    iu_csr = ui_csr.T.tocsr(copy=False)

    iu_csr_cuda = CuCSRMatrix(iu_csr)
    ui_csr_cuda = CuCSRMatrix(ui_csr)
    user_factors_cuda = CuDenseMatrix(user_factors)
    item_factors_cuda = CuDenseMatrix(item_factors)
    solver = CuLeastSquaresSolver(n_factors)

    for _ in tqdm(range(model.iterations), disable=verbose == 0):
        solver.least_squares(
            ui_csr_cuda,
            user_factors_cuda,
            item_factors_cuda,
            model.regularization,
            model.cg_steps,
        )
        user_factors_cuda.to_host(user_factors)
        user_factors[:, :n_user_explicit_factors] = user_explicit_factors
        user_factors_cuda = CuDenseMatrix(user_factors)

        solver.least_squares(
            iu_csr_cuda,
            item_factors_cuda,
            user_factors_cuda,
            model.regularization,
            model.cg_steps,
        )
        item_factors_cuda.to_host(item_factors)
        item_factors[:, n_factors - n_item_explicit_factors :] = item_explicit_factors
        item_factors_cuda = CuDenseMatrix(item_factors)

    user_factors_cuda.to_host(user_factors)
    item_factors_cuda.to_host(item_factors)
