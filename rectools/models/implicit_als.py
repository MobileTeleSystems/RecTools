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
from copy import deepcopy

import implicit.gpu
import numpy as np
from implicit.cpu.als import AlternatingLeastSquares as CPUAlternatingLeastSquares
from implicit.gpu.als import AlternatingLeastSquares as GPUAlternatingLeastSquares
from implicit.utils import check_random_state
from scipy import sparse
from tqdm.auto import tqdm

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError

from .vector import Distance, Factors, VectorModel

AVAILABLE_RECOMMEND_METHODS = ("loop",)
AnyAlternatingLeastSquares = tp.Union[CPUAlternatingLeastSquares, GPUAlternatingLeastSquares]
TrainedFactors = tp.Union[np.ndarray, implicit.gpu.Matrix]


class ImplicitALSWrapperModel(VectorModel):
    """
    Wrapper for `implicit.als.AlternatingLeastSquares`
    with possibility to use explicit features and GPU support.

    See https://implicit.readthedocs.io/en/latest/als.html for details of base model.

    Parameters
    ----------
    model : AnyAlternatingLeastSquares
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

    def __init__(self, model: AnyAlternatingLeastSquares, verbose: int = 0, fit_features_together: bool = False):
        super().__init__(verbose=verbose)

        self.model: AnyAlternatingLeastSquares
        self._model = model  # for refit; TODO: try to do it better

        self.fit_features_together = fit_features_together
        self.use_gpu = isinstance(model, GPUAlternatingLeastSquares)

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.model = deepcopy(self._model)
        ui_csr = dataset.get_user_item_matrix(include_weights=True).astype(np.float32)

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
        return Factors(get_users_vectors(self.model))

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        return Factors(get_items_vectors(self.model))

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
        return get_users_vectors(self.model), get_items_vectors(self.model)


def get_users_vectors(model: AnyAlternatingLeastSquares) -> np.ndarray:
    """
    Get users vectors from ALS model as numpy array

    Parameters
    ----------
    model : AnyAlternatingLeastSquares
        Model to get vectors from. Can be CPU or GPU model

    Returns
    -------
    np.ndarray
       User vectors
    """
    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
        return model.user_factors.to_numpy()
    return model.user_factors


def get_items_vectors(model: AnyAlternatingLeastSquares) -> np.ndarray:
    """
    Get items vectors from ALS model as numpy array

    Parameters
    ----------
    model : AnyAlternatingLeastSquares
        Model to get vectors from. Can be CPU or GPU model

    Returns
    -------
    np.ndarray
        Item vectors
    """
    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
        return model.item_factors.to_numpy()
    return model.item_factors


def fit_als_with_features_separately(
    model: AnyAlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    verbose: int = 0,
) -> tp.Tuple[TrainedFactors, TrainedFactors]:
    """
    Fit ALS model with explicit features, explicit features fit separately from latent.

    Parameters
    ----------
    model: AnyAlternatingLeastSquares
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
        Combined latent, explicit user factors and paired item factors.
    item_factors : np.ndarray
        Combined latent, paired user factors and explicit item factors.
    """
    iu_csr = ui_csr.T.tocsr(copy=False)
    model.fit(ui_csr, show_progress=verbose > 0)

    user_factors_chunks = [get_users_vectors(model)]
    item_factors_chunks = [get_items_vectors(model)]

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
    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
        user_factors = implicit.gpu.Matrix(user_factors)
        item_factors = implicit.gpu.Matrix(item_factors)
    return user_factors, item_factors


def _fit_paired_factors(
    model: AnyAlternatingLeastSquares, xy_csr: sparse.csr_matrix, y_factors: np.ndarray
) -> np.ndarray:
    features_model_params = {
        "factors": y_factors.shape[1],
        "regularization": model.regularization,
        "alpha": model.alpha,
        "dtype": model.dtype,
        "iterations": 1,
        "random_state": model.random_state,
    }
    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
        features_model = GPUAlternatingLeastSquares(**features_model_params)
        features_model.item_factors = implicit.gpu.Matrix(y_factors)
        features_model.fit(xy_csr)
        x_factors = features_model.user_factors.to_numpy()
    else:
        features_model_params.update(
            {
                "num_threads": model.num_threads,
                "use_native": model.use_native,
                "use_cg": model.use_cg,
            }
        )
        features_model = CPUAlternatingLeastSquares(**features_model_params)
        features_model.item_factors = y_factors
        features_model.fit(xy_csr)
        x_factors = features_model.user_factors
    return x_factors


def fit_als_with_features_together(
    model: AnyAlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    verbose: int = 0,
) -> tp.Tuple[TrainedFactors, TrainedFactors]:
    """
    Fit ALS model with explicit features, explicit features fit together with latent.

    Parameters
    ----------
    model: AnyAlternatingLeastSquares
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
        Combined explicit user factors, latent factors and user factors paired to items.
    item_factors : np.ndarray
        Combined item factors paired to users, latent factors and explicit item factors.
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
    n_latent_factors = model.factors
    model.factors = n_latent_factors + n_user_explicit_factors + n_item_explicit_factors

    # Prepare latent factors with the same math logic as in implicit library
    random_state = check_random_state(model.random_state)
    if isinstance(model, GPUAlternatingLeastSquares):
        user_latent_factors = random_state.uniform(
            low=-0.5 / n_latent_factors, high=0.5 / n_latent_factors, size=(n_users, n_latent_factors)
        )
        item_latent_factors = random_state.uniform(
            low=-0.5 / n_latent_factors, high=0.5 / n_latent_factors, size=(n_items, n_latent_factors)
        )
    else:
        user_latent_factors = random_state.random((n_users, n_latent_factors)) * 0.01
        item_latent_factors = random_state.random((n_items, n_latent_factors)) * 0.01

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

    # Give the positive examples more weight if asked for
    if model.alpha != 1.0:
        ui_csr = model.alpha * ui_csr

    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
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
    model: CPUAlternatingLeastSquares,
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

    # invalidate cached norms and squared factors
    model._item_norms = model._user_norms = None  # pylint: disable=protected-access
    model._YtY = None  # pylint: disable=protected-access
    model._XtX = None  # pylint: disable=protected-access

    solver = model.solver

    for _ in tqdm(range(model.iterations), disable=verbose == 0):

        solver(
            ui_csr,
            user_factors,
            item_factors,
            model.regularization,
            model.num_threads,
        )
        user_factors[:, :n_user_explicit_factors] = user_explicit_factors

        solver(
            iu_csr,
            item_factors,
            user_factors,
            model.regularization,
            model.num_threads,
        )
        item_factors[:, n_factors - n_item_explicit_factors :] = item_explicit_factors


def _fit_combined_factors_on_gpu_inplace(
    model: GPUAlternatingLeastSquares,
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

    iu_csr_cuda = implicit.gpu.CSRMatrix(iu_csr)
    ui_csr_cuda = implicit.gpu.CSRMatrix(ui_csr)
    X = implicit.gpu.Matrix(user_factors)
    Y = implicit.gpu.Matrix(item_factors)

    user_factors = X
    item_factors = Y

    # invalidate cached norms and squared factors
    model._item_norms = model._user_norms = None  # pylint: disable=protected-access
    model._item_norms_host = model._user_norms_host = None  # pylint: disable=protected-access
    model._YtY = model._XtX = None  # pylint: disable=protected-access

    _YtY = implicit.gpu.Matrix.zeros(model.factors, model.factors)
    _XtX = implicit.gpu.Matrix.zeros(model.factors, model.factors)

    for _ in tqdm(range(model.iterations), disable=verbose == 0):

        model.solver.calculate_yty(Y, _YtY, model.regularization)
        model.solver.least_squares(ui_csr_cuda, X, _YtY, Y, model.cg_steps)

        user_factors_np = X.to_numpy()
        user_factors_np[:, :n_user_explicit_factors] = user_explicit_factors
        X = implicit.gpu.Matrix(user_factors_np)

        model.solver.calculate_yty(X, _XtX, model.regularization)
        model.solver.least_squares(iu_csr_cuda, Y, _XtX, X, model.cg_steps)

        item_factors_np = Y.to_numpy()
        item_factors_np[:, n_factors - n_item_explicit_factors :] = item_explicit_factors
        Y = implicit.gpu.Matrix(item_factors_np)
