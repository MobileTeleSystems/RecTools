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
from implicit.cpu.als import AlternatingLeastSquares as CpuAlternatingLeastSquares
from implicit.gpu.als import AlternatingLeastSquares as GpuAlternatingLeastSquares
from implicit.utils import check_random_state
from scipy import sparse
from tqdm.auto import tqdm

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError

from .vector import Distance, Factors, VectorModel

MAX_GPU_FACTORS = 1024
AVAILABLE_RECOMMEND_METHODS = ("loop",)
AlternatingLeastSquares = tp.Union[CpuAlternatingLeastSquares, GpuAlternatingLeastSquares]


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

        if fit_features_together:
            raise NotImplementedError("We temporarily do not support fitting features together")

        if isinstance(model, GpuAlternatingLeastSquares) and model.factors > MAX_GPU_FACTORS:  # pragma: no cover
            raise ValueError(f"When using GPU max number of factors is {MAX_GPU_FACTORS}")
        self.model: AlternatingLeastSquares
        self._model = model  # for refit; TODO: try to do it better

        self.fit_features_together = fit_features_together

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.model = deepcopy(self._model)
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        if self.fit_features_together:
            raise NotImplementedError("We temporarily do not support fitting features together")
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
    model.fit(ui_csr, show_progress=verbose > 0)

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
    feaures_model_params = dict(
        factors=y_factors.shape[1],
        regularization=model.regularization,
        alpha=model.alpha,
        dtype=model.dtype,
        iterations=1,
        random_state=model.random_state
    )
    if isinstance(model, GpuAlternatingLeastSquares):  # pragma: no cover
        features_model = GpuAlternatingLeastSquares(**feaures_model_params)
    else:
        features_model = CpuAlternatingLeastSquares(**feaures_model_params)  
    features_model.item_factors = y_factors
    features_model.fit(xy_csr)
    return features_model.user_factors