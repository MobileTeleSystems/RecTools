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

import typing as tp
from copy import deepcopy

import implicit.gpu
import numpy as np
import typing_extensions as tpe
from implicit.als import AlternatingLeastSquares
from implicit.cpu.als import AlternatingLeastSquares as CPUAlternatingLeastSquares
from implicit.gpu.als import AlternatingLeastSquares as GPUAlternatingLeastSquares
from implicit.utils import check_random_state
from pydantic import BeforeValidator, ConfigDict, SerializationInfo, WrapSerializer
from scipy import sparse
from tqdm.auto import tqdm

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError
from rectools.models.base import ModelConfig
from rectools.utils.misc import get_class_or_function_full_path, import_object
from rectools.utils.serialization import DType, RandomState

from .rank import Distance
from .utils import convert_arr_to_implicit_gpu_matrix
from .vector import Factors, VectorModel

ALS_STRING = "AlternatingLeastSquares"

AnyAlternatingLeastSquares = tp.Union[CPUAlternatingLeastSquares, GPUAlternatingLeastSquares]
AlternatingLeastSquaresType = tp.Union[tp.Type[AnyAlternatingLeastSquares], tp.Literal["AlternatingLeastSquares"]]


def _get_alternating_least_squares_class(spec: tp.Any) -> tp.Any:
    if spec in (ALS_STRING, get_class_or_function_full_path(AlternatingLeastSquares)):
        return "AlternatingLeastSquares"
    if isinstance(spec, str):
        return import_object(spec)
    return spec


def _serialize_alternating_least_squares_class(
    cls: AlternatingLeastSquaresType, handler: tp.Callable, info: SerializationInfo
) -> tp.Union[None, str, AnyAlternatingLeastSquares]:
    if cls in (CPUAlternatingLeastSquares, GPUAlternatingLeastSquares) or cls == "AlternatingLeastSquares":
        return ALS_STRING
    if info.mode == "json":
        return get_class_or_function_full_path(cls)
    return cls


AlternatingLeastSquaresClass = tpe.Annotated[
    AlternatingLeastSquaresType,
    BeforeValidator(_get_alternating_least_squares_class),
    WrapSerializer(
        func=_serialize_alternating_least_squares_class,
        when_used="always",
    ),
]


class AlternatingLeastSquaresConfig(tpe.TypedDict):
    """Config for implicit `AlternatingLeastSquares` model."""

    cls: tpe.NotRequired[AlternatingLeastSquaresClass]
    factors: tpe.NotRequired[int]
    regularization: tpe.NotRequired[float]
    alpha: tpe.NotRequired[float]
    dtype: tpe.NotRequired[DType]
    use_native: tpe.NotRequired[bool]
    use_cg: tpe.NotRequired[bool]
    use_gpu: tpe.NotRequired[bool]
    iterations: tpe.NotRequired[int]
    calculate_training_loss: tpe.NotRequired[bool]
    num_threads: tpe.NotRequired[int]
    random_state: tpe.NotRequired[RandomState]


class ImplicitALSWrapperModelConfig(ModelConfig):
    """Config for `ImplicitALSWrapperModel`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: AlternatingLeastSquaresConfig
    fit_features_together: bool = False
    recommend_n_threads: tp.Optional[int] = None
    recommend_use_gpu_ranking: tp.Optional[bool] = None


class ImplicitALSWrapperModel(VectorModel[ImplicitALSWrapperModelConfig]):
    """
    Wrapper for `implicit.als.AlternatingLeastSquares`
    with possibility to use explicit features and GPU support.

    See https://implicit.readthedocs.io/en/latest/als.html for details of base model.

    Parameters
    ----------
    model : AnyAlternatingLeastSquares
        Base model that will be used.
    fit_features_together: bool, default False
        Whether fit explicit features together with latent features or not.
        Used only if explicit features are present in dataset.
        See documentations linked above for details.
    recommend_n_threads: Optional[int], default ``None``
        Number of threads to use for recommendation ranking on CPU.
        Specifying ``0`` means to default to the number of cores on the machine.
        If ``None``, then number of threads will be set same as `model.num_threads`.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_n_threads` attribute.
    recommend_use_gpu_ranking: Optional[bool], default ``None``
        Flag to use GPU for recommendation ranking. If ``None``, then will be set same as
        `model.use_gpu`.
        `implicit.gpu.HAS_CUDA` will also be checked before inference.  Please note that GPU and CPU
        ranking may provide different ordering of items with identical scores in recommendation
        table. If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_use_gpu_ranking` attribute.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    """

    recommends_for_warm = False
    recommends_for_cold = False

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    config_class = ImplicitALSWrapperModelConfig

    def __init__(
        self,
        model: AnyAlternatingLeastSquares,
        fit_features_together: bool = False,
        recommend_n_threads: tp.Optional[int] = None,
        recommend_use_gpu_ranking: tp.Optional[bool] = None,
        verbose: int = 0,
    ):
        self._config = self._make_config(
            model=model,
            verbose=verbose,
            fit_features_together=fit_features_together,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
        )

        super().__init__(verbose=verbose)

        self.model: AnyAlternatingLeastSquares
        self._model = model  # for refit

        self.fit_features_together = fit_features_together

        if recommend_n_threads is None:
            recommend_n_threads = model.num_threads if isinstance(model, CPUAlternatingLeastSquares) else 0
        self.recommend_n_threads = recommend_n_threads

        if recommend_use_gpu_ranking is None:
            recommend_use_gpu_ranking = isinstance(model, GPUAlternatingLeastSquares)
        self.recommend_use_gpu_ranking = recommend_use_gpu_ranking

    @classmethod
    def _make_config(
        cls,
        model: AnyAlternatingLeastSquares,
        verbose: int,
        fit_features_together: bool,
        recommend_n_threads: tp.Optional[int] = None,
        recommend_use_gpu_ranking: tp.Optional[bool] = None,
    ) -> ImplicitALSWrapperModelConfig:
        model_cls = (
            model.__class__
            if model.__class__ not in (CPUAlternatingLeastSquares, GPUAlternatingLeastSquares)
            else "AlternatingLeastSquares"
        )
        inner_model_config = {
            "cls": model_cls,
            "factors": model.factors,
            "regularization": model.regularization,
            "alpha": model.alpha,
            "dtype": model.dtype,
            "iterations": model.iterations,
            "calculate_training_loss": model.calculate_training_loss,
            "random_state": model.random_state,
        }
        if isinstance(model, GPUAlternatingLeastSquares):
            inner_model_config.update({"use_gpu": True})
        else:
            inner_model_config.update(
                {
                    "use_gpu": False,
                    "use_native": model.use_native,
                    "use_cg": model.use_cg,
                    "num_threads": model.num_threads,
                }
            )

        return ImplicitALSWrapperModelConfig(
            cls=cls,
            # https://github.com/python/mypy/issues/8890
            model=tp.cast(AlternatingLeastSquaresConfig, inner_model_config),
            verbose=verbose,
            fit_features_together=fit_features_together,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
        )

    def _get_config(self) -> ImplicitALSWrapperModelConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: ImplicitALSWrapperModelConfig) -> tpe.Self:
        inner_model_params = config.model.copy()
        inner_model_cls = inner_model_params.pop("cls", AlternatingLeastSquares)
        if inner_model_cls == ALS_STRING:
            inner_model_cls = AlternatingLeastSquares  # Not actually a class, but it's ok
        model = inner_model_cls(**inner_model_params)  # type: ignore  # mypy misses we replaced str with a func
        return cls(
            model=model,
            verbose=config.verbose,
            fit_features_together=config.fit_features_together,
            recommend_n_threads=config.recommend_n_threads,
            recommend_use_gpu_ranking=config.recommend_use_gpu_ranking,
        )

    def _fit(self, dataset: Dataset) -> None:
        self.model = deepcopy(self._model)
        self._fit_model_for_epochs(dataset, self.model.iterations)

    def _fit_partial(self, dataset: Dataset, epochs: int) -> None:
        if not self.is_fitted:
            self.model = deepcopy(self._model)
            prev_epochs = 0
        else:
            prev_epochs = self.model.iterations

        self._fit_model_for_epochs(dataset, epochs)
        self.model.iterations = epochs + prev_epochs

    def _fit_model_for_epochs(self, dataset: Dataset, epochs: int) -> None:
        ui_csr = dataset.get_user_item_matrix(include_weights=True).astype(np.float32)

        if self.fit_features_together:
            fit_als_with_features_together_inplace(
                self.model,
                ui_csr,
                dataset.get_hot_user_features(),
                dataset.get_hot_item_features(),
                epochs,
                self.verbose,
            )
        else:
            fit_als_with_features_separately_inplace(
                self.model,
                ui_csr,
                dataset.get_hot_user_features(),
                dataset.get_hot_item_features(),
                epochs,
                self.verbose,
            )

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


def fit_als_with_features_separately_inplace(
    model: AnyAlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    iterations: int,
    verbose: int = 0,
) -> None:
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
    """
    # If model was fitted we should drop any learnt embeddings except actual latent factors
    if model.user_factors is not None and model.item_factors is not None:
        user_factors = get_users_vectors(model)[:, : model.factors]
        item_factors = get_items_vectors(model)[:, : model.factors]
        _set_factors(model, user_factors, item_factors)

    iu_csr = ui_csr.T.tocsr(copy=False)
    model.iterations = iterations
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

    _set_factors(model, user_factors, item_factors)


def _set_factors(model: AnyAlternatingLeastSquares, user_factors: np.ndarray, item_factors: np.ndarray) -> None:
    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
        user_factors = convert_arr_to_implicit_gpu_matrix(user_factors)
        item_factors = convert_arr_to_implicit_gpu_matrix(item_factors)
    model.user_factors = user_factors
    model.item_factors = item_factors


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
        features_model.item_factors = convert_arr_to_implicit_gpu_matrix(y_factors)
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
        features_model.item_factors = y_factors.copy()
        features_model.fit(xy_csr)
        x_factors = features_model.user_factors
    return x_factors


def _init_latent_factors_cpu(
    model: CPUAlternatingLeastSquares, n_users: int, n_items: int
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Logic is copied and pasted from original implicit library code.
    This method is used only for model that hasn't been fitted yet.
    """
    random_state = check_random_state(model.random_state)
    user_latent_factors = random_state.random((n_users, model.factors)) * 0.01
    item_latent_factors = random_state.random((n_items, model.factors)) * 0.01
    return user_latent_factors, item_latent_factors


def _init_latent_factors_gpu(
    model: GPUAlternatingLeastSquares, n_users: int, n_items: int
) -> tp.Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """
    Logic is copied and pasted from original implicit library code.
    This method is used only for model that hasn't been fitted yet.
    """
    random_state = check_random_state(model.random_state)
    user_latent_factors = random_state.uniform(
        low=-0.5 / model.factors, high=0.5 / model.factors, size=(n_users, model.factors)
    )
    item_latent_factors = random_state.uniform(
        low=-0.5 / model.factors, high=0.5 / model.factors, size=(n_items, model.factors)
    )
    return user_latent_factors, item_latent_factors


def fit_als_with_features_together_inplace(
    model: AnyAlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    iterations: int,
    verbose: int = 0,
) -> None:
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
    """
    n_users, n_items = ui_csr.shape

    if model.user_factors is None or model.item_factors is None:
        user_factors, item_factors, n_user_explicit_factors, n_item_explicit_factors = (
            _init_user_item_factors_for_combined_training_with_features(
                model, n_users, n_items, user_features, item_features
            )
        )
    else:
        user_factors = get_users_vectors(model)
        item_factors = get_items_vectors(model)
        n_user_explicit_factors = user_features.values.shape[1] if user_features is not None else 0
        n_item_explicit_factors = item_features.values.shape[1] if item_features is not None else 0

    # Fix number of factors
    n_latent_factors = model.factors
    model.factors = n_latent_factors + n_user_explicit_factors + n_item_explicit_factors

    # Give the positive examples more weight if asked for (implicit library logic copy)
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
            iterations,
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
            iterations,
        )

    # Fix back model factors
    model.factors = n_latent_factors


def _init_user_item_factors_for_combined_training_with_features(
    model: AnyAlternatingLeastSquares,
    n_users: int,
    n_items: int,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
) -> tp.Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Init user and item factors for model that hasn't been initialized yet.
    Final factors will include latent factors, explicit factors from
    user/item features and their paired item/user factors.
    This method is only used when `fit_features_together` is set to ``True``
    """
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

    # Prepare latent factors with the same math logic as in implicit library
    if isinstance(model, GPUAlternatingLeastSquares):  # pragma: no cover
        user_latent_factors, item_latent_factors = _init_latent_factors_gpu(model, n_users, n_items)
    else:
        user_latent_factors, item_latent_factors = _init_latent_factors_cpu(model, n_users, n_items)

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

    return user_factors, item_factors, n_user_explicit_factors, n_item_explicit_factors


def _fit_combined_factors_on_cpu_inplace(
    model: CPUAlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_user_explicit_factors: int,
    n_item_explicit_factors: int,
    verbose: int,
    iterations: int,
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

    for _ in tqdm(range(iterations), disable=verbose == 0):

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

    model.user_factors = user_factors
    model.item_factors = item_factors


def _fit_combined_factors_on_gpu_inplace(
    model: GPUAlternatingLeastSquares,
    ui_csr: sparse.csr_matrix,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    n_user_explicit_factors: int,
    n_item_explicit_factors: int,
    verbose: int,
    iterations: int,
) -> None:  # pragma: no cover
    n_factors = user_factors.shape[1]
    user_explicit_factors = user_factors[:, :n_user_explicit_factors].copy()
    item_explicit_factors = item_factors[:, n_factors - n_item_explicit_factors :].copy()
    iu_csr = ui_csr.T.tocsr(copy=False)

    iu_csr_cuda = implicit.gpu.CSRMatrix(iu_csr)
    ui_csr_cuda = implicit.gpu.CSRMatrix(ui_csr)
    X = convert_arr_to_implicit_gpu_matrix(user_factors)
    Y = convert_arr_to_implicit_gpu_matrix(item_factors)

    # invalidate cached norms and squared factors
    model._item_norms = model._user_norms = None  # pylint: disable=protected-access
    model._item_norms_host = model._user_norms_host = None  # pylint: disable=protected-access
    model._YtY = model._XtX = None  # pylint: disable=protected-access

    _YtY = implicit.gpu.Matrix.zeros(*item_factors.shape)
    _XtX = implicit.gpu.Matrix.zeros(*user_factors.shape)

    for _ in tqdm(range(iterations), disable=verbose == 0):

        model.solver.calculate_yty(Y, _YtY, model.regularization)
        model.solver.least_squares(ui_csr_cuda, X, _YtY, Y, model.cg_steps)

        user_factors_np = X.to_numpy()
        user_factors_np[:, :n_user_explicit_factors] = user_explicit_factors
        X = convert_arr_to_implicit_gpu_matrix(user_factors_np)

        model.solver.calculate_yty(X, _XtX, model.regularization)
        model.solver.least_squares(iu_csr_cuda, Y, _XtX, X, model.cg_steps)

        item_factors_np = Y.to_numpy()
        item_factors_np[:, n_factors - n_item_explicit_factors :] = item_explicit_factors
        Y = convert_arr_to_implicit_gpu_matrix(item_factors_np)

    model.user_factors = X
    model.item_factors = Y
