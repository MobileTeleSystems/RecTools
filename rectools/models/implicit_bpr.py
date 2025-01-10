import typing as tp
from copy import deepcopy

import numpy as np
import typing_extensions as tpe
from implicit.bpr import BayesianPersonalizedRanking

# pylint: disable=no-name-in-module
from implicit.cpu.bpr import BayesianPersonalizedRanking as CPUBayesianPersonalizedRanking
from implicit.gpu.bpr import BayesianPersonalizedRanking as GPUBayesianPersonalizedRanking

# pylint: enable=no-name-in-module
from pydantic import BeforeValidator, ConfigDict, SerializationInfo, WrapSerializer

from rectools.dataset.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools.models.base import ModelConfig
from rectools.models.rank import Distance
from rectools.models.vector import Factors, VectorModel
from rectools.utils.misc import get_class_or_function_full_path, import_object
from rectools.utils.serialization import DType, RandomState

BPR_STRING = "BayesianPersonalizedRanking"

AnyBayesianPersonalizedRanking = tp.Union[CPUBayesianPersonalizedRanking, GPUBayesianPersonalizedRanking]
BayesianPersonalizedRankingType = tp.Union[
    tp.Type[AnyBayesianPersonalizedRanking], tp.Literal["BayesianPersonalizedRanking"]
]


def _get_bpr_class(spec: tp.Any) -> tp.Any:
    if spec in (BPR_STRING, get_class_or_function_full_path(BayesianPersonalizedRanking)):
        return "BayesianPersonalizedRanking"
    if isinstance(spec, str):
        return import_object(spec)
    return spec


def _serialize_bpr_class(
    cls: BayesianPersonalizedRankingType, handler: tp.Callable, info: SerializationInfo
) -> tp.Union[None, str, AnyBayesianPersonalizedRanking]:
    if cls in (CPUBayesianPersonalizedRanking, GPUBayesianPersonalizedRanking) or cls == "BayesianPersonalizedRanking":
        return BPR_STRING
    if info.mode == "json":
        return get_class_or_function_full_path(cls)
    return cls


BayesianPersonalizedRankingClass = tpe.Annotated[
    BayesianPersonalizedRankingType,
    BeforeValidator(_get_bpr_class),
    WrapSerializer(
        func=_serialize_bpr_class,
        when_used="always",
    ),
]


class BayesianPersonalizedRankingConfig(tpe.TypedDict):
    """Config for implicit `BayesianPersonalizedRanking` model."""

    cls: tpe.NotRequired[BayesianPersonalizedRankingClass]
    factors: tpe.NotRequired[int]
    learning_rate: tpe.NotRequired[float]
    regularization: tpe.NotRequired[float]
    dtype: tpe.NotRequired[DType]
    num_threads: tpe.NotRequired[int]
    iterations: tpe.NotRequired[int]
    verify_negative_samples: tpe.NotRequired[bool]
    random_state: tpe.NotRequired[RandomState]
    use_gpu: tpe.NotRequired[bool]


class ImplicitBPRWrapperModelConfig(ModelConfig):
    """Config for `ImplicitBPRWrapperModel`"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: BayesianPersonalizedRankingConfig


class ImplicitBPRWrapperModel(VectorModel[ImplicitBPRWrapperModelConfig]):
    """
    Wrapper for `implicit.bpr.BayesianPersonalizedRanking` model.

    See https://benfred.github.io/implicit/api/models/cpu/bpr.html for details of the base model.

    Parameters
    ----------
    model : BayesianPersonalizedRanking
        Base model to wrap.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    recommends_for_warm = False
    recommends_for_cold = False

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    config_class = ImplicitBPRWrapperModelConfig

    def __init__(self, model: AnyBayesianPersonalizedRanking, verbose: int = 0):
        self._config = self._make_config(model, verbose)
        super().__init__(verbose=verbose)
        self.model: AnyBayesianPersonalizedRanking
        self._model = model  # for refit

        self.use_gpu = isinstance(model, GPUBayesianPersonalizedRanking)
        if not self.use_gpu:
            self.n_threads = model.num_threads

    @classmethod
    def _make_config(cls, model: AnyBayesianPersonalizedRanking, verbose: int) -> ImplicitBPRWrapperModelConfig:
        model_cls = (
            model.__class__
            if model.__class__ not in (CPUBayesianPersonalizedRanking, GPUBayesianPersonalizedRanking)
            else "BayesianPersonalizedRanking"
        )

        inner_model_config = {
            "cls": model_cls,
            "factors": model.factors,
            "learning_rate": model.learning_rate,
            "dtype": None,
            "regularization": model.regularization,
            "iterations": model.iterations,
            "verify_negative_samples": model.verify_negative_samples,
            "random_state": model.random_state,
        }
        if isinstance(model, GPUBayesianPersonalizedRanking):  # pragma: no cover
            inner_model_config["use_gpu"] = True
        else:
            inner_model_config.update(
                {
                    "use_gpu": False,
                    "dtype": model.dtype,
                    "num_threads": model.num_threads,
                }
            )

        return ImplicitBPRWrapperModelConfig(
            cls=cls,
            model=tp.cast(BayesianPersonalizedRankingConfig, inner_model_config),
            verbose=verbose,
        )

    def _get_config(self) -> ImplicitBPRWrapperModelConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: ImplicitBPRWrapperModelConfig) -> tpe.Self:
        inner_model_params = deepcopy(config.model)
        inner_model_cls = inner_model_params.pop("cls", BayesianPersonalizedRanking)
        inner_model_cls = tp.cast(tp.Callable, inner_model_cls)
        if inner_model_cls == BPR_STRING:
            inner_model_cls = BayesianPersonalizedRanking
        model = inner_model_cls(**inner_model_params)
        return cls(model=model, verbose=config.verbose)

    def _fit(self, dataset: Dataset) -> None:
        self.model = deepcopy(self._model)

        ui_csr = dataset.get_user_item_matrix(include_weights=True).astype(np.float32)
        self.model.fit(ui_csr, show_progress=self.verbose > 0)

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        return Factors(get_users_vectors(self.model))

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        return Factors(get_items_vectors(self.model))

    def get_vectors(self) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Return user and item vector representation from fitted model.

        Returns
        -------
        (np.ndarray, np.ndarray)
            User and item vectors.
            Shapes are (n_users, n_factors) and (n_items, n_factors).
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return get_users_vectors(self.model), get_items_vectors(self.model)


def get_users_vectors(model: AnyBayesianPersonalizedRanking) -> np.ndarray:
    """
    Get user vectors from BPR model as a numpy array.

    Parameters
    ----------
    model : BayesianPersonalizedRanking
        Fitted BPR model. Can be CPU or GPU model

    Returns
    -------
    np.ndarray
        User vectors.
    """
    if isinstance(model, GPUBayesianPersonalizedRanking):  # pragma: no cover
        return model.user_factors.to_numpy()
    return model.user_factors


def get_items_vectors(model: AnyBayesianPersonalizedRanking) -> np.ndarray:
    """
    Get item vectors from BPR model as a numpy array.

    Parameters
    ----------
    model : BayesianPersonalizedRanking
        Fitted BPR model. Can be CPU or GPU model

    Returns
    -------
    np.ndarray
        Item vectors.
    """
    if isinstance(model, GPUBayesianPersonalizedRanking):  # pragma: no cover
        return model.item_factors.to_numpy()
    return model.item_factors
