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

import numpy as np
import typing_extensions as tpe
from lightfm import LightFM
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
from scipy import sparse

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError
from rectools.models.utils import recommend_from_scores
from rectools.types import InternalIds, InternalIdsArray
from rectools.utils.misc import get_class_or_function_full_path, import_object
from rectools.utils.serialization import RandomState

from .base import FixedColdRecoModelMixin, InternalRecoTriplet, ModelConfig, Scores
from .rank import Distance
from .vector import Factors, VectorModel

LIGHT_FM_CLS_STRING = "LightFM"


def _get_light_fm_class(spec: tp.Any) -> tp.Any:
    if not isinstance(spec, str):
        return spec
    if spec == LIGHT_FM_CLS_STRING:
        return LightFM
    return import_object(spec)


def _serialize_light_fm_class(cls: tp.Type[LightFM]) -> str:
    if cls is LightFM:
        return LIGHT_FM_CLS_STRING
    return get_class_or_function_full_path(cls)


LightFMClass = tpe.Annotated[
    tp.Type[LightFM],
    BeforeValidator(_get_light_fm_class),
    PlainSerializer(
        func=_serialize_light_fm_class,
        return_type=str,
        when_used="json",
    ),
]


class LightFMConfig(tpe.TypedDict):
    """Config for `LightFM` model."""

    cls: tpe.NotRequired[LightFMClass]
    no_components: tpe.NotRequired[int]
    k: tpe.NotRequired[int]
    n: tpe.NotRequired[int]
    learning_schedule: tpe.NotRequired[str]
    loss: tpe.NotRequired[str]
    learning_rate: tpe.NotRequired[float]
    rho: tpe.NotRequired[float]
    epsilon: tpe.NotRequired[float]
    item_alpha: tpe.NotRequired[float]
    user_alpha: tpe.NotRequired[float]
    max_sampled: tpe.NotRequired[int]
    random_state: tpe.NotRequired[RandomState]


class LightFMWrapperModelConfig(ModelConfig):
    """Config for `LightFMWrapperModel`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: LightFMConfig
    epochs: int = 1
    num_threads: int = 1
    recommend_n_threads: tp.Optional[int] = None
    recommend_use_gpu_ranking: bool = True


class LightFMWrapperModel(FixedColdRecoModelMixin, VectorModel[LightFMWrapperModelConfig]):
    """
    Wrapper for `lightfm.LightFM`.

    See https://making.lyst.com/lightfm/docs/home.html for details of base model.

    SparseFeatures are used for this model, if you use DenseFeatures, it'll be converted to sparse.
    Also it's usually better to use categorical features.
    If you have real features (age, price, etc.), you can binarize it.

    Parameters
    ----------
    model : LightFM
        Base model that will be used.
    epochs: int, default 1
        Will be used as `epochs` parameter for `LightFM.fit`.
    num_threads: int, default 1
        Will be used as `num_threads` parameter for `LightFM.fit`. Should be larger then 0.
        Can also be used as number of threads for recommendation ranking on CPU.
        See `recommend_n_threads` for details.
    recommend_n_threads: Optional[int], default ``None``
        Number of threads to use for recommendation ranking on CPU.
        Specifying ``0`` means to default to the number of cores on the machine.
        If ``None``, then number of threads will be set same as `num_threads`.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_n_threads` attribute.
    recommend_use_gpu_ranking: bool, default ``True``
        Flag to use GPU for recommendation ranking. Please note that GPU and CPU ranking may provide
        different ordering of items with identical scores in recommendation table.
        If ``True``, `implicit.gpu.HAS_CUDA` will also be checked before ranking.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_use_gpu_ranking` attribute.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    """

    recommends_for_warm = True
    recommends_for_cold = True

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    config_class = LightFMWrapperModelConfig

    def __init__(
        self,
        model: LightFM,
        epochs: int = 1,
        num_threads: int = 1,
        recommend_n_threads: tp.Optional[int] = None,
        recommend_use_gpu_ranking: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.model: LightFM
        self._model = model
        self.n_epochs = epochs
        self.n_threads = num_threads
        self._recommend_n_threads = recommend_n_threads  # used to make a config
        self.recommend_n_threads = num_threads
        if recommend_n_threads is not None:
            self.recommend_n_threads = recommend_n_threads
        self.recommend_use_gpu_ranking = recommend_use_gpu_ranking

    def _get_config(self) -> LightFMWrapperModelConfig:
        inner_model = self._model
        inner_config = {
            "cls": inner_model.__class__,
            "no_components": inner_model.no_components,
            "k": inner_model.k,
            "n": inner_model.n,
            "learning_schedule": inner_model.learning_schedule,
            "loss": inner_model.loss,
            "learning_rate": inner_model.learning_rate,
            "rho": inner_model.rho,
            "epsilon": inner_model.epsilon,
            "item_alpha": inner_model.item_alpha,
            "user_alpha": inner_model.user_alpha,
            "max_sampled": inner_model.max_sampled,
            "random_state": inner_model.initial_random_state,  # random_state is an object and can't be serialized
        }
        return LightFMWrapperModelConfig(
            cls=self.__class__,
            model=tp.cast(LightFMConfig, inner_config),  # https://github.com/python/mypy/issues/8890
            epochs=self.n_epochs,
            num_threads=self.n_threads,
            recommend_n_threads=self._recommend_n_threads,
            recommend_use_gpu_ranking=self.recommend_use_gpu_ranking,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: LightFMWrapperModelConfig) -> tpe.Self:
        params = config.model.copy()
        model_cls = params.pop("cls", LightFM)
        model = model_cls(**params)
        return cls(
            model=model,
            epochs=config.epochs,
            num_threads=config.num_threads,
            recommend_n_threads=config.recommend_n_threads,
            recommend_use_gpu_ranking=config.recommend_use_gpu_ranking,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:
        self.model = deepcopy(self._model)
        self._fit_partial(dataset, self.n_epochs)

    def _fit_partial(self, dataset: Dataset, epochs: int) -> None:
        if not self.is_fitted:
            self.model = deepcopy(self._model)

        ui_coo = dataset.get_user_item_matrix(include_weights=True).tocoo(copy=False)
        user_features = self._prepare_features(dataset.get_hot_user_features(), dataset.n_hot_users)
        item_features = self._prepare_features(dataset.get_hot_item_features(), dataset.n_hot_items)
        sample_weight = None if self._model.loss == "warp-kos" else ui_coo

        self.model.fit_partial(
            ui_coo,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=self.n_threads,
            verbose=self.verbose > 0,
        )

    @staticmethod
    def _prepare_features(features: tp.Optional[Features], n_hot: int) -> tp.Optional[sparse.csr_matrix]:
        if features is None:
            return None

        features_csr = features.get_sparse()

        identity = sparse.identity(n_hot, dtype="float32", format="csr")
        identity.resize(features_csr.shape[0], n_hot)

        features_csr = sparse.hstack(
            (
                identity,
                features_csr,
            ),
            format="csr",
        )
        return features_csr

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        user_features = self._prepare_features(dataset.user_features, dataset.n_hot_users)
        user_biases, user_embeddings = self.model.get_user_representations(user_features)
        return Factors(user_embeddings, user_biases)

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        item_features = self._prepare_features(dataset.item_features, dataset.n_hot_items)
        item_biases, item_embeddings = self.model.get_item_representations(item_features)
        return Factors(item_embeddings, item_biases)

    # pylint: disable=unsubscriptable-object
    def get_vectors(self, dataset: Dataset, add_biases: bool = True) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Return user and item vector representations from fitted model.

        Parameters
        ----------
        dataset: Dataset
            Dataset with input data.
            Usually it's the same dataset that was used to fit model.
        add_biases: bool, default True
            LightFM model stores separately embeddings and biases for users and items.
            If `False`, only embeddings will be returned.
            If `True`, biases will be added as 2 first columns (see `Returns` section for details).

        Returns
        -------
        (np.ndarray, np.ndarray)
            User and item embeddings.

            If `add_biases` is ``False``, shapes are ``(n_users, no_components)`` and ``(n_items, no_components)``.

            If `add_biases` is ``True``, shapes are ``(n_users, no_components + 2)`` and
            ``(n_items, no_components + 2)``. In that case ``(user_biases_column, ones_column)``
            will be added to user embeddings, and ``(ones_column, item_biases_column)`` - to item embeddings.
            So, if you calculate `user_embeddings @ item_embeddings.T`, for each user-item pair
            you will get value `user_embedding @ item_embedding + user_bias + item_bias`.
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        users = self._get_users_factors(dataset)
        user_embeddings = users.embeddings
        items = self._get_items_factors(dataset)
        item_embeddings = items.embeddings

        if add_biases:
            user_biases: np.ndarray = users.biases  # type: ignore
            item_biases: np.ndarray = items.biases  # type: ignore
            user_embeddings = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_embeddings))
            item_embeddings = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_embeddings))

        return user_embeddings, item_embeddings

    def _get_cold_reco(
        self, dataset: Dataset, k: int, sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray]
    ) -> tp.Tuple[InternalIds, Scores]:
        all_scores = self._get_items_factors(dataset).biases
        if all_scores is None:
            raise RuntimeError("Model must have biases")
        reco_ids, scores = recommend_from_scores(all_scores, k, sorted_whitelist=sorted_item_ids_to_recommend)
        return reco_ids, scores

    def _recommend_u2i_warm(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        return self._recommend_u2i(user_ids, dataset, k, False, sorted_item_ids_to_recommend)

    def _recommend_i2i_warm(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        return self._recommend_i2i(target_ids, dataset, k, sorted_item_ids_to_recommend)
