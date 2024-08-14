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

import typing as tp
from copy import deepcopy

import numpy as np
from lightfm import LightFM
from scipy import sparse
from sklearn.base import clone

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError
from rectools.models.utils import recommend_from_scores
from rectools.types import InternalIds, InternalIdsArray

from .base import FixedColdRecoModelMixin, InternalRecoTriplet, Scores
from .rank import Distance
from .vector import Factors, VectorModel


class LightFMWrapperModel(FixedColdRecoModelMixin, VectorModel):
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
        Will be used as `num_threads` parameter for `LightFM.fit`.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    """

    recommends_for_warm = True
    recommends_for_cold = True

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    def __init__(
        self,
        model: LightFM,
        epochs: int = 1,
        num_threads: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)

        self.model: LightFM
        self._model = model
        self.n_epochs = epochs
        self.n_threads = num_threads

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.model = deepcopy(self._model)

        ui_coo = dataset.get_user_item_matrix(include_weights=True).tocoo(copy=False)
        user_features = self._prepare_features(dataset.get_hot_user_features(), dataset.n_hot_users)
        item_features = self._prepare_features(dataset.get_hot_item_features(), dataset.n_hot_items)
        sample_weight = None if self._model.loss == "warp-kos" else ui_coo

        self.model.fit(
            ui_coo,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=self.n_epochs,
            num_threads=self.n_threads,
            verbose=self.verbose > 0,
        )

    def _fit_partial(self, dataset: Dataset, epochs: tp.Optional[int] = None) -> None:
        if not self.is_fitted:
            self.model = deepcopy(self._model)

        ui_coo = dataset.get_user_item_matrix(include_weights=True).tocoo(copy=False)
        user_features = self._prepare_features(dataset.get_hot_user_features(), dataset.n_hot_users)
        item_features = self._prepare_features(dataset.get_hot_item_features(), dataset.n_hot_items)
        epochs = epochs if epochs is not None else self.n_epochs
        sample_weight = None if self._model.loss == "warp-kos" else ui_coo

        if self.is_fitted:
            self.model._check_initialized()  # pylint: disable=W0212
            self._resize_model(ui_coo, user_features, item_features)

        self.model.fit_partial(
            ui_coo,
            user_features=user_features,
            item_features=item_features,
            sample_weight=sample_weight,
            epochs=epochs,
            num_threads=self.n_threads,
            verbose=self.verbose > 0,
        )

    # Based on LightFMResizable by @JohnPaton
    # https://github.com/lyst/lightfm/issues/347#issuecomment-707829342
    def _resize_model(
        self,
        interactions: sparse.coo_matrix,
        user_features: tp.Optional[sparse.csr_matrix] = None,
        item_features: tp.Optional[sparse.csr_matrix] = None,
    ) -> None:
        """Resizes the model to accommodate new users/items/features"""
        no_components = self.model.no_components
        no_user_features, no_item_features = interactions.shape

        if user_features and hasattr(user_features, "shape"):
            no_user_features = user_features.shape[-1]
        if item_features and hasattr(item_features, "shape"):
            no_item_features = item_features.shape[-1]

        if (
            no_user_features == self.model.user_embeddings.shape[0]
            and no_item_features == self.model.item_embeddings.shape[0]
        ):
            return

        new_model = clone(self.model)
        new_model._initialize(no_components, no_item_features, no_user_features)  # pylint: disable=W0212

        for attr in (
            "item_embeddings",
            "item_embedding_gradients",
            "item_embedding_momentum",
            "item_biases",
            "item_bias_gradients",
            "item_bias_momentum",
            "user_embeddings",
            "user_embedding_gradients",
            "user_embedding_momentum",
            "user_biases",
            "user_bias_gradients",
            "user_bias_momentum",
        ):
            # extend attribute matrices with new rows/cols from
            # freshly initialized model with right shape
            old_array = getattr(self.model, attr)
            old_slice = [slice(None, i) for i in old_array.shape]
            new_array = getattr(new_model, attr)
            new_array[tuple(old_slice)] = old_array
            setattr(self.model, attr, new_array)

        return

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
