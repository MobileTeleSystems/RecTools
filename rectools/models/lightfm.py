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

import numpy as np
from lightfm import LightFM
from scipy import sparse

from rectools.dataset import Dataset, Features
from rectools.exceptions import NotFittedError

from .rank import Distance
from .vector import Factors, VectorModel


class LightFMWrapperModel(VectorModel):
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
        user_features = self._prepare_features(dataset.user_features)
        item_features = self._prepare_features(dataset.item_features)

        self.model.fit(
            ui_coo,
            user_features=user_features,
            item_features=item_features,
            sample_weight=ui_coo,
            epochs=self.n_epochs,
            num_threads=self.n_threads,
            verbose=self.verbose > 0,
        )

    @staticmethod
    def _prepare_features(features: tp.Optional[Features]) -> tp.Optional[sparse.csr_matrix]:
        if features is None:
            return None

        features_csr = features.get_sparse()
        features_csr = sparse.hstack(
            (
                sparse.identity(features_csr.shape[0], dtype="float32", format="csr"),
                features_csr,
            ),
            format="csr",
        )
        return features_csr

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        user_features = self._prepare_features(dataset.user_features)
        user_biases, user_embeddings = self.model.get_user_representations(user_features)
        return Factors(user_embeddings, user_biases)

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        item_features = self._prepare_features(dataset.item_features)
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
