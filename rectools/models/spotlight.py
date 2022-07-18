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

import numpy as np
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.interactions import Interactions as SpotlightInteractions

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.vector import Distance, Factors, VectorModel


class SpotlightFactorizationWrapperModel(VectorModel):
    """
    Wrapper for `spotlight.factorization.implicit.ImplicitFactorizationModel`
    or for `spotlight.factorization.explicit.ExplicitFactorizationModel`.

    See https://github.com/maciejkula/spotlight/blob/master/spotlight/factorization/implicit.py for details.
    See https://github.com/maciejkula/spotlight/blob/master/spotlight/factorization/explicit.py for details.

    A wrapper for an implicit or explicit feedback matrix factorization model.
    Both models use a classic matrix factorization approach, with latent vectors used
    to represent both users and items. Their dot product gives the
    predicted score for a user-item pair.
    The latent representation is given by
    :class:`spotlight.factorization.representations.BilinearNet`.

    The implicit model is trained through negative sampling: for any known
    user-item pair, one or more items are randomly sampled to act
    as negatives (expressing a lack of preference by the user for
    the sampled item).

    Parameters
    ----------
    model : ImplicitFactorizationModel | ExplicitFactorizationModel
        Base model that will be used.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    """

    u2i_dist = Distance.DOT
    i2i_dist = Distance.COSINE

    def __init__(self, base_model: tp.Union[ImplicitFactorizationModel, ExplicitFactorizationModel], verbose: int = 0):
        super().__init__(verbose=verbose)
        self.base_model = base_model

    @staticmethod
    def _convert_to_spotlight_dataset(dataset: Dataset) -> tp.Type[SpotlightInteractions]:
        """
        Dataset class to spotlight interactions class conversion (Interactions object).

        Contains (at a minimum) pair of user-item
        interactions, but can also be enriched with ratings, timestamps,
        and interaction weights.

        For *implicit feedback* scenarios, user ids and item ids should
        only be provided for user-item pairs where an interaction was
        observed. All pairs that are not provided are treated as missing
        observations, and often interpreted as (implicit) negative
        signals.

        For *explicit feedback* scenarios, user ids, item ids, and
        ratings should be provided for all user-item-rating triplets
        that were observed in the dataset.
        ----------
        dataset: class:`rectools.dataset.Dataset`
            The dataset to convert.

        https://github.com/maciejkula/spotlight/blob/master/spotlight/interactions.py

        Returns
        -------
        interactions: class: `spotlight.interactions.Interactions`
            Spotlight dataset
        """
        interactions = SpotlightInteractions(
            user_ids=dataset.interactions.df[Columns.User].to_numpy("int32"),
            item_ids=dataset.interactions.df[Columns.Item].to_numpy("int32"),
            ratings=dataset.interactions.df[Columns.Weight].to_numpy("float32"),
            timestamps=dataset.interactions.df[Columns.Datetime].to_numpy("int64"),
        )

        return interactions

    def _fit(  # type: ignore
        self,
        dataset: Dataset,
    ) -> None:
        """
        Fit the base model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit call.

        Parameters
        ----------
        dataset: class: `rectools.dataset.Dataset`
        The input dataset.
        """
        interactions = self._convert_to_spotlight_dataset(dataset)
        self.base_model.fit(interactions, self.verbose > 0)

    def _get_users_factors(self, dataset: Dataset) -> Factors:
        # pylint: disable=protected-access
        user_embeddings = self.base_model._net.user_embeddings.weight.data.cpu().detach().numpy()
        user_biases = self.base_model._net.user_biases.weight.data.cpu().detach().numpy().reshape(-1)
        return Factors(user_embeddings, user_biases)

    def _get_items_factors(self, dataset: Dataset) -> Factors:
        # pylint: disable=protected-access
        item_embeddings = self.base_model._net.item_embeddings.weight.data.cpu().detach().numpy()
        item_biases = self.base_model._net.item_biases.weight.data.cpu().detach().numpy().reshape(-1)
        return Factors(item_embeddings, item_biases)

    def get_vectors(self, add_biases: bool = True) -> tp.Tuple[np.ndarray, np.ndarray]:
        # pylint: disable=protected-access
        """
        Return user and item vector representations with biases from fitted base model.

        Parameters
        ----------
        add_biases: bool, default True
            If `False`, only embeddings will be returned.
            If `True`, biases will be added as 2 first columns (see `Returns` section for details).

        Returns
        -------
        (np.ndarray, np.ndarray)
            User and item embeddings with biases.
        -------
        """
        user_embeddings = self.base_model._net.user_embeddings.weight.data.cpu().detach().numpy()
        item_embeddings = self.base_model._net.item_embeddings.weight.data.cpu().detach().numpy()

        if add_biases:
            user_biases = self.base_model._net.user_biases.weight.data.cpu().detach().numpy()
            item_biases = self.base_model._net.item_biases.weight.data.cpu().detach().numpy()
            user_embeddings = np.hstack((user_biases, np.ones((user_biases.size, 1)), user_embeddings))
            item_embeddings = np.hstack((np.ones((item_biases.size, 1)), item_biases, item_embeddings))

        return user_embeddings, item_embeddings
