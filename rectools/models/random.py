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

"""Random Model."""

import random
import typing as tp

import numpy as np
from tqdm.auto import tqdm

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.utils import fast_isin_for_sorted_test_elements

from .base import ModelBase, Scores
from .utils import get_viewed_item_ids

# Experiments have shown that for random sampling without replacement if k / n > 0.025
# where n - size of population, k - required number of samples
# it's faster to use `np.random.choice(population, k, replace=False)
# otherwise it's better to use `random.sample(population, k)
K_TO_N_MIN_NUMPY_RATIO = 0.025


class RandomModel(ModelBase):
    """
    Model generating random recommendations.

    By default all items that are present
    in `dataset.item_id_map` will be used for recommendations.

    Numbers ranging from <n recommendations for user> to 1 will be used as a "score" in recommendations.

    Parameters
    ----------
    random_state : int, optional, default ``None``
        Pseudorandom number generator state to control the sampling.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    def __init__(self, random_state: tp.Optional[int] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.random_state = random_state
        self.all_item_ids: np.ndarray

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.all_item_ids = dataset.item_id_map.internal_ids

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        if filter_viewed:
            user_items = dataset.get_user_item_matrix(include_weights=False)

        if sorted_item_ids_to_recommend is not None:
            item_ids = np.unique(sorted_item_ids_to_recommend)
        else:
            item_ids = self.all_item_ids

        item_indices = list(range(item_ids.size))  # for random.sample

        np.random.seed(self.random_state)
        random.seed(self.random_state, version=2)

        all_user_ids = []
        all_reco_ids = []
        all_scores: tp.List[float] = []
        for user_id in tqdm(user_ids, disable=self.verbose == 0):
            if filter_viewed:
                viewed_ids = get_viewed_item_ids(user_items, user_id)  # sorted
                n_reco = k + viewed_ids.size
            else:
                n_reco = k

            n_reco = min(n_reco, item_ids.size)

            if n_reco / item_ids.size < K_TO_N_MIN_NUMPY_RATIO:
                reco_indices = random.sample(item_indices, n_reco)
                reco_ids = item_ids[reco_indices]
            else:
                reco_ids = np.random.choice(item_ids, n_reco, replace=False)

            if filter_viewed:
                reco_ids = reco_ids[fast_isin_for_sorted_test_elements(reco_ids, viewed_ids, invert=True)][:k]

            reco_scores = np.arange(reco_ids.size, 0, -1)

            all_user_ids.extend([user_id] * len(reco_ids))
            all_reco_ids.extend(reco_ids)
            all_scores.extend(reco_scores)

        return all_user_ids, all_reco_ids, all_scores

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        return self._recommend_u2i(target_ids, dataset, k, False, sorted_item_ids_to_recommend)
