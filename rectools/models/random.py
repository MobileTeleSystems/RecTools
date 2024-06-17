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

"""Random Model."""

import random
import typing as tp

import numpy as np
from tqdm.auto import tqdm

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.types import AnyIdsArray, InternalId, InternalIdsArray
from rectools.utils import fast_isin_for_sorted_test_elements

from .base import ModelBase, Scores, SemiInternalRecoTriplet
from .utils import get_viewed_item_ids


class _RandomGen:
    def __init__(self, random_state: tp.Optional[int] = None) -> None:
        self.python_gen = random.Random(random_state)  # nosec
        self.np_gen = np.random.default_rng(random_state)


class _RandomSampler:
    def __init__(self, values: np.ndarray, random_gen: _RandomGen) -> None:
        self.python_gen = random_gen.python_gen
        self.np_gen = random_gen.np_gen
        self.values = values
        self.values_list = list(values)  # for random.sample

    def sample(self, n: int) -> np.ndarray:
        if n < 25:  # Empiric value, for optimization
            sampled = np.asarray(self.python_gen.sample(self.values_list, n))
        else:
            sampled = self.np_gen.choice(self.values, n, replace=False)
        return sampled


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

    recommends_for_warm = False
    recommends_for_cold = True

    def __init__(self, random_state: tp.Optional[int] = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.random_state = random_state
        self.random_gen = _RandomGen(random_state)

        self.all_item_ids: np.ndarray

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.all_item_ids = dataset.item_id_map.internal_ids

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        if filter_viewed:
            user_items = dataset.get_user_item_matrix(include_weights=False)

        item_ids = sorted_item_ids_to_recommend if sorted_item_ids_to_recommend is not None else self.all_item_ids
        sampler = _RandomSampler(item_ids, self.random_gen)

        all_user_ids = []
        all_reco_ids: tp.List[InternalId] = []
        all_scores: tp.List[float] = []
        for user_id in tqdm(user_ids, disable=self.verbose == 0):
            if filter_viewed:
                viewed_ids = get_viewed_item_ids(user_items, user_id)  # sorted
                n_reco = k + viewed_ids.size
            else:
                n_reco = k

            n_reco = min(n_reco, item_ids.size)
            reco_ids = sampler.sample(n_reco)

            if filter_viewed:
                reco_ids = reco_ids[fast_isin_for_sorted_test_elements(reco_ids, viewed_ids, invert=True)][:k]

            reco_scores = np.arange(reco_ids.size, 0, -1)

            all_user_ids.extend([user_id] * len(reco_ids))
            all_reco_ids.extend(reco_ids.tolist())
            all_scores.extend(reco_scores.tolist())

        return all_user_ids, all_reco_ids, all_scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        return self._recommend_u2i(target_ids, dataset, k, False, sorted_item_ids_to_recommend)

    def _recommend_cold(
        self,
        target_ids: AnyIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> SemiInternalRecoTriplet:
        item_ids = sorted_item_ids_to_recommend if sorted_item_ids_to_recommend is not None else self.all_item_ids
        sampler = _RandomSampler(item_ids, self.random_gen)
        n_reco = min(k, item_ids.size)

        reco_ids_lst = []
        for _ in tqdm(target_ids, disable=self.verbose == 0):
            reco_ids = sampler.sample(n_reco)
            reco_ids_lst.append(reco_ids)

        reco_item_ids = np.concatenate(reco_ids_lst)
        reco_target_ids = np.repeat(target_ids, n_reco)
        reco_scores = np.tile(np.arange(n_reco, 0, -1), len(target_ids))
        return reco_target_ids, reco_item_ids, reco_scores
