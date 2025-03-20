#  Copyright 2024-2025 MTS (Mobile Telesystems)
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

"""EASE model."""

import typing as tp
import warnings

import numpy as np
import typing_extensions as tpe
from implicit.gpu import HAS_CUDA
from scipy import sparse

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.models.base import ModelConfig
from rectools.types import InternalIdsArray

from .base import ModelBase, Scores
from .rank import Distance, ImplicitRanker


class EASEModelConfig(ModelConfig):
    """Config for `EASE` model."""

    regularization: float = 500.0
    recommend_n_threads: int = 0
    recommend_use_gpu_ranking: bool = True


class EASEModel(ModelBase[EASEModelConfig]):
    """
    Embarrassingly Shallow Autoencoders for Sparse Data model.

    See https://arxiv.org/abs/1905.03375.

    Please note that this algorithm requires a lot of RAM during `fit` method.
    Out-of-memory issues are possible for big datasets.
    Reasonable catalog size for local development is about 30k items.
    Reasonable amount of interactions is about 20m.

    Parameters
    ----------
    regularization : float
        The regularization factor of the weights.
    num_threads: Optional[int], default ``None``
        Deprecated, use `recommend_n_threads` instead.
        Number of threads used for recommendation ranking on CPU.
    recommend_n_threads: int, default 0
        Number of threads to use for recommendation ranking on CPU.
        Specifying ``0`` means to default to the number of cores on the machine.
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

    recommends_for_warm = False
    recommends_for_cold = False

    config_class = EASEModelConfig

    def __init__(
        self,
        regularization: float = 500.0,
        num_threads: tp.Optional[int] = None,
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.weight: np.ndarray
        self.regularization = regularization

        if num_threads is not None:
            warnings.warn(
                """
            `num_threads` argument is deprecated and will be removed in future releases.
            Please use `recommend_n_threads` instead.
            """
            )
            recommend_n_threads = num_threads

        self.recommend_n_threads = recommend_n_threads
        self.recommend_use_gpu_ranking = recommend_use_gpu_ranking

    def _get_config(self) -> EASEModelConfig:
        return EASEModelConfig(
            cls=self.__class__,
            regularization=self.regularization,
            recommend_n_threads=self.recommend_n_threads,
            recommend_use_gpu_ranking=self.recommend_use_gpu_ranking,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: EASEModelConfig) -> tpe.Self:
        return cls(
            regularization=config.regularization,
            recommend_n_threads=config.recommend_n_threads,
            recommend_use_gpu_ranking=config.recommend_use_gpu_ranking,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        ui_csr = dataset.get_user_item_matrix(include_weights=True)

        gram_matrix = ui_csr.T @ ui_csr
        gram_matrix += self.regularization * sparse.identity(gram_matrix.shape[0]).astype(np.float32)
        gram_matrix = gram_matrix.todense()

        gram_matrix_inv = np.linalg.inv(gram_matrix)

        self.weight = np.array(gram_matrix_inv / (-np.diag(gram_matrix_inv)))
        np.fill_diagonal(self.weight, 0.0)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        user_items = dataset.get_user_item_matrix(include_weights=True)

        ranker = ImplicitRanker(
            distance=Distance.DOT,
            subjects_factors=user_items,
            objects_factors=self.weight,
            use_gpu=self.recommend_use_gpu_ranking and HAS_CUDA,
            num_threads=self.recommend_n_threads,
        )

        ui_csr_for_filter = user_items[user_ids] if filter_viewed else None

        all_user_ids, all_reco_ids, all_scores = ranker.rank(
            subject_ids=user_ids,
            k=k,
            filter_pairs_csr=ui_csr_for_filter,
            sorted_object_whitelist=sorted_item_ids_to_recommend,
        )

        return all_user_ids, all_reco_ids, all_scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        similarity = self.weight[target_ids]
        if sorted_item_ids_to_recommend is not None:
            similarity = similarity[:, sorted_item_ids_to_recommend]

        n_reco = min(k, similarity.shape[1])
        unsorted_reco_positions = similarity.argpartition(-n_reco, axis=1)[:, -n_reco:]
        unsorted_reco_scores = np.take_along_axis(similarity, unsorted_reco_positions, axis=1)

        sorted_reco_positions = unsorted_reco_scores.argsort()[:, ::-1]

        all_reco_scores = np.take_along_axis(unsorted_reco_scores, sorted_reco_positions, axis=1)
        all_reco_ids = np.take_along_axis(unsorted_reco_positions, sorted_reco_positions, axis=1)

        all_target_ids = np.repeat(target_ids, n_reco)

        if sorted_item_ids_to_recommend is not None:
            all_reco_ids = sorted_item_ids_to_recommend[all_reco_ids]

        return all_target_ids, all_reco_ids.flatten(), all_reco_scores.flatten()
