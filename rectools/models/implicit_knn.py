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
import warnings
from copy import deepcopy

import implicit.nearest_neighbours
import numpy as np
import typing_extensions as tpe
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, ItemItemRecommender, TFIDFRecommender
from implicit.utils import ParameterWarning
from pydantic import BeforeValidator, ConfigDict, PlainSerializer
from scipy import sparse
from tqdm.auto import tqdm

from rectools import InternalIds
from rectools.dataset import Dataset
from rectools.types import InternalId, InternalIdsArray
from rectools.utils import fast_isin_for_sorted_test_elements
from rectools.utils.misc import get_class_or_function_full_path, import_object

from .base import ModelBase, ModelConfig, Scores
from .utils import get_viewed_item_ids, recommend_from_scores

_base_item_item_recommender_classes = (
    ItemItemRecommender,
    CosineRecommender,
    TFIDFRecommender,
    BM25Recommender,
)


def _get_item_item_recommender_class(spec: tp.Any) -> tp.Any:
    if not isinstance(spec, str):
        return spec

    base_class_names = {cls.__name__ for cls in _base_item_item_recommender_classes}
    if spec in base_class_names:
        return getattr(implicit.nearest_neighbours, spec)

    return import_object(spec)


def _serialize_item_item_recommender_class(cls: tp.Type[ItemItemRecommender]) -> str:
    if cls in _base_item_item_recommender_classes:
        return cls.__name__
    return get_class_or_function_full_path(cls)


ItemItemRecommenderClass = tpe.Annotated[
    tp.Type[ItemItemRecommender],
    BeforeValidator(_get_item_item_recommender_class),
    PlainSerializer(
        func=_serialize_item_item_recommender_class,
        return_type=str,
        when_used="json",
    ),
]


class ItemItemRecommenderConfig(tpe.TypedDict):
    """Config for `implicit` `ItemItemRecommender` model and its successors."""

    cls: ItemItemRecommenderClass
    K: tpe.NotRequired[int]
    K1: tpe.NotRequired[float]
    B: tpe.NotRequired[float]
    num_threads: tpe.NotRequired[int]


class ImplicitItemKNNWrapperModelConfig(ModelConfig):
    """Config for `ImplicitItemKNNWrapperModel`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ItemItemRecommenderConfig


class ImplicitItemKNNWrapperModel(ModelBase[ImplicitItemKNNWrapperModelConfig]):
    """
    Wrapper for `implicit.nearest_neighbours.ItemItemRecommender` and its successors.

    See https://github.com/benfred/implicit/blob/main/implicit/nearest_neighbours.py for details.

    Parameters
    ----------
    model : ItemItemRecommender
        Base model that will be used.
    verbose : int, default 0
        Degree of verbose output. If 0, no output will be provided.
    """

    recommends_for_warm = False
    recommends_for_cold = False
    config_class = ImplicitItemKNNWrapperModelConfig

    def __init__(self, model: ItemItemRecommender, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.model: ItemItemRecommender
        self._model = model

    def _get_config(self) -> ImplicitItemKNNWrapperModelConfig:
        inner_model = self._model
        inner_model_config = {
            "cls": inner_model.__class__,
            "K": inner_model.K,
            "num_threads": inner_model.num_threads,
        }
        if isinstance(inner_model, BM25Recommender):
            # NOBUG: If it's a custom class, we don't know its params
            inner_model_config.update({"K1": inner_model.K1, "B": inner_model.B})
        return ImplicitItemKNNWrapperModelConfig(
            cls=self.__class__,
            model=tp.cast(ItemItemRecommenderConfig, inner_model_config),
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: ImplicitItemKNNWrapperModelConfig) -> tpe.Self:
        model_cls = config.model["cls"]
        params = dict(config.model.copy())  # `cls` param is required and cannot be popped
        del params["cls"]
        model = model_cls(**params)
        return cls(model=model, verbose=config.verbose)

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self.model = deepcopy(self._model)

        # There is implicit conversion from float32 to float 64 in implicit library
        # in `normalize` function on the line `X.data = X.data / sqrt(bincount(X.row, X.data**2))[X.row]`.
        # But this function is not used in the base `ItemItemRecommender` class.
        # So we convert it here to make every class including the base on work.
        ui_csr = dataset.get_user_item_matrix(include_weights=True, dtype=np.float64)

        # implicit library processes weights in coo_matrix format and then warns about converting it to csr
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=ParameterWarning, message="Method expects CSR input")
            self.model.fit(ui_csr, show_progress=self.verbose > 0)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        user_items = dataset.get_user_item_matrix(include_weights=True)

        all_user_ids = []
        all_reco_ids: tp.List[int] = []
        all_scores: tp.List[float] = []
        for user_id in tqdm(user_ids, disable=self.verbose == 0):
            reco_ids, reco_scores = self._recommend_for_user(
                user_id,
                user_items,
                k,
                filter_viewed,
                sorted_item_ids_to_recommend,
            )
            all_user_ids.extend([user_id] * len(reco_ids))
            all_reco_ids.extend(reco_ids)
            all_scores.extend(reco_scores)

        return all_user_ids, all_reco_ids, all_scores

    def _recommend_for_user(
        self,
        user_id: InternalId,
        user_items: sparse.csr_matrix,
        k: int,
        filter_viewed: bool,
        sorted_item_ids: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, Scores]:
        if filter_viewed:
            viewed_ids = get_viewed_item_ids(user_items, user_id)  # sorted
        else:
            viewed_ids = np.array([], dtype=int)

        # Set filter_already_liked_items=False because if there are not enough reco it uses already liked
        # even if filter_already_liked_items=True
        if sorted_item_ids is not None:
            sorted_filtered_item_ids = sorted_item_ids[~fast_isin_for_sorted_test_elements(sorted_item_ids, viewed_ids)]
            n_items = user_items.shape[1]
            reco, scores = self.model.recommend(
                user_id, user_items[user_id], N=n_items, filter_already_liked_items=False
            )
            valid_items_mask = fast_isin_for_sorted_test_elements(reco, sorted_filtered_item_ids)

        else:
            n_items = k + viewed_ids.size
            reco, scores = self.model.recommend(
                user_id, user_items[user_id], N=n_items, filter_already_liked_items=False
            )
            valid_items_mask = fast_isin_for_sorted_test_elements(reco, viewed_ids, invert=True)

        reco = reco[valid_items_mask][:k]
        scores = scores[valid_items_mask][:k]
        return reco, scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        similarity = self.model.similarity
        if sorted_item_ids_to_recommend is not None:
            similarity = similarity[:, sorted_item_ids_to_recommend]

        all_target_ids = []
        all_reco_ids: tp.List[np.ndarray] = []
        all_scores: tp.List[np.ndarray] = []
        for target_id in tqdm(target_ids, disable=self.verbose == 0):
            reco_ids, reco_scores = self._recommend_for_item(
                similarity=similarity,
                target_id=target_id,
                k=k,
            )
            all_target_ids.extend([target_id] * len(reco_ids))
            all_reco_ids.append(reco_ids)
            all_scores.append(reco_scores)

        all_reco_ids_arr = np.concatenate(all_reco_ids)

        if sorted_item_ids_to_recommend is not None:
            all_reco_ids_arr = sorted_item_ids_to_recommend[all_reco_ids_arr]

        return all_target_ids, all_reco_ids_arr, np.concatenate(all_scores)

    @staticmethod
    def _recommend_for_item(
        similarity: sparse.csr_matrix,
        target_id: InternalId,
        k: int,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        slice_ = slice(similarity.indptr[target_id], similarity.indptr[target_id + 1])
        similar_item_ids = similarity.indices[slice_]
        similar_item_scores = similarity.data[slice_]
        reco_similar_ids, reco_scores = recommend_from_scores(similar_item_scores, k=k)
        reco_ids = similar_item_ids[reco_similar_ids]
        return reco_ids, reco_scores
