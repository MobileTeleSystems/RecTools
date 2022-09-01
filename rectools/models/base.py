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

"""Base model."""

import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns, ExternalIds, InternalIds
from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError

T = tp.TypeVar("T", bound="ModelBase")
Scores = tp.Union[tp.Sequence[float], np.ndarray]


class ModelBase:
    """
    Base model class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(self, *args: tp.Any, verbose: int = 0, **kwargs: tp.Any) -> None:
        self.is_fitted = False
        self.verbose = verbose

    def fit(self: T, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> T:
        """
        Fit model.

        Parameters
        ----------
        dataset : Dataset
            Dataset with input data.

        Returns
        -------
        self
        """
        self._fit(dataset, *args, **kwargs)
        self.is_fitted = True
        return self

    def _fit(self, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError()

    def recommend(
        self,
        users: ExternalIds,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        add_rank_col: bool = True,
    ) -> pd.DataFrame:
        r"""
        Recommend items for users.

        To use this method model must be fitted.

        Parameters
        ----------
        users : np.ndarray
            Array of external user ids to recommend for.
        dataset : Dataset
            Dataset with input data.
            Usually it's the same dataset that was used to fit model.
        k : int
            Derived number of recommendations for every user.
            Pay attention that in some cases real number of recommendations may be less than `k`.
        filter_viewed : bool
            Whether to filter from recommendations items that user has already interacted with.
        items_to_recommend : np.ndarray, optional
            Whitelist of item external ids.
            If given, only these items will be used for recommendations.
            Otherwise all items from dataset will be used.
        add_rank_col : bool, default True
            Whether to add rank column to recommendations.
            If True column `Columns.Rank` will be added.
            This column contain integers from 1 to ``number of user recommendations``.
            In any case recommendations are sorted per rank for every user.
            The lesser the rank the more recommendation is relevant.

        Returns
        -------
        pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Score`\ [, `Columns.Rank`]\.
            1st column contains external user ids,
            2nd - external ids of recommended items sorted for each user by relevance,
            3rd - score that model gives for the user-item pair,
            4th (present only if `add_rank_col` is ``True``) - integers from ``1`` to number of user recommendations.
            Recommendations for every user are always sorted by relevance.

        Raises
        ------
        NotFittedError
            If called for not fitted model.

        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        if k <= 0:
            raise ValueError("`k` must be positive integer")

        try:
            user_ids = dataset.user_id_map.convert_to_internal(users)
        except KeyError:
            raise KeyError("All given users must be present in `dataset.user_id_map`")

        if items_to_recommend is not None:
            item_ids_to_recommend = dataset.item_id_map.convert_to_internal(items_to_recommend, strict=False)
            sorted_item_ids_to_recommend = np.unique(item_ids_to_recommend)
        else:
            sorted_item_ids_to_recommend = None

        reco_user_ids, reco_item_ids, reco_scores = self._recommend_u2i(
            user_ids,
            dataset,
            k,
            filter_viewed,
            sorted_item_ids_to_recommend,
        )

        reco = pd.DataFrame(
            {
                Columns.User: dataset.user_id_map.convert_to_external(reco_user_ids),
                Columns.Item: dataset.item_id_map.convert_to_external(reco_item_ids),
                Columns.Score: reco_scores,
            }
        )

        if add_rank_col:
            reco[Columns.Rank] = reco.groupby(Columns.User, sort=False).cumcount() + 1
        return reco

    def recommend_to_items(
        self,
        target_items: ExternalIds,
        dataset: Dataset,
        k: int,
        filter_itself: bool = True,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        add_rank_col: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend items for target items.

        To use this method model must be fitted.

        Parameters
        ----------
        target_items : np.ndarray
            Array of external item ids to recommend for.
        dataset : Dataset
            Dataset with input data.
            Usually it's the same dataset that was used to fit model.
        k : int
            Derived number of recommendations for every target item.
            Pay attention that in some cases real number of recommendations may be less than `k`.
        filter_itself : bool, default True
            If True, item will be excluded from recommendations to itself.
        items_to_recommend : np.ndarray, optional, default None
             Whitelist of item external ids.
             If given, only these items will be used for recommendations.
             Otherwise all items from dataset will be used.
        add_rank_col : bool, default True
             Whether to add rank column to recommendations.
             If True column `Columns.Rank` will be added.
             This column contain integers from 1 to ``number of item recommendations``.
             In any case recommendations are sorted per rank for every target item.
             Less rank means more relevant recommendation.

        Returns
        -------
        pd.DataFrame
            Recommendations table with columns `Columns.TargetItem`, `Columns.Item`, `Columns.Score`, [,`Columns.Rank`].
            1st column contains external target item ids,
            2nd - external ids of recommended items sorted for each target item by relevance,
            3rd - score that model gives for the target-item pair,
            4th (present only if `add_rank_col` is ``True``) - integers from 1 to number of recommendations.
            Recommendations for every target item are always sorted by relevance.

        Raises
        ------
        NotFittedError
            If called for not fitted model.

        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        if k <= 0:
            raise ValueError("`k` must be positive integer")

        try:
            target_ids = dataset.item_id_map.convert_to_internal(target_items)
        except KeyError:
            raise KeyError("All given target items must be present in `dataset.item_id_map`")

        if items_to_recommend is not None:
            item_ids_to_recommend = dataset.item_id_map.convert_to_internal(items_to_recommend, strict=False)
            sorted_item_ids_to_recommend = np.unique(item_ids_to_recommend)
        else:
            sorted_item_ids_to_recommend = None

        requested_k = k + 1 if filter_itself else k

        reco_target_ids, reco_item_ids, reco_scores = self._recommend_i2i(
            target_ids,
            dataset,
            requested_k,
            sorted_item_ids_to_recommend,
        )

        if filter_itself:
            df_reco = (
                pd.DataFrame({"tid": reco_target_ids, "iid": reco_item_ids, "score": reco_scores})
                .query("tid != iid")
                .groupby("tid", sort=False)
                .head(k)
            )
            reco_target_ids, reco_item_ids, reco_scores = df_reco[["tid", "iid", "score"]].values.T

        reco = pd.DataFrame(
            {
                Columns.TargetItem: dataset.item_id_map.convert_to_external(reco_target_ids),
                Columns.Item: dataset.item_id_map.convert_to_external(reco_item_ids),
                Columns.Score: reco_scores,
            }
        )

        if add_rank_col:
            reco[Columns.Rank] = reco.groupby(Columns.TargetItem, sort=False).cumcount() + 1
        return reco

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        raise NotImplementedError()

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        raise NotImplementedError()
