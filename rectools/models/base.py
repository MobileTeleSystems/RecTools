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

from rectools import AnyIds, Columns, InternalIds
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
        users: AnyIds,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: tp.Optional[AnyIds] = None,
        add_rank_col: bool = True,
        assume_external_ids: bool = True,
        return_external_ids: bool = True,
    ) -> pd.DataFrame:
        r"""
        Recommend items for users.

        To use this method model must be fitted.

        Parameters
        ----------
        users : array-like
            Array of user ids to recommend for.
            User ids are supposed to be external if `assume_external_ids` is ``True`` (default).
            Internal otherwise.
        dataset : Dataset
            Dataset with input data.
            Usually it's the same dataset that was used to fit model.
        k : int
            Derived number of recommendations for every user.
            Pay attention that in some cases real number of recommendations may be less than `k`.
        filter_viewed : bool
            Whether to filter from recommendations items that user has already interacted with.
        items_to_recommend : array-like, optional, default None
            Whitelist of item ids.
            If given, only these items will be used for recommendations.
            Otherwise all items from dataset will be used.
            Item ids are supposed to be external if `assume_external_ids` is `True`` (default).
            Internal otherwise.
        add_rank_col : bool, default True
            Whether to add rank column to recommendations.
            If True column `Columns.Rank` will be added.
            This column contain integers from 1 to ``number of user recommendations``.
            In any case recommendations are sorted per rank for every user.
            The lesser the rank the more recommendation is relevant.
        assume_external_ids : bool, default True
            When ``True`` all input user and item ids are supposed to be external.
            Internal otherwise. Works faster with ``False``.
        return_external_ids : bool, default True
            When ``True`` user and item ids in returning recommendations table will be external.
            Internal otherwise. Works faster with ``False``.

        Returns
        -------
        pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Score`[, `Columns.Rank`].
            External user and item ids are used by default. For internal ids set `return_external_ids` to ``False``.
            1st column contains user ids,
            2nd - ids of recommended items sorted by relevance for each user,
            3rd - score that model gives for the user-item pair,
            4th (present only if `add_rank_col` is ``True``) - integers from ``1`` to number of user recommendations.

        Raises
        ------
        NotFittedError
            If called for not fitted model.
        TypeError, ValueError
            If arguments have inappropriate type or value
        KeyError
            If some of given users are not in `dataset.user_id_map`
        """
        self._check_is_fitted()
        self._check_k(k)

        if assume_external_ids:
            try:
                user_ids = dataset.user_id_map.convert_to_internal(users)
            except KeyError:
                raise KeyError("All given users must be present in `dataset.user_id_map`")
        else:
            user_ids = np.asarray(users)
            if not np.issubdtype(user_ids.dtype, np.integer):
                raise TypeError("Internal user ids are always integer")

        sorted_item_ids_to_recommend = self._get_sorted_item_ids_to_recommend(
            items_to_recommend, dataset, assume_external_ids
        )

        reco_user_ids, reco_item_ids, reco_scores = self._recommend_u2i(
            user_ids,
            dataset,
            k,
            filter_viewed,
            sorted_item_ids_to_recommend,
        )

        if return_external_ids:
            reco_user_ids = dataset.user_id_map.convert_to_external(reco_user_ids)
            reco_item_ids = dataset.item_id_map.convert_to_external(reco_item_ids)

        reco = self._make_reco_table(reco_user_ids, reco_item_ids, reco_scores, Columns.User, add_rank_col)
        return reco

    def recommend_to_items(
        self,
        target_items: AnyIds,
        dataset: Dataset,
        k: int,
        filter_itself: bool = True,
        items_to_recommend: tp.Optional[AnyIds] = None,
        add_rank_col: bool = True,
        assume_external_ids: bool = True,
        return_external_ids: bool = True,
    ) -> pd.DataFrame:
        """
        Recommend items for target items.

        To use this method model must be fitted.

        Parameters
        ----------
        target_items : array-like
            Array of item ids to recommend for.
            Item ids are supposed to be external if `assume_external_ids` is `True`` (default).
            Internal otherwise.
        dataset : Dataset
            Dataset with input data.
            Usually it's the same dataset that was used to fit model.
        k : int
            Derived number of recommendations for every target item.
            Pay attention that in some cases real number of recommendations may be less than `k`.
        filter_itself : bool, default True
            If True, item will be excluded from recommendations to itself.
        items_to_recommend : array-like, optional, default None
            Whitelist of item ids.
            If given, only these items will be used for recommendations.
            Otherwise all items from dataset will be used.
            Item ids are supposed to be external if `assume_external_ids` is `True`` (default).
            Internal otherwise.
        add_rank_col : bool, default True
             Whether to add rank column to recommendations.
             If True column `Columns.Rank` will be added.
             This column contain integers from 1 to ``number of item recommendations``.
             In any case recommendations are sorted per rank for every target item.
             Less rank means more relevant recommendation.
        assume_external_ids : bool, default True
            When ``True`` all input item ids are supposed to be external.
            Internal otherwise. Works faster with ``False``.
        return_external_ids : bool, default True
            When ``True`` item ids in returning recommendations table will be external.
            Internal otherwise. Works faster with ``False``.

        Returns
        -------
        pd.DataFrame
            Recommendations table with columns `Columns.TargetItem`, `Columns.Item`, `Columns.Score`[, `Columns.Rank`].
            External item ids are used by default. For internal ids set `return_external_ids` to ``False``.
            1st column contains target item ids,
            2nd - ids of recommended items sorted by relevance for each target item,
            3rd - score that model gives for the target-item pair,
            4th (present only if `add_rank_col` is ``True``) - integers from 1 to number of recommendations.

        Raises
        ------
        NotFittedError
            If called for not fitted model.
        TypeError, ValueError
            If arguments have inappropriate type or value
        KeyError
            If some of given target items are not in `dataset.item_id_map`
        """
        self._check_is_fitted()
        self._check_k(k)

        if assume_external_ids:
            try:
                target_ids = dataset.item_id_map.convert_to_internal(target_items)
            except KeyError:
                raise KeyError("All given target items must be present in `dataset.item_id_map`")
        else:
            target_ids = np.asarray(target_items)
            if not np.issubdtype(target_ids.dtype, np.integer):
                raise TypeError("Internal item ids are always integer")

        sorted_item_ids_to_recommend = self._get_sorted_item_ids_to_recommend(
            items_to_recommend, dataset, assume_external_ids
        )

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

        if return_external_ids:
            reco_target_ids = dataset.item_id_map.convert_to_external(reco_target_ids)
            reco_item_ids = dataset.item_id_map.convert_to_external(reco_item_ids)

        reco = self._make_reco_table(reco_target_ids, reco_item_ids, reco_scores, Columns.TargetItem, add_rank_col)
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

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

    @classmethod
    def _check_k(cls, k: int) -> None:
        if k <= 0:
            raise ValueError("`k` must be positive integer")

    @classmethod
    def _make_reco_table(
        cls,
        subject_ids: AnyIds,
        item_ids: AnyIds,
        scores: Scores,
        subject_col: str,
        add_rank_col: bool,
    ) -> pd.DataFrame:
        reco = pd.DataFrame(
            {
                subject_col: subject_ids,
                Columns.Item: item_ids,
                Columns.Score: scores,
            }
        )

        if add_rank_col:
            reco[Columns.Rank] = reco.groupby(subject_col, sort=False).cumcount() + 1

        return reco

    @classmethod
    def _get_sorted_item_ids_to_recommend(
        cls, items_to_recommend: tp.Optional[AnyIds], dataset: Dataset, assume_external_ids: bool
    ) -> tp.Optional[np.ndarray]:
        if items_to_recommend is None:
            return None

        if assume_external_ids:
            item_ids_to_recommend = dataset.item_id_map.convert_to_internal(items_to_recommend, strict=False)
        else:
            item_ids_to_recommend = np.asarray(items_to_recommend)
            if not np.issubdtype(item_ids_to_recommend.dtype, np.integer):
                raise TypeError("Internal ids are always integer")

        sorted_item_ids_to_recommend = np.unique(item_ids_to_recommend)
        return sorted_item_ids_to_recommend
