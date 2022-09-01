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

"""Popular in category model."""

import typing as tp
import warnings
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from rectools import Columns, InternalIds
from rectools.dataset import Dataset, Interactions, features

from .base import Scores
from .popular import PopularModel


class MixingStrategy(Enum):
    """Types of mixing strategy"""

    ROTATE = "rotate"
    GROUP = "group"


class RatioStrategy(Enum):
    """Types of ratio strategy"""

    EQUAL = "equal"
    PROPORTIONAL = "proportional"


class PopularInCategoryModel(PopularModel):
    """
    Model generating recommendations based on popularity of items.

    Parameters
    ----------
    category_feature: str
        Name of category feature in item features dataframe.
    n_categories: int, optional, default ``None``
        Number of most popular categories to take for recommendations
    mixing_strategy: {"rotate", "group"}, default `"rotate"`
        Method of mixing recommendations from different categories. The following methods are
        available:
        - `rotate` - items from different categories take turns in final recommendations,
        starting from the most popular category
        - `group` - items from each category are grouped together. Categories are sorted
        by popularity
    ratio_strategy: {"equal", "proportional"}, default `"proportional"`
        Method of defining ratios for categories. The following methods are available:
        - `equal` - all categories gain equal ratios in recommendations. Exceeding places
        for items are given to most popular categories
        - `proportional` - categories gain ratios in recommendations based on their popularity.
        Each category gains at least one item in recommendations
        if number of categories doesn't exceed number of recs.
    popularity : {"n_users", "n_interactions", "mean_weight", "sum_weight"}, default `"n_users"`
        Method of calculating item popularity.
        To evaluate `popularity score` the following methods are available:
        - `n_users` - number of unique users that interacted with item;
        - `n_interactions` - number of interactions with item;
        - `mean_weight` - mean item interactions weight;
        - `sum_weight` - total item interactions weight.
    period : timedelta, optional, default ``None``
        Period before last interaction to consider interactions for popularity calculation.
        Either `period` or `begin_from` can be set at once.
        If both are ``None`` all interactions will be used.
    begin_from : datetime, optional, default ``None``
        Exact datetime to consider interactions from for popularity calculation.
        Either `period` or `begin_from` can be set at once.
        If both are ``None`` all interactions will be used.
    add_cold : bool, default ``False``
        If ``True`` cold items will be added to the end of popularity list and can be recommended.
        Item is cold if it's not present in interactions at all (but present in id map)
        or not present in last interactions defined by either `period` or `begin_from` arguments.
        Order of cold items is unpredictable.
        Cold items score will be equal to ``0``.
    inverse : bool, default ``False``
        If ``True`` least popular items will be selected.
    verbose : int, default ``0``
        Degree of verbose output. If ``0``, no output will be provided.
    """

    def __init__(
        self,
        category_feature: str,
        n_categories: tp.Optional[int] = None,
        mixing_strategy: tp.Optional[str] = "rotate",
        ratio_strategy: tp.Optional[str] = "proportional",
        popularity: str = "n_users",
        period: tp.Optional[timedelta] = None,
        begin_from: tp.Optional[datetime] = None,
        add_cold: bool = False,
        inverse: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            popularity=popularity,
            period=period,
            begin_from=begin_from,
            add_cold=add_cold,
            inverse=inverse,
            verbose=verbose,
        )

        self.category_feature = category_feature
        self.category_columns: tp.List[int] = []
        self.category_interactions: tp.Dict[int, pd.DataFrame] = {}
        self.category_scores: pd.Series
        self.models: tp.Dict[int, PopularModel] = {}
        self.n_effective_categories: int

        if n_categories is None or n_categories > 0:
            self.n_categories = n_categories
        else:
            raise ValueError(f"`n_categories` must be a positive number. Got {n_categories}")

        try:
            self.mixing_strategy = MixingStrategy(mixing_strategy)
        except ValueError:
            possible_values = {item.value for item in MixingStrategy.__members__.values()}
            raise ValueError(f"`mixing_strategy` must be one of the {possible_values}. Got {mixing_strategy}.")

        try:
            self.ratio_strategy = RatioStrategy(ratio_strategy)
        except ValueError:
            possible_values = {item.value for item in RatioStrategy.__members__.values()}
            raise ValueError(f"`ratio_strategy` must be one of the {possible_values}. Got {ratio_strategy}.")

    def _check_category_feature(self, dataset: Dataset) -> None:
        if not dataset.item_features:
            raise ValueError(
                "Dataset must have `item_features` for PopularInCategoryModel. "
                "Specify `item_features_df` when creating Dataset"
            )
        if not isinstance(dataset.item_features, features.SparseFeatures):
            raise TypeError("Only sparse features are supported for PopularInCategoryModel. ")
        for num_col, (name, value) in enumerate(dataset.item_features.names):
            if name == self.category_feature and value != features.DIRECT_FEATURE_VALUE:
                self.category_columns.append(num_col)
        if not self.category_columns:
            raise ValueError("`category_feature` must be present in `cat_item_features` when creating Dataset")

    def _calc_category_scores(self, dataset: Dataset, interactions: pd.DataFrame) -> None:
        scores_dict = {}
        for column_num in self.category_columns:
            item_idx = dataset.item_features.values.getcol(column_num).nonzero()[0]  # type: ignore
            self.category_interactions[column_num] = interactions[interactions[Columns.Item].isin(item_idx)].copy()
            # Category interactions might be empty
            if self.category_interactions[column_num].shape[0] == 0:
                self.category_columns.remove(column_num)
            else:
                col, func = self._get_groupby_col_and_agg_func(self.popularity)
                scores_dict[column_num] = self.category_interactions[column_num][col].apply(func)
        self.category_scores = pd.Series(scores_dict).sort_values(ascending=False)

    def _define_categories_for_analysis(self) -> None:
        if self.n_categories:
            if len(self.category_columns) >= self.n_categories:
                self.n_effective_categories = self.n_categories
                relevant_categories = self.category_scores.head(self.n_categories).index
                self.category_scores = self.category_scores.loc[relevant_categories]
                self.category_columns = relevant_categories
            else:
                self.n_effective_categories = len(self.category_columns)
                warnings.warn(
                    "`n_categories` exceeds number of unique category values. "
                    f"Only {self.n_effective_categories} categories will be analysed"
                )
        else:
            self.n_effective_categories = len(self.category_columns)

    def _fit(self, dataset: Dataset) -> None:  # type: ignore
        self._check_category_feature(dataset)
        interactions = self._filter_interactions(dataset.interactions.df)
        self._calc_category_scores(dataset, interactions)
        self._define_categories_for_analysis()

        for column_num in self.category_columns:
            category_model = PopularModel(
                popularity=self.popularity.value, add_cold=self.add_cold, inverse=self.inverse
            )
            category_interactions = Interactions(self.category_interactions[column_num])
            category_dataset = Dataset(
                user_id_map=dataset.user_id_map, item_id_map=dataset.item_id_map, interactions=category_interactions
            )
            category_model.fit(category_dataset)
            self.models[column_num] = category_model

    def _get_num_recs_for_each_category(self, k: int) -> pd.Series:
        if self.ratio_strategy == RatioStrategy.PROPORTIONAL:
            sum_scores = self.category_scores.sum()
            num_recs = np.floor(k * self.category_scores / sum_scores).astype("int32")

            # Because of np.floor not all of the required k recommendations were distributed
            exceeding_recs = k - num_recs.sum()
            num_recs.iloc[:exceeding_recs] += 1

            # Now we redistribute some of the recommendations to categories which didn't receive any numbers at all
            zero_mask = num_recs == 0
            may_decrease_mask = num_recs > 1
            num_changing_zeros = min(zero_mask.sum(), may_decrease_mask.sum())
            if num_changing_zeros > 0:
                indexes_to_increase_zeros = np.arange(len(num_recs))[zero_mask][:num_changing_zeros]
                indexes_to_decrease_others = np.arange(len(num_recs))[may_decrease_mask][-num_changing_zeros:]
                num_recs.iloc[indexes_to_increase_zeros] = 1
                num_recs.iloc[indexes_to_decrease_others] -= 1

        elif self.ratio_strategy == RatioStrategy.EQUAL:
            num_recs = pd.Series({num_col: k // self.n_effective_categories for num_col in self.category_scores.index})
            exceeding_recs = k - num_recs.sum()
            num_recs.iloc[:exceeding_recs] += 1
        return num_recs

    def _get_full_recs_from_main_and_fallback(
        self,
        main_recs: tp.List[pd.DataFrame],
        fallback_recs: tp.List[pd.DataFrame],
        k: int,
        user_ids: np.ndarray,
    ) -> pd.DataFrame:
        cat_recs = pd.concat(main_recs, sort=False)
        cat_recs.drop_duplicates(subset=[Columns.User, Columns.Item], inplace=True)

        num_recs_per_user = cat_recs[Columns.User].value_counts()
        user_w_insufficient_recs = num_recs_per_user[num_recs_per_user < k].index

        # Some users were not present in main_recs, but could be present in fallback_recs
        # within cold categories (categories with num_recs = 0). They need to be added
        # explicitly to receive recommendations
        user_w_no_recs = np.setdiff1d(user_ids, num_recs_per_user.index)
        user_w_insufficient_recs = np.union1d(user_w_insufficient_recs, user_w_no_recs)

        sufficient_mask = ~cat_recs[Columns.User].isin(user_w_insufficient_recs)
        sufficient_recs = cat_recs[sufficient_mask]
        insufficient_recs = cat_recs[~sufficient_mask].copy()
        insufficient_recs["is_main_rec"] = True

        extra_recs = pd.concat(fallback_recs, sort=False)
        extra_recs = extra_recs[extra_recs[Columns.User].isin(user_w_insufficient_recs)].copy()
        extra_recs["is_main_rec"] = False

        insufficient_recs = pd.concat([insufficient_recs, extra_recs], sort=False)
        insufficient_recs.drop_duplicates(subset=[Columns.User, Columns.Item], inplace=True)

        # Extra recommendations are given in a specific logic to guarantee that fallback recommendations
        # never replace main recommendations in final result. And popular category doesn't dominate
        # over other categories in fallback recs. Thus `rotate` mixing strategy is applied before getting
        # k recs for each user.
        insufficient_recs.sort_values(
            by=[Columns.User, "is_main_rec", "category_rank", "category_priority"],
            ascending=[True, False, True, True],
            inplace=True,
        )
        insufficient_recs = insufficient_recs.groupby(Columns.User).head(k)
        full_recs = pd.concat([sufficient_recs, insufficient_recs], sort=False)
        return full_recs

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        num_recs = self._get_num_recs_for_each_category(k)
        main_recs = []
        fallback_recs = []
        for priority, num_col in enumerate(num_recs.index):
            model = self.models[num_col]
            all_user_ids, all_reco_ids, all_scores = model._recommend_u2i(  # pylint: disable=protected-access
                user_ids=user_ids,
                dataset=dataset,
                k=k,
                filter_viewed=filter_viewed,
                sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
            )
            reco_df = pd.DataFrame(
                {
                    Columns.User: all_user_ids,
                    Columns.Item: all_reco_ids,
                    Columns.Score: all_scores,
                    "category_priority": priority,
                }
            )
            reco_df["category_rank"] = reco_df.groupby([Columns.User], sort=False).cumcount()
            main_mask = reco_df["category_rank"] < num_recs.loc[num_col]
            main_recs.append(reco_df[main_mask])
            fallback_recs.append(reco_df[~main_mask])

        full_recs = self._get_full_recs_from_main_and_fallback(main_recs, fallback_recs, k, user_ids)

        if self.mixing_strategy == MixingStrategy.GROUP:
            full_recs.sort_values(by=[Columns.User, "category_priority", "category_rank"], inplace=True)
        elif self.mixing_strategy == MixingStrategy.ROTATE:
            full_recs["category_rank"] = full_recs.groupby([Columns.User, "category_priority"], sort=False).cumcount()
            full_recs.sort_values(by=[Columns.User, "category_rank", "category_priority"], inplace=True)
        return full_recs[Columns.User].values, full_recs[Columns.Item].values, full_recs[Columns.Score].values

    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        _, single_reco, single_scores = self._recommend_u2i(
            user_ids=dataset.user_id_map.internal_ids[:1],
            dataset=dataset,
            k=k,
            filter_viewed=False,
            sorted_item_ids_to_recommend=sorted_item_ids_to_recommend,
        )

        n_targets = len(target_ids)
        n_reco_per_target = len(single_reco)

        all_target_ids = np.repeat(target_ids, n_reco_per_target)
        all_reco_ids = np.tile(single_reco, n_targets)
        all_scores = np.tile(single_scores, n_targets)
        return all_target_ids, all_reco_ids, all_scores
