#  Copyright 2023-2024 MTS (Mobile Telesystems)
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
import pandas as pd

from rectools.columns import Columns
from rectools.dataset import Dataset, Features, IdMap, Interactions
from rectools.metrics import calc_metrics
from rectools.metrics.base import MetricAtK
from rectools.models.base import ModelBase
from rectools.types import ExternalIds

from .splitter import Splitter


def _gen_2x_internal_ids_dataset(
    interactions_internal_df: pd.DataFrame,
    user_features: tp.Optional[Features],
    item_features: tp.Optional[Features],
    prefer_warm_inference_over_cold: bool,
) -> Dataset:
    """
    Make new dataset based on given interactions and features from base dataset.
    Assume that interactions dataframe contains internal ids.
    Returned dataset contains 2nd level of internal ids.
    """
    user_id_map = IdMap.from_values(interactions_internal_df[Columns.User].values)  # 1x internal -> 2x internal
    item_id_map = IdMap.from_values(interactions_internal_df[Columns.Item].values)  # 1x internal -> 2x internal
    interactions_train = Interactions.from_raw(interactions_internal_df, user_id_map, item_id_map)  # 2x internal

    def _handle_features(features: tp.Optional[Features], id_map: IdMap) -> tp.Tuple[tp.Optional[Features], IdMap]:
        if features is None:
            return None, id_map

        if prefer_warm_inference_over_cold:
            all_features_ids = np.arange(len(features))  # 1x internal
            id_map = id_map.add_ids(all_features_ids, raise_if_already_present=False)

        features = features.take(id_map.get_external_sorted_by_internal())  # 2x internal
        return features, id_map

    user_features_new, user_id_map = _handle_features(user_features, user_id_map)
    item_features_new, item_id_map = _handle_features(item_features, item_id_map)

    dataset = Dataset(
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        interactions=interactions_train,
        user_features=user_features_new,
        item_features=item_features_new,
    )
    return dataset


def cross_validate(  # pylint: disable=too-many-locals
    dataset: Dataset,
    splitter: Splitter,
    metrics: tp.Dict[str, MetricAtK],
    models: tp.Dict[str, ModelBase],
    k: int,
    filter_viewed: bool,
    items_to_recommend: tp.Optional[ExternalIds] = None,
    prefer_warm_inference_over_cold: bool = True,
    ref_models: tp.Optional[tp.List[str]] = None,
    validate_ref_models: bool = False,
) -> tp.Dict[str, tp.Any]:
    """
    Run cross validation on multiple models with multiple metrics.

    Parameters
    ----------
    dataset : Dataset
        Dataset with full data.
    splitter : Splitter
        Instance of any `rectools.model_selection.Splitter` subclasses.
    metrics : dict(str -> MetricAtK)
        Dict of initialized metric objects to calculate,
        where key is metric name and value is metric object.
    models : dict(str -> ModelBase)
        Dict of initialized model objects to fit and measure quality,
        where key is model name and value is model object.
    k : int
        Derived number of recommendations for every user.
        For some models actual number of recommendations may be less than `k`.
    filter_viewed : bool
        Whether to filter from recommendations items that user has already interacted with.
    items_to_recommend : array-like, optional, default None
        Whitelist of external item ids.
        If given, only these items will be used for recommendations.
    prefer_warm_inference_over_cold : bool, default True
        Whether to keep features for test users and items that were not present in train.
        Set to `True` to enable "warm" recommendations for all applicable models.
        Set to `False` to treat all new users and items as "cold" and not to provide features for them.
        If new users and items are filtered from test in splitter, this argument has no effect.
    ref_models : list(str), optional, default None
        The keys from `models` argument to compute intersection metrics. These models
        recommendations will be used as `ref_reco` for other models intersection metrics calculation.
        Obligatory only if `IntersectionMetric` instances present in `metrics`.
    validate_ref_models : bool, default False
        If True include models specified in `ref_models` to all metrics calculations
        and receive their metrics from cross-validation.

    Returns
    -------
    dict
        Dictionary with structure
        {
            "splits": [
                {"i_split": 0, <split_info>},
                {"i_split": 1, <split_info>},
                ...
            ],
            "metrics": [
                {"model": "model_1", "i_split": 0, <metrics>},
                {"model": "model_2", "i_split": 0, <metrics>},
                {"model": "model_1", "i_split": 1, <metrics>},
                ...
            ]
        }
    """
    interactions = dataset.interactions

    split_iterator = splitter.split(interactions, collect_fold_stats=True)

    split_infos = []
    metrics_all = []

    for train_ids, test_ids, split_info in split_iterator:
        split_infos.append(split_info)

        # ### Prepare split data
        interactions_df_train = interactions.df.iloc[train_ids]  # 1x internal
        # We need to avoid fitting models on sparse matrices with all zero rows/columns =>
        # => we need to create a fold dataset which contains only hot users and items for current training
        fold_dataset = _gen_2x_internal_ids_dataset(
            interactions_df_train, dataset.user_features, dataset.item_features, prefer_warm_inference_over_cold
        )

        interactions_df_test = interactions.df.iloc[test_ids]  # 1x internal
        test_users = interactions_df_test[Columns.User].unique()  # 1x internal
        catalog = interactions_df_train[Columns.Item].unique()  # 1x internal

        if items_to_recommend is not None:
            item_ids_to_recommend = dataset.item_id_map.convert_to_internal(
                items_to_recommend, strict=False
            )  # 1x internal
        else:
            item_ids_to_recommend = None

        # ### Train ref models if any
        ref_reco = {}
        for model_name in ref_models or []:
            model = models[model_name]
            model.fit(fold_dataset)
            ref_reco[model_name] = model.recommend(
                users=test_users,
                dataset=fold_dataset,
                k=k,
                filter_viewed=filter_viewed,
                items_to_recommend=item_ids_to_recommend,
            )

        # ### Generate recommendations and calc metrics
        for model_name, model in models.items():
            if model_name in ref_reco and not validate_ref_models:
                continue

            if model_name in ref_reco:
                reco = ref_reco[model_name]
            else:
                model.fit(fold_dataset)
                reco = model.recommend(  # 1x internal
                    users=test_users,
                    dataset=fold_dataset,
                    k=k,
                    filter_viewed=filter_viewed,
                    items_to_recommend=item_ids_to_recommend,
                )

            metric_values = calc_metrics(
                metrics,
                reco=reco,
                interactions=interactions_df_test,
                prev_interactions=interactions_df_train,
                catalog=catalog,
                ref_reco=ref_reco,
            )
            res = {"model": model_name, "i_split": split_info["i_split"]}
            res.update(metric_values)
            metrics_all.append(res)

    result = {"splits": split_infos, "metrics": metrics_all}
    return result
