#  Copyright 2023-2025 MTS (Mobile Telesystems)
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

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.dataset.context import get_context
from rectools.metrics import calc_metrics
from rectools.metrics.base import MetricAtK
from rectools.models.base import ErrorBehaviour, ModelBase
from rectools.types import ExternalIds

from .splitter import Splitter


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
    on_unsupported_targets: ErrorBehaviour = "warn",
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
    on_unsupported_targets : Literal["raise", "warn", "ignore"], default "warn"
        How to handle warm/cold target users when model doesn't support warm/cold inference.
        Specify "warn" to filter with warning (default in `cross_validate`).
        Specify "ignore" to filter unsupported targets without a warning.
        It is highly recommended to pass `CoveredUsers` DQ metric to catch all models with
        insufficient recommendations for each fold.
        Specify "raise" to raise ValueError in case unsupported targets are passed. In cross-validation
        this may cause unexpected errors for some of the complicated models.


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
    split_iterator = splitter.split(dataset.interactions, collect_fold_stats=True)

    split_infos = []
    metrics_all = []

    for train_ids, test_ids, split_info in split_iterator:
        split_infos.append(split_info)

        fold_dataset = dataset.filter_interactions(
            row_indexes_to_keep=train_ids,
            keep_external_ids=True,
            keep_features_for_removed_entities=prefer_warm_inference_over_cold,
        )
        interactions_df_test = dataset.interactions.df.loc[test_ids]
        interactions_df_test[Columns.User] = dataset.user_id_map.convert_to_external(interactions_df_test[Columns.User])
        interactions_df_test[Columns.Item] = dataset.item_id_map.convert_to_external(interactions_df_test[Columns.Item])

        test_users = interactions_df_test[Columns.User].unique()
        prev_interactions = fold_dataset.get_raw_interactions()
        catalog = prev_interactions[Columns.Item].unique()
        test_fold_context = None
        if any(model.require_recommend_context for _, model in models.items()):
            test_fold_context = get_context(interactions_df_test)
        # ### Train ref models if any
        ref_reco = {}
        for model_name in ref_models or []:
            model = models[model_name]
            model.fit(fold_dataset)
            context = None
            if model.require_recommend_context:
                context = test_fold_context
            ref_reco[model_name] = model.recommend(
                users=test_users,
                dataset=fold_dataset,
                k=k,
                filter_viewed=filter_viewed,
                items_to_recommend=items_to_recommend,
                on_unsupported_targets=on_unsupported_targets,
                context=context,
            )

        # ### Generate recommendations and calc metrics
        for model_name, model in models.items():
            if model_name in ref_reco and not validate_ref_models:
                continue

            if model_name in ref_reco:
                reco = ref_reco[model_name]
            else:
                model.fit(fold_dataset)
                context = None
                if model.require_recommend_context:
                    context = test_fold_context
                reco = model.recommend(
                    users=test_users,
                    dataset=fold_dataset,
                    k=k,
                    filter_viewed=filter_viewed,
                    items_to_recommend=items_to_recommend,
                    on_unsupported_targets=on_unsupported_targets,
                    context=context,
                )

            metric_values = calc_metrics(
                metrics,
                reco=reco,
                interactions=interactions_df_test,
                prev_interactions=prev_interactions,
                catalog=catalog,
                ref_reco=ref_reco,
            )
            res = {"model": model_name, "i_split": split_info["i_split"]}
            res.update(metric_values)
            metrics_all.append(res)

    result = {"splits": split_infos, "metrics": metrics_all}
    return result
