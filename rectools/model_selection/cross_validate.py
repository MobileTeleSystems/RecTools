import typing as tp
import warnings

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
) -> Dataset:
    """
    Make new dataset based on given interactions and features from base dataset.
    Assume that interactions dataframe contains internal ids.
    Returned dataset contains 2nd level of internal ids.
    """
    user_id_map = IdMap.from_values(interactions_internal_df[Columns.User].values)  # 1x internal -> 2x internal
    item_id_map = IdMap.from_values(interactions_internal_df[Columns.Item].values)  # 1x internal -> 2x internal
    interactions_train = Interactions.from_raw(interactions_internal_df, user_id_map, item_id_map)  # 2x internal
    user_features_new = item_features_new = None
    if user_features is not None:
        user_features_new = user_features.take(user_id_map.get_external_sorted_by_internal())  # 2x internal
    if item_features is not None:
        item_features_new = item_features.take(item_id_map.get_external_sorted_by_internal())  # 2x internal
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
        where key is metric name and value is metric object.
    k : int
        Derived number of recommendations for every user.
        For some models actual number of recommendations may be less than `k`.
    filter_viewed : bool
        Whether to filter from recommendations items that user has already interacted with.
    items_to_recommend : array-like, optional, default None
        Whitelist of external item ids.
        If given, only these items will be used for recommendations.

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
    if not splitter.filter_cold_users:  # TODO: remove when cold users support added
        warnings.warn(
            "Currently models do not support recommendations for cold users. "
            "Set `filter_cold_users` to `False` only for custom models. "
            "Otherwise you will get `KeyError`."
        )

    interactions = dataset.interactions

    split_iterator = splitter.split(interactions, collect_fold_stats=True)

    split_infos = []
    metrics_all = []

    for train_ids, test_ids, split_info in split_iterator:
        split_infos.append(split_info)

        interactions_df_train = interactions.df.iloc[train_ids]  # 1x internal
        # We need to avoid fitting models on sparse matrices with all zero rows/columns =>
        # => we need to create a fold dataset which contains only hot users and items for current training
        fold_dataset = _gen_2x_internal_ids_dataset(interactions_df_train, dataset.user_features, dataset.item_features)

        interactions_df_test = interactions.df.iloc[test_ids]  # 1x internal
        test_users = interactions_df_test[Columns.User].unique()  # 1x internal
        catalog = interactions_df_train[Columns.Item].unique()  # 1x internal

        if items_to_recommend is not None:
            item_ids_to_recommend = dataset.item_id_map.convert_to_internal(
                items_to_recommend, strict=False
            )  # 1x internal
        else:
            item_ids_to_recommend = None

        for model_name, model in models.items():
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
            )
            res = {"model": model_name, "i_split": split_info["i_split"]}
            res.update(metric_values)
            metrics_all.append(res)

    result = {"splits": split_infos, "metrics": metrics_all}
    return result
