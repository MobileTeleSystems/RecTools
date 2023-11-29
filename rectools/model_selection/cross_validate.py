import typing as tp

from rectools.columns import Columns
from rectools.dataset.dataset import Dataset
from rectools.dataset.interactions import Interactions
from rectools.metrics.base import MetricAtK
from rectools.metrics.scoring import calc_metrics
from rectools.model_selection.splitter import Splitter
from rectools.models.base import ModelBase
from rectools.types import ExternalIds


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
                {"i_split": 1, <split_info>},
                {"i_split": 2, <split_info>},
                ...
            ],
            "metrics": [
                {"model": "model_1", "i_split": 1, <metrics>},
                {"model": "model_1", "i_split": 2, <metrics>},
                {"model": "model_2", "i_split": 1, <metrics>},
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

        interactions_df_train = interactions.df.iloc[train_ids]  # internal ids
        fold_dataset = Dataset(
            user_id_map=dataset.user_id_map,
            item_id_map=dataset.item_id_map,
            interactions=Interactions(interactions_df_train),
            user_features=dataset.user_features,
            item_features=dataset.item_features,
        )

        interactions_df_test = interactions.df.iloc[test_ids]  # internal ids
        test_users = interactions_df_test[Columns.User].unique()  # internal ids

        catalog = interactions_df_train[Columns.Item].unique()  # internal ids

        if items_to_recommend is not None:
            item_ids_to_recommend = dataset.item_id_map.convert_to_internal(
                items_to_recommend, strict=False
            )  # internal ids
        else:
            item_ids_to_recommend = None

        for model_name, model in models.items():
            model.fit(fold_dataset)
            reco = model.recommend(  # internal ids
                users=test_users,
                dataset=dataset,
                k=k,
                filter_viewed=filter_viewed,
                items_to_recommend=item_ids_to_recommend,
                assume_external_ids=False,
                return_external_ids=False,
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
