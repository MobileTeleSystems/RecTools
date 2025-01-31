import typing as tp

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from rectools import Columns

from .candidate_ranking import Reranker


class CatBoostReranker(Reranker):
    """
    A reranker using CatBoost models for classification or ranking tasks.

    This class supports both `CatBoostClassifier` and `CatBoostRanker` models to rerank candidates
    based on their features and optionally provided additional parameters for fitting and pool creation.
    """

    def __init__(
        self,
        model: tp.Union[CatBoostClassifier, CatBoostRanker],
        fit_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        pool_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        """
        Initialize the CatBoostReranker with `model`, `fit_kwargs` and `pool_kwargs`.

        Parameters
        ----------
        model : ClassifierBase | RankerBase
            A CatBoost model instance used for reranking. Can be either a classifier or a ranker.
        fit_kwargs : dict(str -> any), optional, default ``None``
            Additional keyword arguments to be passed to the `fit` method of the CatBoost model.
        pool_kwargs : dict(str -> any), optional, default ``None``
            Additional keyword arguments to be used when creating the CatBoost `Pool`.
        """
        super().__init__(model)
        self.is_classifier = isinstance(model, CatBoostClassifier)
        self.fit_kwargs = fit_kwargs
        self.pool_kwargs = pool_kwargs

    def prepare_training_pool(self, candidates_with_target: pd.DataFrame) -> Pool:
        """
        Prepare a CatBoost `Pool` for training from the given candidates with target.

        Depending on whether the model is a classifier or a ranker, the pool is prepared differently.
        For classifiers, only data and label are used. For rankers, group information is also included.

        Parameters
        ----------
        candidates_with_target : pd.DataFrame
            DataFrame containing candidate features and target values, along with user and item identifiers.

        Returns
        -------
        Pool
            A CatBoost Pool object ready for training.
        """
        if self.is_classifier:
            pool_kwargs = {
                "data": candidates_with_target.drop(columns=Columns.UserItem + [Columns.Target]),
                "label": candidates_with_target[Columns.Target],
            }
        else:
            candidates_with_target = candidates_with_target.sort_values(by=[Columns.User])
            pool_kwargs = {
                "data": candidates_with_target.drop(columns=Columns.UserItem + [Columns.Target]),
                "label": candidates_with_target[Columns.Target],
                "group_id": candidates_with_target[Columns.User].values,
            }

        if self.pool_kwargs is not None:
            pool_kwargs.update(self.pool_kwargs)

        return Pool(**pool_kwargs)

    def fit(self, candidates_with_target: pd.DataFrame) -> None:
        """
        Fit the CatBoost model using the given candidates with target data.

        This method prepares the training pool and fits the model using the specified fit parameters.

        Parameters
        ----------
        candidates_with_target : pd.DataFrame
            DataFrame containing candidate features and target values, along with user and item identifiers.

        Returns
        -------
        None
        """
        training_pool = self.prepare_training_pool(candidates_with_target)

        fit_kwargs = {"X": training_pool}
        if self.fit_kwargs is not None:
            fit_kwargs.update(self.fit_kwargs)

        self.model.fit(**fit_kwargs)
