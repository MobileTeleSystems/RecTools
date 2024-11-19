import typing as tp

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from rectools import Columns

from .candidate_ranking import Reranker


class CatBoostReranker(Reranker):
    """TODO: add description"""

    def __init__(
        self,
        model: tp.Union[CatBoostClassifier, CatBoostRanker] = CatBoostRanker(verbose=False),
        fit_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
        pool_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        super().__init__(model)
        self.is_classifier = isinstance(model, CatBoostClassifier)
        self.fit_kwargs = fit_kwargs
        self.pool_kwargs = pool_kwargs

    def prepare_fit_kwargs(self, candidates_with_target: pd.DataFrame) -> tp.Dict[str, tp.Any]:
        """TODO: add description"""
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

        fit_kwargs = {"X": Pool(**pool_kwargs)}

        if self.fit_kwargs is not None:
            fit_kwargs.update(self.fit_kwargs)

        return fit_kwargs