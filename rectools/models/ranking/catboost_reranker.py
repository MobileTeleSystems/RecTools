#  Copyright 2024 MTS (Mobile Telesystems)
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

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from rectools import Columns

from .candidate_ranking import Reranker


class CatBoostReranker(Reranker):
    """TODO: add description"""

    def __init__(
        self,
        model: tp.Union[CatBoostClassifier, CatBoostRanker],
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
