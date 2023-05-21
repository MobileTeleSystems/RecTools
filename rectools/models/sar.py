#  Copyright 2023 MTS (Mobile Telesystems)
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
import numpy as np
from recommenders.models.sar import SAR
from rectools import ExternalIds



from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools import Columns, ExternalIds, InternalIds
from rectools.models.base import ModelBase, Scores

class SarWrapper(ModelBase) :
    """
    Simple Algorithm for Recommendations (SAR) implementation

    SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transaction history
    and items description. The core idea behind SAR is to recommend items like those that a user already has
    demonstrated an affinity to. It does this by 1) estimating the affinity of users for items, 2) estimating
    similarity across items, and then 3) combining the estimates to generate a set of recommendations for a given user.

    !!! Can't recomend items from other dataset except from original (sorted_item_ids_to_recommend does't do anything) !!!

    Parameters
    ----------
    time_decay_coefficient : float, default 30
        Number of days till ratings are decayed by 1/2
    time_now int | None, default None 
        Current time for time decay calculation
    timedecay_formula : bool, default False
        Flag to apply time decay
    """
    def __init__(self, time_decay_coefficient=30, time_now=None, timedecay_formula=False) :
        self.is_fitted = False
        self._model = SAR(
            col_user=Columns.User,
            col_item=Columns.Item,
            col_rating=Columns.Weight,
            col_timestamp=Columns.Datetime,
            col_prediction=Columns.Score,
            time_decay_coefficient=time_decay_coefficient,
            time_now=time_now,
            timedecay_formula=timedecay_formula,
            normalize=True
        )
    def _fit(self, dataset : Dataset) -> None :
        self.is_fitted = True
        self._model.fit(dataset.interactions.df)

    def _recommend_u2i(
        self,
        user_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        df = dataset.interactions.df
        
        result = self._model.recommend_k_items(df[df[Columns.User].isin(user_ids.tolist())], top_k=k, remove_seen=filter_viewed)

        return [result[Columns.User].to_numpy(), result[Columns.Item].to_numpy(), result[Columns.Score].to_numpy()]
    
    def _recommend_i2i(
        self,
        target_ids: np.ndarray,
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[np.ndarray],
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:
        res_items = pd.DataFrame(
            {
                Columns.TargetItem,
                Columns.Item,
                Columns.Score
            }
        )

        df = dataset.interactions.df

        for i in target_ids :
            tmp = self._model.get_item_based_topk(pd.DataFrame(data={Columns.Item : [i]}), top_k=k)
            tmp.drop(Columns.User)
            tmp[Columns.TargetItem] = i
            res_items.append(tmp)

        return [res_items[Columns.TargetItem].to_numpy(), res_items[Columns.Item].to_numpy(), res_items[Columns.Score].to_numpy()]