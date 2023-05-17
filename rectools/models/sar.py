import typing as tp
import pandas as pd
from recommenders.models.sar import SAR
from rectools import ExternalIds

from rectools.dataset import Dataset
from rectools.exceptions import NotFittedError
from rectools import Columns

sar = tp.TypeVar("sar", bound="SarWrapper")

class SarWrapper() :
    def __init__(self, time_decay_coefficient=30, time_now=None, timedecay_formula=False,) :
        self.is_fitted = False
        self._model = SAR(
            col_user=Columns.User,
            col_item=Columns.Item,
            col_rating=Columns.Weight,
            col_timestamp=Columns.Datetime,
            col_prediction=Columns.Score,
            time_decay_coefficient=30,
            time_now=None,
            timedecay_formula=False,
        )
    def _fit(self, dataset : Dataset) -> None :
        self.is_fitted = True
        self._model.fit(dataset.interactions.df)

    def fit(self : sar, dataset : Dataset) -> sar :
        self._fit(dataset)
        return self

    def recomend(self, users : ExternalIds, dataset : Dataset, k : int, filter_viewed : bool = True, add_rank_col : bool = True) -> pd.DataFrame:
        if not self.is_fitted :
            raise NotFittedError(self.__class__.__name__)
        if k < 0 :
            raise ValueError("`k` must be positive integer")
        
        try:
            user_ids = dataset.user_id_map.convert_to_internal(users)
        except KeyError:
            raise KeyError("All given users must be present in `dataset.user_id_map`")
        
        df = dataset.interactions.df
        
        result = self._model.recommend_k_items(df[df[Columns.User].isin(user_ids.tolist())], top_k=k, remove_seen=filter_viewed)

        final_result = pd.DataFrame(
            {
                Columns.User: dataset.user_id_map.convert_to_external(result[Columns.User]),
                Columns.Item: dataset.item_id_map.convert_to_external(result[Columns.Item]),
                Columns.Score: result[Columns.Score]
            }
        )
        
        if (add_rank_col) :
            final_result[Columns.Rank] = final_result.groupby(Columns.User, sort=False).cumcount() + 1

        return final_result

    def recommend_to_items(self, target_items: ExternalIds, dataset: Dataset, k: int, add_rank_col : bool = True) -> pd.DataFrame:
        if not self.is_fitted :
            raise NotFittedError(self.__class__.__name__)
        if k < 0 :
            raise ValueError("`k` must be positive integer")
        
        try:
            item_ids = dataset.item_id_map.convert_to_internal(target_items)
        except KeyError:
            raise KeyError("All given items must be present in `dataset.item_id_map`")
        
        res_items = pd.DataFrame(
            {
                Columns.TargetItem,
                Columns.Item,
                Columns.Score
            }
        )

        df = dataset.interactions.df

        for i in item_ids :
            tmp = self._model.get_item_based_topk(df[df[Columns.Item] == i], top_k=k)
            tmp.drop(Columns.User)
            tmp[Columns.TargetItem] = i
            res_items.append(tmp)

        final_result = pd.DataFrame(
            {
                Columns.TargetItem: dataset.item_id_map.convert_to_external(res_items[Columns.TargetItem]),
                Columns.Item: dataset.item_id_map.convert_to_external(res_items[Columns.Item]),
                Columns.Score: res_items[Columns.Score]
            }
        )

        if (add_rank_col) :
            final_result[Columns.Rank] = final_result.groupby(Columns.User, sort=False).cumcount() + 1
        
        return final_result