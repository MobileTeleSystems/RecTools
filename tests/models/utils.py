from copy import deepcopy

import pandas as pd

from rectools.dataset import Dataset
from rectools.models.base import ModelBase


def assert_second_fit_refits_model(model: ModelBase, dataset: Dataset) -> None:
    model_1 = deepcopy(model).fit(dataset)
    model_2 = deepcopy(model).fit(dataset).fit(dataset)
    k = dataset.item_id_map.external_ids.size

    reco_u2i_1 = model_1.recommend(dataset.user_id_map.external_ids, dataset, k, False)
    reco_u2i_2 = model_2.recommend(dataset.user_id_map.external_ids, dataset, k, False)
    pd.testing.assert_frame_equal(reco_u2i_1, reco_u2i_2, atol=0.001)

    reco_i2i_1 = model_1.recommend_to_items(dataset.item_id_map.external_ids, dataset, k, False)
    reco_i2i_2 = model_2.recommend_to_items(dataset.item_id_map.external_ids, dataset, k, False)
    pd.testing.assert_frame_equal(reco_i2i_1, reco_i2i_2, atol=0.001)
