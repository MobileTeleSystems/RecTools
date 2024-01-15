import typing as tp

import pandas as pd
import pytest

from rectools import Columns
from rectools.visuals.visual_app import AppDataStorage


class TestAppDataStorage:
    @pytest.fixture
    def recos(self) -> tp.Dict[tp.Hashable, pd.DataFrame]:
        recos = {
            "model1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
            "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
        }
        return recos

    @pytest.fixture
    def item_data(self) -> pd.DataFrame:
        item_data = pd.DataFrame(
            {Columns.Item: [3, 4, 5, 6, 7, 8], "feature_1": ["one", "two", "three", "five", "one", "two"]}
        )
        return item_data

    @pytest.fixture
    def interactions(self) -> pd.DataFrame:
        interactions = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
        return interactions

    def test_u2i(self, recos: tp.Dict[tp.Hashable, pd.DataFrame], item_data: pd.DataFrame, interactions: pd.DataFrame):
        # corner case: empty dict, no available ids
        requests_dict = {"user_one": 1}
        ads = AppDataStorage(
            recos=recos, item_data=item_data, interactions=interactions, is_u2i=True, requests_dict=requests_dict
        )

        assert ads.is_u2i
        assert ads.request_colname == Columns.User
        assert ads.requests_dict == requests_dict
        assert ads.model_names == ["model1", "model2"]
        assert ads.request_names == ["user_one"]

        assert list(ads.processed_interactions.keys()) == ["user_one"]
        expected_interactions = pd.DataFrame({Columns.Item: [3, 7], "feature_1": ["one", "one"]})
        pd.testing.assert_frame_equal(ads.processed_interactions["user_one"], expected_interactions)

        expected_processed_recos = {
            "model1": {"user_one": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.99]})},
            "model2": {"user_one": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]})},
        }
        assert expected_processed_recos.keys() == ads.processed_recos.keys()
        for model_name, model_recos in expected_processed_recos.items():
            assert model_recos.keys() == ads.processed_recos[model_name].keys()
            for user_name, user_recos in model_recos.items():
                pd.testing.assert_frame_equal(user_recos, ads.processed_recos[model_name][user_name])


