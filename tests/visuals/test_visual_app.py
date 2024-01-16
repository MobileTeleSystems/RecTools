import typing as tp

import pandas as pd
import pytest

from rectools import Columns
from rectools.visuals.visual_app import AppDataStorage, VisualApp

RECOS_U2I: tp.Dict[tp.Hashable, pd.DataFrame] = {
    "model1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
    "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
}

RECOS_I2I: tp.Dict[tp.Hashable, pd.DataFrame] = {
    "model1": pd.DataFrame({Columns.TargetItem: [3, 4], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
    "model2": pd.DataFrame({Columns.TargetItem: [3, 4], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
}

ITEM_DATA = pd.DataFrame({Columns.Item: [3, 4, 5, 6, 7, 8], "feature_1": ["one", "two", "three", "five", "one", "two"]})
INTERACTIONS = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
REQUESTS_DICT_U2I: tp.Dict[tp.Hashable, tp.Hashable] = {"user_one": 1}
REUQESTS_DICT_I2I: tp.Dict[tp.Hashable, tp.Hashable] = {"item_three": 3}


class TestAppDataStorage:
    def test_u2i(self) -> None:

        ads = AppDataStorage(
            recos=RECOS_U2I,
            item_data=ITEM_DATA,
            interactions=INTERACTIONS,
            is_u2i=True,
            requests_dict=REQUESTS_DICT_U2I,
        )

        assert ads.is_u2i
        assert ads.request_colname == Columns.User
        assert ads.requests_dict == REQUESTS_DICT_U2I
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

    def test_i2i(self) -> None:
        ads = AppDataStorage(recos=RECOS_I2I, item_data=ITEM_DATA, is_u2i=False, requests_dict=REUQESTS_DICT_I2I)

        assert not ads.is_u2i
        assert ads.request_colname == Columns.TargetItem
        assert ads.requests_dict == REUQESTS_DICT_I2I
        assert ads.model_names == ["model1", "model2"]
        assert ads.request_names == ["item_three"]

        expected_i2i_interactions = pd.DataFrame({Columns.TargetItem: [3, 4], Columns.Item: [3, 4]})
        pd.testing.assert_frame_equal(expected_i2i_interactions, ads.interactions, check_like=True)

        assert list(ads.processed_interactions.keys()) == ["item_three"]
        expected_interactions = pd.DataFrame({Columns.Item: [3], "feature_1": ["one"]})
        pd.testing.assert_frame_equal(ads.processed_interactions["item_three"], expected_interactions)

        expected_processed_recos = {
            "model1": {"item_three": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.99]})},
            "model2": {"item_three": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]})},
        }
        assert expected_processed_recos.keys() == ads.processed_recos.keys()
        for model_name, model_recos in expected_processed_recos.items():
            assert model_recos.keys() == ads.processed_recos[model_name].keys()
            for user_name, user_recos in model_recos.items():
                pd.testing.assert_frame_equal(user_recos, ads.processed_recos[model_name][user_name])

    def test_missing_columns_validation(self) -> None:

        # Missing `Columns.User` for u2i
        with pytest.raises(KeyError):
            incorrect_u2i_recos: tp.Dict[tp.Hashable, pd.DataFrame] = {
                "model1": pd.DataFrame({Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
                "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
            }
            AppDataStorage(
                recos=incorrect_u2i_recos,
                item_data=ITEM_DATA,
                is_u2i=True,
                requests_dict=REQUESTS_DICT_U2I,
                interactions=INTERACTIONS,
            )

        # Missing `Columns.Item`
        with pytest.raises(KeyError):
            incorrect_u2i_recos = {
                "model1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
                "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Rank: [1, 1]}),
            }
            AppDataStorage(
                recos=incorrect_u2i_recos,
                item_data=ITEM_DATA,
                is_u2i=True,
                requests_dict=REQUESTS_DICT_U2I,
                interactions=INTERACTIONS,
            )

        # Missing `Columns.TargetItem` for i2i
        with pytest.raises(KeyError):
            AppDataStorage(recos=RECOS_U2I, item_data=ITEM_DATA, is_u2i=False, requests_dict=REUQESTS_DICT_I2I)

        # Missing `Columns.Item` in item_data
        with pytest.raises(KeyError):
            AppDataStorage(
                recos=RECOS_U2I,
                item_data=ITEM_DATA.drop(columns=[Columns.Item]),
                interactions=INTERACTIONS,
                is_u2i=True,
                requests_dict=REQUESTS_DICT_U2I,
            )

    def test_incorrect_interactions_for_reco_case(self) -> None:

        # u2i without interactions
        with pytest.raises(ValueError):
            AppDataStorage(recos=RECOS_U2I, item_data=ITEM_DATA, is_u2i=True, requests_dict=REQUESTS_DICT_U2I)

        # i2i with interactions
        with pytest.raises(ValueError):
            AppDataStorage(
                recos=RECOS_I2I,
                item_data=ITEM_DATA,
                is_u2i=False,
                requests_dict=REUQESTS_DICT_I2I,
                interactions=INTERACTIONS,
            )

    def test_empty_requests(self) -> None:
        with pytest.raises(ValueError):
            AppDataStorage(
                recos=RECOS_U2I,
                item_data=ITEM_DATA,
                interactions=INTERACTIONS,
                is_u2i=True,
                requests_dict={},
            )


class TestVisualApp:
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize("formatters", (None, {"feature_1": lambda x: f"<b>{x}</b>"}))
    def test_happy_path_u2i(self, auto_display: bool, formatters: tp.Optional[tp.Dict[str, tp.Callable]]) -> None:
        VisualApp(
            recos=RECOS_U2I,
            item_data=ITEM_DATA,
            requests_dict=REQUESTS_DICT_U2I,
            is_u2i=True,
            interactions=INTERACTIONS,
            auto_display=auto_display,
            formatters=formatters,
        )

    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize("formatters", (None, {"feature_1": lambda x: f"<b>{x}</b>"}))
    def test_happy_path_i2i(self, auto_display: bool, formatters: tp.Optional[tp.Dict[str, tp.Callable]]) -> None:
        VisualApp(
            recos=RECOS_I2I,
            item_data=ITEM_DATA,
            requests_dict=REUQESTS_DICT_I2I,
            is_u2i=False,
            auto_display=auto_display,
            formatters=formatters,
        )
