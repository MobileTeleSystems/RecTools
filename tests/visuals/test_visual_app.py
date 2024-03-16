import tempfile
import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.visuals.visual_app import AppDataStorage, ItemToItemVisualApp, TablesDict, VisualApp, VisualAppBase

RECO_U2I: TablesDict = {
    "model1": pd.DataFrame(
        {Columns.User: [1, 2, 3, 4], Columns.Item: [3, 4, 3, 4], Columns.Score: [0.99, 0.9, 0.5, 0.5]}
    ),
    "model2": pd.DataFrame({Columns.User: [1, 2, 3, 4], Columns.Item: [5, 6, 5, 6], Columns.Rank: [1, 1, 1, 1]}),
}

RECO_I2I: TablesDict = {
    "model1": pd.DataFrame(
        {Columns.TargetItem: [3, 4, 5, 5], Columns.Item: [3, 4, 7, 8], Columns.Score: [0.99, 0.9, 0.7, 0.5]}
    ),
    "model2": pd.DataFrame({Columns.TargetItem: [3, 4], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
}

ITEM_DATA = pd.DataFrame({Columns.Item: [3, 4, 5, 6, 7, 8], "feature_1": ["one", "two", "three", "five", "one", "two"]})
INTERACTIONS = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
SELECTED_REQUESTS_U2I: tp.Dict[tp.Hashable, tp.Hashable] = {"user_one": 1, "user_three": 3}
SELECTED_REQUESTS_I2I: tp.Dict[tp.Hashable, tp.Hashable] = {"item_three": 3}


def check_data_storages_equal(one: AppDataStorage, two: AppDataStorage) -> None:
    assert one.id_col == two.id_col
    assert one.is_u2i == two.is_u2i
    assert one.model_names == two.model_names
    assert one.request_names == two.request_names
    assert one.selected_requests == two.selected_requests

    assert one.grouped_interactions.keys() == two.grouped_interactions.keys()
    for model_name, model_df in one.grouped_interactions.items():
        pd.testing.assert_frame_equal(model_df, two.grouped_interactions[model_name])

    assert one.grouped_reco.keys() == two.grouped_reco.keys()
    for model_name, model_reco_dict in one.grouped_reco.items():
        assert model_reco_dict.keys() == two.grouped_reco[model_name].keys()
        for request_name, request_df in model_reco_dict.items():
            pd.testing.assert_frame_equal(request_df, two.grouped_reco[model_name][request_name], check_dtype=False)


class TestAppDataStorage:
    def test_u2i(self) -> None:

        ads = AppDataStorage.from_raw(
            reco=RECO_U2I,
            item_data=ITEM_DATA,
            interactions=INTERACTIONS,
            is_u2i=True,
            selected_requests=SELECTED_REQUESTS_U2I,
        )

        assert ads.is_u2i
        assert ads.id_col == Columns.User
        assert ads.selected_requests == SELECTED_REQUESTS_U2I
        assert ads.model_names == ["model1", "model2"]
        assert ads.request_names == ["user_one", "user_three"]

        assert list(ads.grouped_interactions.keys()) == ["user_one", "user_three"]
        expected_interactions = pd.DataFrame({Columns.Item: [3, 7], "feature_1": ["one", "one"]})
        pd.testing.assert_frame_equal(ads.grouped_interactions["user_one"], expected_interactions)

        expected_grouped_reco = {
            "model1": {
                "user_one": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.99]}),
                "user_three": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.5]}),
            },
            "model2": {
                "user_one": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]}),
                "user_three": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]}),
            },
        }
        assert expected_grouped_reco.keys() == ads.grouped_reco.keys()
        for model_name, model_reco in expected_grouped_reco.items():
            assert model_reco.keys() == ads.grouped_reco[model_name].keys()
            for user_name, user_reco in model_reco.items():
                pd.testing.assert_frame_equal(user_reco, ads.grouped_reco[model_name][user_name])

    def test_i2i(self) -> None:
        ads = AppDataStorage.from_raw(
            reco=RECO_I2I, item_data=ITEM_DATA, is_u2i=False, selected_requests=SELECTED_REQUESTS_I2I
        )

        assert not ads.is_u2i
        assert ads.id_col == Columns.TargetItem
        assert ads.selected_requests == SELECTED_REQUESTS_I2I
        assert ads.model_names == ["model1", "model2"]
        assert ads.request_names == ["item_three"]

        assert list(ads.grouped_interactions.keys()) == ["item_three"]
        expected_interactions = pd.DataFrame({Columns.Item: [3], "feature_1": ["one"]})
        pd.testing.assert_frame_equal(ads.grouped_interactions["item_three"], expected_interactions)

        expected_grouped_reco = {
            "model1": {"item_three": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.99]})},
            "model2": {"item_three": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]})},
        }
        assert expected_grouped_reco.keys() == ads.grouped_reco.keys()
        for model_name, model_reco in expected_grouped_reco.items():
            assert model_reco.keys() == ads.grouped_reco[model_name].keys()
            for user_name, user_reco in model_reco.items():
                pd.testing.assert_frame_equal(user_reco, ads.grouped_reco[model_name][user_name])

    def test_i2i_interactions(self) -> None:
        expected_i2i_interactions = pd.DataFrame({Columns.TargetItem: [3, 4, 5], Columns.Item: [3, 4, 5]})
        actual = AppDataStorage._prepare_interactions_for_i2i(reco=RECO_I2I)  # pylint: disable=protected-access
        pd.testing.assert_frame_equal(expected_i2i_interactions, actual, check_like=True)

    def test_missing_columns_validation(self) -> None:

        # Missing `Columns.User` for u2i
        with pytest.raises(KeyError):
            incorrect_u2i_reco: TablesDict = {
                "model1": pd.DataFrame({Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
                "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
            }
            AppDataStorage.from_raw(
                reco=incorrect_u2i_reco,
                item_data=ITEM_DATA,
                is_u2i=True,
                selected_requests=SELECTED_REQUESTS_U2I,
                interactions=INTERACTIONS,
            )

        # Missing `Columns.Item`
        with pytest.raises(KeyError):
            incorrect_u2i_reco = {
                "model1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
                "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Rank: [1, 1]}),
            }
            AppDataStorage.from_raw(
                reco=incorrect_u2i_reco,
                item_data=ITEM_DATA,
                is_u2i=True,
                selected_requests=SELECTED_REQUESTS_U2I,
                interactions=INTERACTIONS,
            )

        # Missing `Columns.TargetItem` for i2i
        with pytest.raises(KeyError):
            AppDataStorage.from_raw(
                reco=RECO_U2I, item_data=ITEM_DATA, is_u2i=False, selected_requests=SELECTED_REQUESTS_I2I
            )

        # Missing `Columns.Item` in item_data
        with pytest.raises(KeyError):
            AppDataStorage.from_raw(
                reco=RECO_U2I,
                item_data=ITEM_DATA.drop(columns=[Columns.Item]),
                interactions=INTERACTIONS,
                is_u2i=True,
                selected_requests=SELECTED_REQUESTS_U2I,
            )

    def test_incorrect_interactions_for_reco_case(self) -> None:

        # u2i without interactions
        with pytest.raises(ValueError):
            AppDataStorage.from_raw(
                reco=RECO_U2I, item_data=ITEM_DATA, is_u2i=True, selected_requests=SELECTED_REQUESTS_U2I
            )

        # i2i with interactions
        with pytest.raises(ValueError):
            AppDataStorage.from_raw(
                reco=RECO_I2I,
                item_data=ITEM_DATA,
                is_u2i=False,
                selected_requests=SELECTED_REQUESTS_I2I,
                interactions=INTERACTIONS,
            )

    def test_empty_requests(self) -> None:
        with pytest.raises(ValueError):
            AppDataStorage.from_raw(
                reco=RECO_U2I,
                item_data=ITEM_DATA,
                interactions=INTERACTIONS,
                is_u2i=True,
                selected_requests={},
            )

    @pytest.mark.parametrize("n_random_requests", (1, 5))
    def test_u2i_with_random_requests(self, n_random_requests: int) -> None:
        ads = AppDataStorage.from_raw(
            reco=RECO_U2I,
            item_data=ITEM_DATA,
            interactions=INTERACTIONS,
            is_u2i=True,
            selected_requests=SELECTED_REQUESTS_U2I,
            n_random_requests=n_random_requests,
        )
        assert "user_one" in ads.request_names
        corrected_n_random_requests = min(n_random_requests, 2)  # only 2 users in reco can be selected from

        for i in range(1, corrected_n_random_requests + 1):
            random_name = f"random_{i}"
            random_id = ads.selected_requests[random_name]
            assert random_name in ads.request_names
            assert random_id != 1  # random id is not same as predefined by user
            total_reco = 0
            for model_name in ["model1", "model2"]:
                expected_reco = RECO_U2I[model_name].query(f"{Columns.User} == @random_id")["item_id"].sort_values()
                actual_reco = ads.grouped_reco[model_name][random_name]["item_id"].sort_values()
                total_reco += expected_reco.shape[0]
                assert np.array_equal(actual_reco, expected_reco)
            assert total_reco > 0  # random user has reco at least from one model

        # correct names in selected_requests
        all_selected_names = set(ads.selected_requests.keys())
        assert all_selected_names == set(
            ["user_one", "user_three"] + [f"random_{i}" for i in range(1, corrected_n_random_requests + 1)]
        )

        # random ids don't have duplicates
        assert len(ads.selected_requests.values()) == len(set(ads.selected_requests.values()))

    @pytest.mark.parametrize("n_random_requests", (2, 5))
    def test_i2i_with_random_requests(self, n_random_requests: int) -> None:
        ads = AppDataStorage.from_raw(
            reco=RECO_I2I,
            item_data=ITEM_DATA,
            is_u2i=False,
            selected_requests=SELECTED_REQUESTS_I2I,
            n_random_requests=n_random_requests,
        )
        assert "item_three" in ads.request_names
        corrected_n_random_requests = min(n_random_requests, 2)  # only 2 target items in reco can be selected from

        for i in range(1, corrected_n_random_requests + 1):
            random_name = f"random_{i}"
            random_id = ads.selected_requests[random_name]
            assert random_name in ads.request_names
            assert random_id != 3  # random id is not same as predefined by user
            total_reco = 0
            for model_name in ["model1", "model2"]:
                expected_reco = (
                    RECO_I2I[model_name].query(f"{Columns.TargetItem} == @random_id")["item_id"].sort_values()
                )
                actual_reco = ads.grouped_reco[model_name][random_name]["item_id"].sort_values()
                total_reco += expected_reco.shape[0]
                assert np.array_equal(actual_reco, expected_reco)
            assert total_reco > 0  # random item has reco at least from one model

        # correct names in selected_requests
        all_selected_names = set(ads.selected_requests.keys())
        assert all_selected_names == set(
            ["item_three"] + [f"random_{i}" for i in range(1, corrected_n_random_requests + 1)]
        )

        # random ids don't have duplicates
        assert len(ads.selected_requests.values()) == len(set(ads.selected_requests.values()))

    def test_save_and_load_equal(self) -> None:
        ads_u2i = AppDataStorage.from_raw(
            reco=RECO_U2I,
            item_data=ITEM_DATA,
            interactions=INTERACTIONS,
            is_u2i=True,
            selected_requests=SELECTED_REQUESTS_U2I,
        )
        ads_i2i = AppDataStorage.from_raw(
            reco=RECO_I2I,
            item_data=ITEM_DATA,
            is_u2i=False,
            selected_requests=SELECTED_REQUESTS_I2I,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ads_u2i.save(tmp)
            loaded_u2i = AppDataStorage.load(tmp)
            check_data_storages_equal(ads_u2i, loaded_u2i)

            with pytest.raises(FileExistsError):
                ads_i2i.save(tmp)

            ads_i2i.save(tmp, overwrite=True)
            loaded_i2i = AppDataStorage.load(tmp)
            check_data_storages_equal(ads_i2i, loaded_i2i)


class TestVisualApp:
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize("n_random_users", (0, 2, 100))
    @pytest.mark.parametrize("formatters", (None, {"feature_1": lambda x: f"<b>{x}</b>"}))
    def test_happy_path(
        self, auto_display: bool, n_random_users: int, formatters: tp.Optional[tp.Dict[str, tp.Callable]]
    ) -> None:
        VisualApp(
            reco=RECO_U2I,
            item_data=ITEM_DATA,
            selected_users=SELECTED_REQUESTS_U2I,
            interactions=INTERACTIONS,
            auto_display=auto_display,
            formatters=formatters,
            n_random_users=n_random_users,
        )

    def test_incorrect_min_width(self) -> None:
        with pytest.raises(ValueError):
            VisualApp(
                reco=RECO_U2I,
                item_data=ITEM_DATA,
                selected_users=SELECTED_REQUESTS_U2I,
                interactions=INTERACTIONS,
                auto_display=True,
                n_random_users=0,
                min_width=5,
            )

    def test_save_and_load_equal_data_storage(self) -> None:
        app = VisualApp(
            reco=RECO_U2I,
            item_data=ITEM_DATA,
            selected_users=SELECTED_REQUESTS_U2I,
            interactions=INTERACTIONS,
            auto_display=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            app.save(tmp)

            item_to_item_loaded = ItemToItemVisualApp.load(tmp, auto_display=False)
            check_data_storages_equal(app.data_storage, item_to_item_loaded.data_storage)

            base_loaded = VisualAppBase.load(tmp, auto_display=False)
            check_data_storages_equal(app.data_storage, base_loaded.data_storage)

            visual_app_loaded = VisualAppBase.load(tmp, auto_display=False)
            check_data_storages_equal(app.data_storage, visual_app_loaded.data_storage)


class TestItemToItemVisualApp:
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize("n_random_items", (0, 2, 100))
    @pytest.mark.parametrize("formatters", (None, {"feature_1": lambda x: f"<b>{x}</b>"}))
    def test_happy_path(
        self, auto_display: bool, n_random_items: int, formatters: tp.Optional[tp.Dict[str, tp.Callable]]
    ) -> None:
        ItemToItemVisualApp(
            reco=RECO_I2I,
            item_data=ITEM_DATA,
            selected_items=SELECTED_REQUESTS_I2I,
            auto_display=auto_display,
            formatters=formatters,
            n_random_items=n_random_items,
        )

    def test_incorrect_min_width(self) -> None:
        with pytest.raises(ValueError):
            ItemToItemVisualApp(
                reco=RECO_I2I,
                item_data=ITEM_DATA,
                selected_items=SELECTED_REQUESTS_I2I,
                auto_display=True,
                n_random_items=0,
                min_width=-10,
            )

    def test_save_and_load_equal_data_storage(self) -> None:
        app = ItemToItemVisualApp(
            reco=RECO_I2I,
            item_data=ITEM_DATA,
            selected_items=SELECTED_REQUESTS_I2I,
            auto_display=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            app.save(tmp)

            item_to_item_loaded = ItemToItemVisualApp.load(tmp, auto_display=False)
            check_data_storages_equal(app.data_storage, item_to_item_loaded.data_storage)

            base_loaded = VisualAppBase.load(tmp, auto_display=False)
            check_data_storages_equal(app.data_storage, base_loaded.data_storage)

            visual_app_loaded = VisualAppBase.load(tmp, auto_display=False)
            check_data_storages_equal(app.data_storage, visual_app_loaded.data_storage)
