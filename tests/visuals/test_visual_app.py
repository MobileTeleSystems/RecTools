import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.visuals.visual_app import ItemToItemVisualApp, TablesDict, VisualApp, _AppDataStorage

RECOS_U2I: TablesDict = {
    "model1": pd.DataFrame(
        {Columns.User: [1, 2, 3, 4], Columns.Item: [3, 4, 3, 4], Columns.Score: [0.99, 0.9, 0.5, 0.5]}
    ),
    "model2": pd.DataFrame({Columns.User: [1, 2, 3, 4], Columns.Item: [5, 6, 5, 6], Columns.Rank: [1, 1, 1, 1]}),
}

RECOS_I2I: TablesDict = {
    "model1": pd.DataFrame(
        {Columns.TargetItem: [3, 4, 5, 5], Columns.Item: [3, 4, 7, 8], Columns.Score: [0.99, 0.9, 0.7, 0.5]}
    ),
    "model2": pd.DataFrame({Columns.TargetItem: [3, 4], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
}

ITEM_DATA = pd.DataFrame({Columns.Item: [3, 4, 5, 6, 7, 8], "feature_1": ["one", "two", "three", "five", "one", "two"]})
INTERACTIONS = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
SELECTED_REQUESTS_U2I: tp.Dict[tp.Hashable, tp.Hashable] = {"user_one": 1}
SELECTED_REQUESTS_I2I: tp.Dict[tp.Hashable, tp.Hashable] = {"item_three": 3}


class TestAppDataStorage:
    def test_u2i(self) -> None:

        ads = _AppDataStorage(
            recos=RECOS_U2I,
            item_data=ITEM_DATA,
            interactions=INTERACTIONS,
            is_u2i=True,
            selected_requests=SELECTED_REQUESTS_U2I,
        )

        assert ads.is_u2i
        assert ads.id_col == Columns.User
        assert ads.selected_requests == SELECTED_REQUESTS_U2I
        assert ads.model_names == ["model1", "model2"]
        assert ads.request_names == ["user_one"]

        assert list(ads.grouped_interactions.keys()) == ["user_one"]
        expected_interactions = pd.DataFrame({Columns.Item: [3, 7], "feature_1": ["one", "one"]})
        pd.testing.assert_frame_equal(ads.grouped_interactions["user_one"], expected_interactions)

        expected_grouped_recos = {
            "model1": {"user_one": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.99]})},
            "model2": {"user_one": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]})},
        }
        assert expected_grouped_recos.keys() == ads.grouped_recos.keys()
        for model_name, model_recos in expected_grouped_recos.items():
            assert model_recos.keys() == ads.grouped_recos[model_name].keys()
            for user_name, user_recos in model_recos.items():
                pd.testing.assert_frame_equal(user_recos, ads.grouped_recos[model_name][user_name])

    def test_i2i(self) -> None:
        ads = _AppDataStorage(
            recos=RECOS_I2I, item_data=ITEM_DATA, is_u2i=False, selected_requests=SELECTED_REQUESTS_I2I
        )

        assert not ads.is_u2i
        assert ads.id_col == Columns.TargetItem
        assert ads.selected_requests == SELECTED_REQUESTS_I2I
        assert ads.model_names == ["model1", "model2"]
        assert ads.request_names == ["item_three"]

        expected_i2i_interactions = pd.DataFrame({Columns.TargetItem: [3, 4, 5], Columns.Item: [3, 4, 5]})
        pd.testing.assert_frame_equal(expected_i2i_interactions, ads.interactions, check_like=True)

        assert list(ads.grouped_interactions.keys()) == ["item_three"]
        expected_interactions = pd.DataFrame({Columns.Item: [3], "feature_1": ["one"]})
        pd.testing.assert_frame_equal(ads.grouped_interactions["item_three"], expected_interactions)

        expected_grouped_recos = {
            "model1": {"item_three": pd.DataFrame({Columns.Item: [3], "feature_1": ["one"], Columns.Score: [0.99]})},
            "model2": {"item_three": pd.DataFrame({Columns.Item: [5], "feature_1": ["three"], Columns.Rank: [1]})},
        }
        assert expected_grouped_recos.keys() == ads.grouped_recos.keys()
        for model_name, model_recos in expected_grouped_recos.items():
            assert model_recos.keys() == ads.grouped_recos[model_name].keys()
            for user_name, user_recos in model_recos.items():
                pd.testing.assert_frame_equal(user_recos, ads.grouped_recos[model_name][user_name])

    def test_missing_columns_validation(self) -> None:

        # Missing `Columns.User` for u2i
        with pytest.raises(KeyError):
            incorrect_u2i_recos: TablesDict = {
                "model1": pd.DataFrame({Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
                "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]}),
            }
            _AppDataStorage(
                recos=incorrect_u2i_recos,
                item_data=ITEM_DATA,
                is_u2i=True,
                selected_requests=SELECTED_REQUESTS_U2I,
                interactions=INTERACTIONS,
            )

        # Missing `Columns.Item`
        with pytest.raises(KeyError):
            incorrect_u2i_recos = {
                "model1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
                "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Rank: [1, 1]}),
            }
            _AppDataStorage(
                recos=incorrect_u2i_recos,
                item_data=ITEM_DATA,
                is_u2i=True,
                selected_requests=SELECTED_REQUESTS_U2I,
                interactions=INTERACTIONS,
            )

        # Missing `Columns.TargetItem` for i2i
        with pytest.raises(KeyError):
            _AppDataStorage(recos=RECOS_U2I, item_data=ITEM_DATA, is_u2i=False, selected_requests=SELECTED_REQUESTS_I2I)

        # Missing `Columns.Item` in item_data
        with pytest.raises(KeyError):
            _AppDataStorage(
                recos=RECOS_U2I,
                item_data=ITEM_DATA.drop(columns=[Columns.Item]),
                interactions=INTERACTIONS,
                is_u2i=True,
                selected_requests=SELECTED_REQUESTS_U2I,
            )

    def test_incorrect_interactions_for_reco_case(self) -> None:

        # u2i without interactions
        with pytest.raises(ValueError):
            _AppDataStorage(recos=RECOS_U2I, item_data=ITEM_DATA, is_u2i=True, selected_requests=SELECTED_REQUESTS_U2I)

        # i2i with interactions
        with pytest.raises(ValueError):
            _AppDataStorage(
                recos=RECOS_I2I,
                item_data=ITEM_DATA,
                is_u2i=False,
                selected_requests=SELECTED_REQUESTS_I2I,
                interactions=INTERACTIONS,
            )

    def test_empty_requests(self) -> None:
        with pytest.raises(ValueError):
            _AppDataStorage(
                recos=RECOS_U2I,
                item_data=ITEM_DATA,
                interactions=INTERACTIONS,
                is_u2i=True,
                selected_requests={},
            )

    @pytest.mark.parametrize("n_random_requests", (1, 5))
    def test_u2i_with_random_requests(self, n_random_requests: int) -> None:
        ads = _AppDataStorage(
            recos=RECOS_U2I,
            item_data=ITEM_DATA,
            interactions=INTERACTIONS,
            is_u2i=True,
            selected_requests=SELECTED_REQUESTS_U2I,
            n_random_requests=n_random_requests,
        )
        assert "user_one" in ads.request_names
        corrected_n_random_requests = min(n_random_requests, 3)  # only 3 users in recos can be selected from

        for i in range(1, corrected_n_random_requests + 1):
            random_name = f"random_{i}"
            random_id = ads.selected_requests[random_name]
            assert random_name in ads.request_names
            assert random_id != 1  # random id is not same as predefined by user
            total_recos = 0
            for model_name in ["model1", "model2"]:
                expected_recos = RECOS_U2I[model_name].query(f"{Columns.User} == @random_id")["item_id"].sort_values()
                actual_recos = ads.grouped_recos[model_name][random_name]["item_id"].sort_values()
                total_recos += expected_recos.shape[0]
                assert np.array_equal(actual_recos, expected_recos)
            assert total_recos > 0  # random user has recos at least from one model

        # correct names in selected_requests
        all_selected_names = set(ads.selected_requests.keys())
        assert all_selected_names == set(
            ["user_one"] + [f"random_{i}" for i in range(1, corrected_n_random_requests + 1)]
        )

        # random ids don't have duplicates
        assert len(ads.selected_requests.values()) == len(set(ads.selected_requests.values()))

    @pytest.mark.parametrize("n_random_requests", (2, 5))
    def test_i2i_with_random_requests(self, n_random_requests: int) -> None:
        ads = _AppDataStorage(
            recos=RECOS_I2I,
            item_data=ITEM_DATA,
            is_u2i=False,
            selected_requests=SELECTED_REQUESTS_I2I,
            n_random_requests=n_random_requests,
        )
        assert "item_three" in ads.request_names
        corrected_n_random_requests = min(n_random_requests, 2)  # only 2 target items in recos can be selected from

        for i in range(1, corrected_n_random_requests + 1):
            random_name = f"random_{i}"
            random_id = ads.selected_requests[random_name]
            assert random_name in ads.request_names
            assert random_id != 3  # random id is not same as predefined by user
            total_recos = 0
            for model_name in ["model1", "model2"]:
                expected_recos = (
                    RECOS_I2I[model_name].query(f"{Columns.TargetItem} == @random_id")["item_id"].sort_values()
                )
                actual_recos = ads.grouped_recos[model_name][random_name]["item_id"].sort_values()
                total_recos += expected_recos.shape[0]
                assert np.array_equal(actual_recos, expected_recos)
            assert total_recos > 0  # random item has recos at least from one model

        # correct names in selected_requests
        all_selected_names = set(ads.selected_requests.keys())
        assert all_selected_names == set(
            ["item_three"] + [f"random_{i}" for i in range(1, corrected_n_random_requests + 1)]
        )

        # random ids don't have duplicates
        assert len(ads.selected_requests.values()) == len(set(ads.selected_requests.values()))


class TestVisualApp:
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize("n_random_users", (0, 2, 100))
    @pytest.mark.parametrize("formatters", (None, {"feature_1": lambda x: f"<b>{x}</b>"}))
    def test_happy_path(
        self, auto_display: bool, n_random_users: int, formatters: tp.Optional[tp.Dict[str, tp.Callable]]
    ) -> None:
        VisualApp(
            recos=RECOS_U2I,
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
                recos=RECOS_U2I,
                item_data=ITEM_DATA,
                selected_users=SELECTED_REQUESTS_U2I,
                interactions=INTERACTIONS,
                auto_display=True,
                n_random_users=0,
                min_width=5,
            )


class TestItemToItemVisualApp:
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize("n_random_items", (0, 2, 100))
    @pytest.mark.parametrize("formatters", (None, {"feature_1": lambda x: f"<b>{x}</b>"}))
    def test_happy_path(
        self, auto_display: bool, n_random_items: int, formatters: tp.Optional[tp.Dict[str, tp.Callable]]
    ) -> None:
        ItemToItemVisualApp(
            recos=RECOS_I2I,
            item_data=ITEM_DATA,
            selected_items=SELECTED_REQUESTS_I2I,
            auto_display=auto_display,
            formatters=formatters,
            n_random_items=n_random_items,
        )

    def test_incorrect_min_width(self) -> None:
        with pytest.raises(ValueError):
            ItemToItemVisualApp(
                recos=RECOS_I2I,
                item_data=ITEM_DATA,
                selected_items=SELECTED_REQUESTS_I2I,
                auto_display=True,
                n_random_items=0,
                min_width=-10,
            )
