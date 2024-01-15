# pylint: disable=too-many-branches

import typing as tp

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from rectools import Columns


class AppDataStorage:
    """
    Helper class to hold all data for showcase purposes.
    - Holds info about interactions, recommendations and item_data
    - Holds `requests_dict`
    """

    def __init__(  # noqa: C901
        self,
        recos: tp.Dict[tp.Hashable, pd.DataFrame],
        item_data: pd.DataFrame,
        requests_dict: tp.Dict[tp.Hashable, tp.Hashable],
        is_u2i: bool = True,
        interactions: tp.Optional[pd.DataFrame] = None,
    ) -> None:
        self.request_colname = Columns.User if is_u2i else Columns.TargetItem
        self.is_u2i = is_u2i
        self.requests_dict = requests_dict
        self.model_names = list(recos.keys())
        self.request_names = list(requests_dict.keys())

        self._check_columns_present_in_recos(recos)
        self.recos = recos

        if Columns.Item not in item_data:
            raise KeyError(f"Missed {Columns.Item} column in item_data")
        self.item_data = item_data

        if interactions is None:
            if is_u2i:
                raise ValueError("For u2i reco showcase you must specify interactions")
            interactions = self._prepare_interactions_for_i2i()
        self.interactions = interactions

        self.processed_interactions = self._process_interactions()
        self.processed_recos = self._process_recos()

    def _process_interactions(self) -> tp.Dict[tp.Hashable, pd.DataFrame]:
        prepared_interactions = {}
        for request_name, request_id in self.requests_dict.items():
            prepared_interactions[request_name] = (
                self.interactions[self.interactions[self.request_colname] == request_id]
                .merge(self.item_data, how="left", on=Columns.Item)
                .drop(columns=[self.request_colname])
            )
        return prepared_interactions

    def _process_recos(self) -> tp.Dict[tp.Hashable, tp.Dict[tp.Hashable, pd.DataFrame]]:
        prepared_recos = {}
        for model_name, full_recos in self.recos.items():
            model_recos = {}
            for request_name, request_id in self.requests_dict.items():
                model_recos[request_name] = (
                    full_recos[full_recos[self.request_colname] == request_id]
                    .merge(self.item_data, how="left", on=Columns.Item)
                    .drop(columns=[self.request_colname])
                )
            prepared_recos[model_name] = model_recos
        return prepared_recos

    def _check_columns_present_in_recos(self, recos: tp.Dict[tp.Hashable, pd.DataFrame]) -> None:
        required_columns = {Columns.User, Columns.Item} if self.is_u2i else {Columns.TargetItem, Columns.Item}
        for model_name, model_recos in recos.items():
            actual_columns = set(model_recos.columns)
            if not actual_columns >= required_columns:
                raise KeyError(f"Missed columns {required_columns - actual_columns} in {model_name} recommendations df")

    def _prepare_interactions_for_i2i(self) -> pd.DataFrame:
        request_ids = set()
        for recos_df in self.recos.values():
            request_ids.update(set(recos_df[Columns.TargetItem].unique()))
        interactions = pd.DataFrame({Columns.TargetItem: list(request_ids), Columns.Item: list(request_ids)})
        return interactions


class VisualApp:
    """
    Main class for recommendations visualization.
    - Provides visual information about requests in `requests_dict` and recos
    """

    def __init__(
        self,
        recos: tp.Dict[tp.Hashable, pd.DataFrame],
        item_data: pd.DataFrame,
        requests_dict: tp.Dict[tp.Hashable, tp.Hashable],
        is_u2i: bool = True,
        interactions: tp.Optional[pd.DataFrame] = None,
        auto_display: bool = True,
        item_df_formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
    ) -> None:
        self.data_storage = AppDataStorage(
            interactions=interactions,
            recos=recos,
            requests_dict=requests_dict,
            item_data=item_data,
            is_u2i=is_u2i,
        )
        self.rows_limit = rows_limit
        self.item_data_formatters = item_df_formatters if item_df_formatters is not None else {}
        if auto_display:
            self.display()

    def _convert_to_html(self, df: pd.DataFrame) -> str:
        html_repr = (
            df.to_html(
                escape=False,
                index=False,
                formatters=self.item_data_formatters,
                max_rows=self.rows_limit,
                border=0,
            )
            .replace("<td>", '<td align="center">')
            .replace("<th>", '<th style="text-align: center; min-width: 100px;">')
        )
        return html_repr

    def _display_interactions(self, request_name: str) -> None:
        """Display viewed items for `request_name`"""
        items_tab = widgets.Tab()
        df = self.data_storage.processed_interactions[request_name]
        items_tab.children = [widgets.HTML(value=self._convert_to_html(df))]
        items_tab.set_title(index=0, title="Interactions")
        display(items_tab)

    def _display_recos(self, request_name: str, model_name: str) -> None:
        """Display recommended items for `request_name` from model `model_name`"""
        items_tab = widgets.Tab()
        df = self.data_storage.processed_recos[model_name][request_name]
        items_tab.children = [widgets.HTML(value=self._convert_to_html(df))]
        items_tab.set_title(index=0, title="Recos")
        display(items_tab)

    def _display_request_id(self, request_name: str) -> None:
        """Display request_id for `request_name`"""
        request_id = self.data_storage.requests_dict[request_name]
        display(widgets.HTML(value=f"{self.data_storage.request_colname}: {request_id}"))

    def _display_model_name(self, model_name: str) -> None:
        """Display model_name"""
        display(widgets.HTML(value=f"Model name: {model_name}"))

    def display(self) -> None:
        """Display full VisualApp widget"""
        request_name_selection = widgets.ToggleButtons(
            options=self.data_storage.request_names,
            description=f"Request {self.data_storage.request_colname}:",
            disabled=False,
            button_style="warning",
        )
        request_id_output = widgets.interactive_output(
            self._display_request_id, {"request_name": request_name_selection}
        )
        interactions_output = widgets.interactive_output(
            self._display_interactions, {"request_name": request_name_selection}
        )
        model_selection = widgets.ToggleButtons(
            options=self.data_storage.model_names,
            description="Model:",
            disabled=False,
            button_style="success",
        )
        model_name_output = widgets.interactive_output(self._display_model_name, {"model_name": model_selection})
        recos_output = widgets.interactive_output(
            self._display_recos, {"request_name": request_name_selection, "model_name": model_selection}
        )

        display(
            widgets.VBox(
                [
                    request_name_selection,
                    request_id_output,
                    interactions_output,
                    model_selection,
                    model_name_output,
                    recos_output,
                ]
            )
        )
