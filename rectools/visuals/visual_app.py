import typing as tp

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from rectools import Columns


class AppDataStorage:
    """
    Storage and processing of data for `VisualApp` widgets. This class is not meant to be used
    directly. Use `VisualApp` class instead
    """

    def __init__(
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

        if interactions is None and is_u2i:
            raise ValueError("For u2i reco you must specify interactions")
        if interactions is not None and not is_u2i:
            raise ValueError("For i2i reco you shouldn't specify interactions")
        if not is_u2i:
            interactions = self._prepare_interactions_for_i2i()
        self.interactions: pd.DataFrame = interactions

        if not self.requests_dict:
            raise ValueError("`requests_dict` is empty")
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
        item_data_cols = self.item_data.columns.to_list()
        for model_name, full_recos in self.recos.items():
            model_recos = {}
            for request_name, request_id in self.requests_dict.items():
                df = (
                    full_recos[full_recos[self.request_colname] == request_id]
                    .merge(self.item_data, how="left", on=Columns.Item)
                    .drop(columns=[self.request_colname])
                )

                # Change ordering of columns: all item_data cols will go first
                first_order_mask = np.array([col in item_data_cols for col in df.columns])
                new_order = df.columns[first_order_mask].append(df.columns[~first_order_mask])
                df = df.reindex(columns=new_order)

                model_recos[request_name] = df
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
    r"""
    Main tool for recommendations visualization. Creates Jupyter widgets for visual analysis and
    comparison of recommendations. Outputs both interactions history and recos for all of the
    requests. `VisualApp` supports user-to-item (u2i) and item-to-item (i2i) cases.

    Models for comparison will be listed from the `recos` dictionary keys.
    Users (for u2i case) or items (for i2i case) display names for comparison will be listed from the
    `requests_dict` keys and ids will be taken for values.
    For u2i case please provide `interactions` and set `is_u2i` = True. For i2i case keep
    `interactions` as a default `None` and set `is_u2i` = False.

    Parameters
    ----------
    recos : tp.Dict[tp.Hashable, pd.DataFrame]
        Recommendations from different models in a form of a dict. Model names are supposed to be
        dict keys. Recommendations from models are supposed to be in form of pandas DataFrames with
        columns:
            - `Columns.Item` - recommended item id
            - `Columns.User` - user id (only for u2i)
            - `Columns.TargetItem` - target item id (only for i2i)
            - Any other columns that you wish to display in widgets (e.g. rank or score)
        The original order of the rows will be preserved. Keep in mind to sort the rows correctly
        before visualizing. The most intuitive wy is to sort by rank in ascending order.
    item_data : pd.DataFrame
        Data for items that is used for visualisation in both interactions and recos widgets.
        Supposed to be in form of a pandas DataFrame with columns:
            - `Columns.Item` - item id
            - Any other columns with item data (e.g. name, category, popularity, image link)
    requests_dict : tp.Dict[tp.Hashable, tp.Hashable]
        Predefined requests that will be displayed in widgets. For u2i case specific users are
        considered as requests. For i2i case - specific items. Their names must be specified as keys
        of the dict and ids as values of the dict.
    is_u2i : bool, optional
        User-to-item recommendation case (opposed to item-to-item), by default True
    interactions : tp.Optional[pd.DataFrame], optional
        Table with interactions history for users. Only needed for u2i case. Supposed to be in form
        of pandas DataFrames with columns:
            - `Columns.User` - user id
            - `Columns.Item` - item id
        The original order of the rows will be preserved. Keep in mind to sort the rows correctly
        before visualizing. The most intuitive wy is to sort by date in descending order. If user
        has too many interactions the lest ones may not be displayed.
        By default None.
    auto_display : bool, optional
        Display widgets right after initialization, by default True
    formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional
        Formatter functions to apply to columns elements in the sections of interactions and recos.
        Keys of the dict must be columns names (item_data, interactions and recos columns can be
        specified here). Values bust be functions that will be applied to corresponding columns
        elements. The result of each function must be a unicode string that represents html code.
        Formatters can be used to format text, create links and display images with html.
        By default None.
    rows_limit : int, optional
        Maximum number of rows to display in the sections of interactions and recos, by default 20

    Examples
    --------
    >>> recos = {
    ...     "model1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
    ...     "model2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]})
    ... }
    >>>
    >>> item_data = pd.DataFrame({
    ...     Columns.Item: [3, 4, 5, 6, 7, 8],
    ...     "feature_1": ["one", "two", "three", "five", "one", "two"]
    ... })
    >>>
    >>> interactions = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
    >>> requests_dict = {"user_one": 1}
    >>> formatters = {"item_id": lambda x: f"<b>{x}</b>"}
    >>>
    >>> widgets = VisualApp(
    ...     recos=recos,
    ...     item_data=item_data,
    ...     interactions=interactions,
    ...     requests_dict=requests_dict,
    ...     formatters=formatters,
    ...     auto_display=False
    ... )
    """

    def __init__(
        self,
        recos: tp.Dict[tp.Hashable, pd.DataFrame],
        item_data: pd.DataFrame,
        requests_dict: tp.Dict[tp.Hashable, tp.Hashable],
        is_u2i: bool = True,
        interactions: tp.Optional[pd.DataFrame] = None,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
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
        self.formatters = formatters if formatters is not None else {}
        if auto_display:
            self.display()

    def _convert_to_html(self, df: pd.DataFrame) -> str:
        html_repr = (
            df.to_html(
                escape=False,
                index=False,
                formatters=self.formatters,
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
