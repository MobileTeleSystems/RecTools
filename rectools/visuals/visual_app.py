import typing as tp

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from rectools import Columns

TablesDict = tp.Dict[tp.Hashable, pd.DataFrame]


class _AppDataStorage:
    """
    Storage and processing of data for `VisualApp` widgets. This class is not meant to be used
    directly. Use `VisualApp` or `ItemToItemVisualApp` class instead
    """

    def __init__(
        self,
        recos: TablesDict,
        item_data: pd.DataFrame,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        is_u2i: bool = True,
        interactions: tp.Optional[pd.DataFrame] = None,
    ) -> None:
        self.request_colname = Columns.User if is_u2i else Columns.TargetItem
        self.is_u2i = is_u2i
        self.selected_requests = selected_requests
        self.model_names = list(recos.keys())
        self.request_names = list(selected_requests.keys())

        self._check_columns_present_in_recos(recos)
        self.recos = recos

        if Columns.Item not in item_data:
            raise KeyError(f"Missed {Columns.Item} column in item_data")
        self.item_data = item_data

        if interactions is None and is_u2i:
            raise ValueError("For u2i reco you must specify interactions")
        if interactions is not None and not is_u2i:
            raise ValueError("For i2i reco you must not specify interactions")
        if not is_u2i:
            interactions = self._prepare_interactions_for_i2i()
        self.interactions: pd.DataFrame = interactions

        if not self.selected_requests:
            raise ValueError("`selected_requests` is empty")
        self.processed_interactions = self._process_interactions(
            interactions=self.interactions,
            selected_requests=self.selected_requests,
            request_colname=self.request_colname,
            item_data=self.item_data,
        )
        self.processed_recos = self._process_recos(
            recos=self.recos,
            selected_requests=self.selected_requests,
            request_colname=self.request_colname,
            item_data=self.item_data,
        )

    @classmethod
    def _process_interactions(
        cls,
        interactions: pd.DataFrame,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        request_colname: str,
        item_data: pd.DataFrame,
    ) -> tp.Dict[tp.Hashable, pd.DataFrame]:
        prepared_interactions = {}
        for request_name, request_id in selected_requests.items():
            prepared_interactions[request_name] = (
                interactions[interactions[request_colname] == request_id]
                .merge(item_data, how="left", on=Columns.Item)
                .drop(columns=[request_colname])
            )
        return prepared_interactions

    @classmethod
    def _process_recos(
        cls,
        recos: TablesDict,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        request_colname: str,
        item_data: pd.DataFrame,
    ) -> tp.Dict[tp.Hashable, TablesDict]:
        prepared_recos = {}
        for model_name, model_recos in recos.items():
            prepared_model_recos = {}
            for request_name, request_id in selected_requests.items():
                prepared_model_recos[request_name] = item_data.merge(
                    model_recos[model_recos[request_colname] == request_id],
                    how="right",
                    on="item_id",
                    suffixes=["_item", "_recos"],
                ).drop(columns=[request_colname])
            prepared_recos[model_name] = prepared_model_recos
        return prepared_recos

    def _check_columns_present_in_recos(self, recos: TablesDict) -> None:
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


class VisualAppBase:
    """
    Base visual app class.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
        self,
        *args: tp.Any,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        **kwargs: tp.Any,
    ) -> None:
        self.rows_limit = rows_limit
        self.formatters = formatters if formatters is not None else {}
        self.data_storage: _AppDataStorage = self._create_data_storage(*args, **kwargs)
        if auto_display:
            self.display()

    def _create_data_storage(self, *args: tp.Any, **kwargs: tp.Any) -> _AppDataStorage:
        raise NotImplementedError()

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
        items_tab.set_title(index=0, title="Recommended")
        display(items_tab)

    def _display_request_id(self, request_name: str) -> None:
        """Display request_id for `request_name`"""
        request_id = self.data_storage.selected_requests[request_name]
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


class VisualApp(VisualAppBase):
    r"""
    Main tool for recommendations visualization. Creates Jupyter widgets for visual
    analysis and comparison of different models. Outputs both interactions history of the selected
    users and their recommended items from different models along with items data.

    Models for comparison will be listed from the `recos` dictionary keys.
    Users display names for comparison will be listed from the `selected_users` keys and ids will be
    taken from `selected_users` values.

    Optionally use `formatters` to process dataframe columns values to desired html outputs.

    Parameters
    ----------
    recos : TablesDict
        Recommendations from different models in a form of a dict. Model names are supposed to be
        dict keys. Recommendations from models are supposed to be in form of pandas DataFrames with
        columns:
            - `Columns.User` - user id
            - `Columns.Item` - recommended item id
            - Any other columns that you wish to display in widgets (e.g. rank or score)
        The original order of the rows will be preserved. Keep in mind to sort the rows correctly
        before visualizing. The most intuitive way is to sort by rank in ascending order.
    item_data : pd.DataFrame
        Data for items that is used for visualisation in both interactions and recos widgets.
        Supposed to be in form of a pandas DataFrame with columns:
            - `Columns.Item` - item id
            - Any other columns with item data (e.g. name, category, popularity, image link)
    selected_users : tp.Dict[tp.Hashable, tp.Hashable]
        Predefined users that will be displayed in widgets. User names must be specified as keys
        of the dict and user ids as values of the dict.
    interactions : tp.Optional[pd.DataFrame], optional, default ``None``
        Table with interactions history for users. Only needed for u2i case. Supposed to be in form
        of pandas DataFrames with columns:
            - `Columns.User` - user id
            - `Columns.Item` - item id
        The original order of the rows will be preserved. Keep in mind to sort the rows correctly
        before visualizing. The most intuitive way is to sort by date in descending order. If user
        has too many interactions the lest ones may not be displayed.
    auto_display : bool, optional, default ``True``
        Display widgets right after initialization.
    formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional, default ``None``
        Formatter functions to apply to columns elements in the sections of interactions and recos.
        Keys of the dict must be columns names (item_data, interactions and recos columns can be
        specified here). Values bust be functions that will be applied to corresponding columns
        elements. The result of each function must be a unicode string that represents html code.
        Formatters can be used to format text, create links and display images with html.
    rows_limit : int, optional, default 20
        Maximum number of rows to display in the sections of interactions and recos.

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
    >>> selected_users = {"user_one": 1}
    >>> formatters = {"item_id": lambda x: f"<b>{x}</b>"}
    >>>
    >>> widgets = VisualApp(
    ...     recos=recos,
    ...     item_data=item_data,
    ...     interactions=interactions,
    ...     selected_users=selected_users,
    ...     formatters=formatters,
    ...     auto_display=False
    ... )
    """

    def __init__(
        self,
        recos: TablesDict,
        interactions: pd.DataFrame,
        item_data: pd.DataFrame,
        selected_users: tp.Dict[tp.Hashable, tp.Hashable],
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
    ) -> None:
        super().__init__(
            recos=recos,
            interactions=interactions,
            item_data=item_data,
            selected_users=selected_users,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
        )

    def _create_data_storage(
        self,
        recos: TablesDict,
        interactions: pd.DataFrame,
        item_data: pd.DataFrame,
        selected_users: tp.Dict[tp.Hashable, tp.Hashable],
    ) -> _AppDataStorage:
        return _AppDataStorage(
            interactions=interactions,
            recos=recos,
            selected_requests=selected_users,
            item_data=item_data,
            is_u2i=True,
        )


class ItemToItemVisualApp(VisualAppBase):
    r"""
    Main tool for item-to-item recommendations visualization. Creates Jupyter widgets for visual
    analysis and comparison of different models. Outputs both target item data and recommended items
    data from different models for all of the selected items.

    Models for comparison will be listed from the `recos` dictionary keys.
    Items display names for comparison will be listed from the `selected_items` keys and ids will be
    taken from `selected_items` values.

    Optionally use `formatters` to process dataframe columns values to desired html outputs.

    Parameters
    ----------
    recos : TablesDict
        Recommendations from different models in a form of a dict. Model names are supposed to be
        dict keys. Recommendations from models are supposed to be in form of pandas DataFrames with
        columns:
            - `Columns.TargetItem` - target item id
            - `Columns.Item` - recommended item id
            - Any other columns that you wish to display in widgets (e.g. rank or score)
        The original order of the rows will be preserved. Keep in mind to sort the rows correctly
        before visualizing. The most intuitive way is to sort by rank in ascending order.
    item_data : pd.DataFrame
        Data for items that is used for visualisation in both interactions and recos widgets.
        Supposed to be in form of a pandas DataFrame with columns:
            - `Columns.Item` - item id
            - Any other columns with item data (e.g. name, category, popularity, image link)
    selected_items : tp.Dict[tp.Hashable, tp.Hashable]
        Predefined items that will be displayed in widgets. Item names must be specified as keys
        of the dict and item ids as values of the dict.
    auto_display : bool, optional, default ``True``
        Display widgets right after initialization.
    formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional, default ``None``
        Formatter functions to apply to columns elements in the sections of interactions and recos.
        Keys of the dict must be columns names (item_data, interactions and recos columns can be
        specified here). Values bust be functions that will be applied to corresponding columns
        elements. The result of each function must be a unicode string that represents html code.
        Formatters can be used to format text, create links and display images with html.
    rows_limit : int, optional, default 20
        Maximum number of rows to display in the sections of interactions and recos.

    Examples
    --------
    >>> recos = {
    ...     "model1": pd.DataFrame({Columns.TargetItem: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
    ...     "model2": pd.DataFrame({Columns.TargetItem: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]})
    ... }
    >>>
    >>> item_data = pd.DataFrame({
    ...     Columns.Item: [3, 4, 5, 6, 7, 8],
    ...     "feature_1": ["one", "two", "three", "five", "one", "two"]
    ... })
    >>>
    >>> selected_items = {"item_one": 1}
    >>> formatters = {"item_id": lambda x: f"<b>{x}</b>"}
    >>>
    >>> widgets = ItemToItemVisualApp(
    ...     recos=recos,
    ...     item_data=item_data,
    ...     selected_items=selected_items,
    ...     formatters=formatters,
    ...     auto_display=False
    ... )
    """

    def __init__(
        self,
        recos: TablesDict,
        item_data: pd.DataFrame,
        selected_items: tp.Dict[tp.Hashable, tp.Hashable],
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
    ) -> None:
        super().__init__(
            recos=recos,
            item_data=item_data,
            selected_items=selected_items,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
        )

    def _create_data_storage(
        self,
        recos: TablesDict,
        item_data: pd.DataFrame,
        selected_items: tp.Dict[tp.Hashable, tp.Hashable],
    ) -> _AppDataStorage:
        return _AppDataStorage(
            recos=recos,
            selected_requests=selected_items,
            item_data=item_data,
            is_u2i=False,
        )
