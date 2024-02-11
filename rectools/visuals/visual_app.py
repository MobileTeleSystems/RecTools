import typing as tp
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from rectools import Columns
from rectools.utils import fast_isin

TablesDict = tp.Dict[tp.Hashable, pd.DataFrame]

MIN_WIDTH_LIMIT = 10


class StorageFiles:
    """Fixed file names for _AppDataStorage saving and loading."""

    Interactions = "interactions.csv"
    Recos = "recos.csv"
    Requests = "requests.csv"


class _AppDataStorage:
    """
    Storage and processing of data for `VisualApp` widgets. This class is not meant to be used
    directly. Use `VisualApp` or `ItemToItemVisualApp` class instead
    """

    def __init__(
        self,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        is_u2i: bool,
        id_col: str,
        grouped_interactions: TablesDict,
        grouped_recos: tp.Dict[tp.Hashable, TablesDict],
    ) -> None:
        self.id_col = id_col
        self.is_u2i = is_u2i
        self.selected_requests = selected_requests
        self.grouped_interactions = grouped_interactions
        self.grouped_recos = grouped_recos

    @classmethod
    def from_raw(
        cls,
        recos: TablesDict,
        item_data: pd.DataFrame,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        is_u2i: bool = True,
        n_random_requests: int = 0,
        interactions: tp.Optional[pd.DataFrame] = None,
    ) -> "_AppDataStorage":
        id_col = Columns.User if is_u2i else Columns.TargetItem
        # is_u2i

        if n_random_requests > 0:
            selected_requests = cls._fill_requests_with_random(
                selected_requests=selected_requests,
                n_random_requests=n_random_requests,
                id_col=id_col,
                recos=recos,
            )
        cls._check_columns_present_in_recos(recos=recos, id_col=id_col)

        if Columns.Item not in item_data:
            raise KeyError(f"Missed {Columns.Item} column in item_data")

        if interactions is not None and not is_u2i:
            raise ValueError("For i2i reco you must not specify interactions")
        if interactions is None:
            if is_u2i:
                raise ValueError("For u2i reco you must specify interactions")
            interactions = cls._prepare_interactions_for_i2i(recos=recos)

        if not selected_requests:
            raise ValueError("`selected_requests` is empty")

        grouped_interactions = cls._group_interactions(
            interactions=interactions,
            selected_requests=selected_requests,
            id_col=id_col,
            item_data=item_data,
        )
        grouped_recos = cls._group_recos(
            recos=recos,
            selected_requests=selected_requests,
            id_col=id_col,
            item_data=item_data,
        )
        return _AppDataStorage(
            id_col=id_col,
            is_u2i=is_u2i,
            selected_requests=selected_requests,
            grouped_interactions=grouped_interactions,
            grouped_recos=grouped_recos,
        )

    @property
    def request_names(self) -> tp.List[tp.Hashable]:
        return list(self.selected_requests.keys())

    @property
    def model_names(self) -> tp.List[tp.Hashable]:
        return list(self.grouped_recos.keys())

    @classmethod
    def _fill_requests_with_random(
        cls,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        n_random_requests: int,
        id_col: str,
        recos: TablesDict,
    ) -> tp.Dict[tp.Hashable, tp.Hashable]:

        # Leave only those ids that were not predefined by user
        # Request ids (e.g. user ids) are stored as values in `selected_requests`
        all_ids = [model_recos[id_col].unique() for model_recos in recos.values()]
        unique_ids = np.unique(np.hstack(all_ids))
        selected_ids = np.array(list(selected_requests.values()))
        selected_mask = fast_isin(unique_ids, selected_ids)
        selecting_from = unique_ids[~selected_mask]

        num_selecting = min(len(selecting_from), n_random_requests)
        new_ids = np.random.choice(selecting_from, num_selecting, replace=False)
        res = selected_requests.copy()
        new_requests: tp.Dict[tp.Hashable, tp.Hashable] = {f"random_{i+1}": new_id for i, new_id in enumerate(new_ids)}
        res.update(new_requests)
        return res

    @classmethod
    def _group_interactions(
        cls,
        interactions: pd.DataFrame,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        id_col: str,
        item_data: pd.DataFrame,
    ) -> TablesDict:
        # Request ids (e.g. user ids) are stored as values in `selected_requests`
        selected_interactions = interactions[interactions[id_col].isin(selected_requests.values())]
        selected_interactions = selected_interactions.merge(item_data, how="left", on="item_id")
        prepared_interactions = {}
        for request_name, request_id in selected_requests.items():
            prepared_interactions[request_name] = selected_interactions[
                selected_interactions[id_col] == request_id
            ].drop(columns=[id_col])
        return prepared_interactions

    @classmethod
    def _group_recos(
        cls,
        recos: TablesDict,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        id_col: str,
        item_data: pd.DataFrame,
    ) -> tp.Dict[tp.Hashable, TablesDict]:
        prepared_recos = {}
        for model_name, model_recos in recos.items():
            # Request ids (e.g. user ids) are stored as values in `selected_requests`
            selected_recos = model_recos[model_recos[id_col].isin(selected_requests.values())]
            prepared_model_recos = {}
            for request_name, request_id in selected_requests.items():
                prepared_model_recos[request_name] = item_data.merge(
                    selected_recos[selected_recos[id_col] == request_id],
                    how="right",
                    on="item_id",
                    suffixes=["_item", "_recos"],
                ).drop(columns=[id_col])
            prepared_recos[model_name] = prepared_model_recos
        return prepared_recos

    @classmethod
    def _ungroup_recos(
        cls,
        grouped_recos: tp.Dict[tp.Hashable, TablesDict],
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        id_col: str,
    ) -> pd.DataFrame:
        res = []
        for model_name, prepared_model_recos in grouped_recos.items():
            for request_name, request_recos in prepared_model_recos.items():
                df = request_recos.copy()
                df[id_col] = selected_requests[request_name]
                df["model"] = model_name
                res.append(df)
        return pd.condat(res, axis=0)

    @classmethod
    def _ungroup_interactions(
        cls,
        grouped_interactions: TablesDict,
        selected_requests: tp.Dict[tp.Hashable, tp.Hashable],
        id_col: str,
    ) -> pd.DataFrame:
        res = []
        for request_name, request_interactions in grouped_interactions.items():
            df = request_interactions.copy()
            df[id_col] = selected_requests[request_name]
            res.append(df)
        return pd.condat(res, axis=0)

    @classmethod
    def _check_columns_present_in_recos(cls, recos: TablesDict, id_col: str) -> None:
        required_columns = {id_col, Columns.Item}
        for model_name, model_recos in recos.items():
            actual_columns = set(model_recos.columns)
            if not actual_columns >= required_columns:
                raise KeyError(f"Missed columns {required_columns - actual_columns} in {model_name} recommendations df")

    @classmethod
    def _prepare_interactions_for_i2i(cls, recos: TablesDict) -> pd.DataFrame:
        request_ids = set()
        for recos_df in recos.values():
            request_ids.update(set(recos_df[Columns.TargetItem].unique()))
        interactions = pd.DataFrame({Columns.TargetItem: list(request_ids), Columns.Item: list(request_ids)})
        return interactions

    @classmethod
    def _df_to_tables_dict(cls, df: pd.DataFrame, colname: str) -> TablesDict:
        res = {}
        for value in df[colname].unique():
            res[value] = df[df[colname] == value]
        return res

    @classmethod
    def _create_requests_df(cls, selected_requests: tp.Dict[tp.Hashable, tp.Hashable]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "request_name": selected_requests.keys(),
                "request_id": selected_requests.values(),
            }
        )
        return df

    def save(self, folder_name: str, overwrite: bool = False) -> None:
        interactions_df = self._ungroup_interactions(
            grouped_interactions=self.grouped_interactions, selected_requests=self.selected_requests, id_col=self.id_col
        )
        recos_df = self._ungroup_recos(
            grouped_recos=self.grouped_recos, selected_requests=self.selected_requests, id_col=self.id_col
        )
        requests_df = self._create_requests_df(self.selected_requests)

        mode = "w" if overwrite else "x"
        interactions_df.to_csv(Path(folder_name, StorageFiles.Interactions), index=False, mode=mode)
        recos_df.to_csv(Path(folder_name, StorageFiles.Recos), index=False, mode=mode)
        requests_df.to_csv(Path(folder_name, StorageFiles.Requests), index=False, mode=mode)

    @classmethod
    def from_saved(cls, folder_name: str) -> "_AppDataStorage":
        interactions = pd.read_csv(Path(folder_name, StorageFiles.Interactions))
        recos = pd.read_csv(Path(folder_name, StorageFiles.Recos))
        selected_requests = pd.read_csv(
            Path(folder_name, StorageFiles.Requests),
            header=None,
            index_col=0,
        )[1].to_dict()

        if Columns.TargetItem in interactions.columns and Columns.User in interactions.columns:
            raise ValueError(
                """Unable to create VisualApp. Saved interactions have both columns:
                {Columns.TargetItem} and {Columns.User}"""
            )

        if Columns.User in interactions.columns:
            is_u2i = True
            id_col = Columns.User
        elif Columns.TargetItem in interactions.columns:
            is_u2i = False
            id_col = Columns.TargetItem
        else:
            raise ValueError(
                """Unable to create VisualApp. Saved interactions don't have any of the columns:
                {Columns.TargetItem} or {Columns.User}"""
            )

        dummy_item_data = pd.DataFrame(columns=[Columns.Item])
        grouped_interactions = cls._group_interactions(
            interactions=interactions, selected_requests=selected_requests, id_col=id_col, item_data=dummy_item_data
        )

        recos_dict = cls._df_to_tables_dict(recos, "model")
        grouped_recos = cls._group_recos(
            recos=recos_dict, selected_requests=selected_requests, id_col=id_col, item_data=dummy_item_data
        )

        return _AppDataStorage(
            selected_requests=selected_requests,
            is_u2i=is_u2i,
            id_col=id_col,
            grouped_interactions=grouped_interactions,
            grouped_recos=grouped_recos,
        )


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
        min_width: int = 100,
        data_storage: tp.Optional[_AppDataStorage] = None,
        **kwargs: tp.Any,
    ) -> None:
        self.rows_limit = rows_limit
        self.formatters = formatters if formatters is not None else {}

        if data_storage is None:
            data_storage = self._create_data_storage(*args, **kwargs)
        self.data_storage: _AppDataStorage = data_storage

        if min_width <= MIN_WIDTH_LIMIT:
            raise ValueError(f"`min_width` must be greater then {MIN_WIDTH_LIMIT}. {min_width} specified")
        self.min_width = min_width
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
            .replace("<th>", f'<th style="text-align: center; min-width: {self.min_width}px;">')
        )
        return html_repr

    def _display_interactions(self, request_name: str) -> None:
        """Display viewed items for `request_name`"""
        items_tab = widgets.Tab()
        df = self.data_storage.grouped_interactions[request_name]
        items_tab.children = [widgets.HTML(value=self._convert_to_html(df))]
        items_tab.set_title(index=0, title="Interactions")
        display(items_tab)

    def _display_recos(self, request_name: str, model_name: str) -> None:
        """Display recommended items for `request_name` from model `model_name`"""
        items_tab = widgets.Tab()
        df = self.data_storage.grouped_recos[model_name][request_name]
        items_tab.children = [widgets.HTML(value=self._convert_to_html(df))]
        items_tab.set_title(index=0, title="Recommended")
        display(items_tab)

    def _display_request_id(self, request_name: str) -> None:
        """Display request_id for `request_name`"""
        request_id = self.data_storage.selected_requests[request_name]
        display(widgets.HTML(value=f"{self.data_storage.id_col}: {request_id}"))

    def _display_model_name(self, model_name: str) -> None:
        """Display model_name"""
        display(widgets.HTML(value=f"Model name: {model_name}"))

    def display(self) -> None:
        """Display full VisualApp widget"""
        request_name_selection = widgets.ToggleButtons(
            options=self.data_storage.request_names,
            description=f"Request {self.data_storage.id_col}:",
            disabled=False,
            button_style="warning",
        )
        # ToggleButtons in ipywidgets have very limited support for styling.
        # Picking specific background colors for buttons is not supported. Currently we are using
        # the `button_style` option to pick the appearance of buttons from pre-defined styles.
        # There are very limited options to choose from (e.g. `success`, `warning`, etc.)
        # See https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Styling.html.
        # Possible hacks are:
        # https://stackoverflow.com/questions/68643117/python-ipywidgets-togglebutton-style-color
        # https://stackoverflow.com/questions/72504234/styling-ipywidgets-accordion-panels-individually

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

    def save(self, folder_name: str, overwrite: bool = False) -> None:
        """_summary_

        Parameters
        ----------
        folder_name : str
            _description_
        overwrite : bool, optional
            _description_, by default False
        """
        self.data_storage.save(folder_name, overwrite)

    @classmethod
    def from_saved(
        cls,
        folder_name: str,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 100,
    ) -> "VisualAppBase":  # todo: make different types of app
        """_summary_

        Parameters
        ----------
        folder_name : str
            _description_
        auto_display : bool, optional
            _description_, by default True
        formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional
            _description_, by default None
        rows_limit : int, optional
            _description_, by default 20
        min_width : int, optional
            _description_, by default 100

        Returns
        -------
        VisualApp
            _description_
        """
        data_storage = _AppDataStorage.from_saved(folder_name=folder_name)
        return VisualAppBase(
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
            min_width=min_width,
            data_storage=data_storage,
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
    interactions : pd.DataFrame
        Table with interactions history for users. Only needed for u2i case. Supposed to be in form
        of pandas DataFrames with columns:
            - `Columns.User` - user id
            - `Columns.Item` - item id
        The original order of the rows will be preserved. Keep in mind to sort the rows correctly
        before visualizing. The most intuitive way is to sort by date in descending order. If user
        has too many interactions the lest ones may not be displayed.
    item_data : pd.DataFrame
        Data for items that is used for visualisation in both interactions and recos widgets.
        Supposed to be in form of a pandas DataFrame with columns:
            - `Columns.Item` - item id
            - Any other columns with item data (e.g. name, category, popularity, image link)
    selected_users : tp.Dict[tp.Hashable, tp.Hashable]
        Predefined users that will be displayed in widgets. User names must be specified as keys
        of the dict and user ids as values of the dict.
    n_random_users : int, default 0
        Number of random users to add for visualization from users in recommendation tables.
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
    min_width : int, optional, default 100
        Minimum column width in pixels for dataframe columns in widgets output. Must be greater then
        10.

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
        n_random_users: int = 0,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 100,
    ) -> None:
        super().__init__(
            recos=recos,
            interactions=interactions,
            item_data=item_data,
            selected_users=selected_users,
            n_random_users=n_random_users,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
            min_width=min_width,
        )

    def _create_data_storage(
        self,
        recos: TablesDict,
        interactions: pd.DataFrame,
        item_data: pd.DataFrame,
        selected_users: tp.Dict[tp.Hashable, tp.Hashable],
        n_random_users: int,
    ) -> _AppDataStorage:
        return _AppDataStorage.from_raw(
            interactions=interactions,
            recos=recos,
            selected_requests=selected_users,
            item_data=item_data,
            is_u2i=True,
            n_random_requests=n_random_users,
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
    n_random_items : int, default 0
        Number of random items to add for visualization from target items in recommendation tables.
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
    min_width : int, optional, default 100
        Minimum column width in pixels for dataframe columns in widgets output. Must be greater then
        10.

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
        n_random_items: int = 0,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 100,
    ) -> None:
        super().__init__(
            recos=recos,
            item_data=item_data,
            selected_items=selected_items,
            n_random_items=n_random_items,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
            min_width=min_width,
        )

    def _create_data_storage(
        self,
        recos: TablesDict,
        item_data: pd.DataFrame,
        selected_items: tp.Dict[tp.Hashable, tp.Hashable],
        n_random_items: int,
    ) -> _AppDataStorage:
        return _AppDataStorage.from_raw(
            recos=recos,
            selected_requests=selected_items,
            item_data=item_data,
            is_u2i=False,
            n_random_requests=n_random_items,
        )
