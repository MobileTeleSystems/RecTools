import typing as tp
from pathlib import Path

import attr
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from rectools import Columns, ExternalId
from rectools.utils import fast_isin

TablesDict = tp.Dict[tp.Hashable, pd.DataFrame]

MIN_WIDTH_LIMIT = 10
REQUEST_NAMES_COL = "request_name"
REQUEST_IDS_COL = "request_id"

VisualAppT = tp.TypeVar("VisualAppT", bound="VisualAppBase")


class StorageFiles:
    """Fixed file names for `AppDataStorage` saving and loading."""

    Interactions = "interactions.csv"
    Recommendations = "recommendations.csv"
    Requests = "requests.csv"


@attr.s(slots=True)
class AppDataStorage:
    """
    Storage and processing of data for `VisualApp` widgets. This class is not meant to be used
    directly. Use `VisualApp` or `ItemToItemVisualApp` class instead
    """

    is_u2i: bool = attr.ib()
    id_col: str = attr.ib()
    selected_requests: tp.Dict[tp.Hashable, ExternalId] = attr.ib()
    grouped_interactions: TablesDict = attr.ib()
    grouped_reco: tp.Dict[tp.Hashable, TablesDict] = attr.ib()

    @classmethod
    def from_raw(
        cls,
        reco: tp.Union[pd.DataFrame, TablesDict],
        item_data: pd.DataFrame,
        selected_requests: tp.Optional[tp.Dict[tp.Hashable, ExternalId]] = None,
        is_u2i: bool = True,
        n_random_requests: int = 0,
        interactions: tp.Optional[pd.DataFrame] = None,
    ) -> "AppDataStorage":
        r"""Create data storage for VisualApp from raw data. This class is not meant to be used
        directly. Use `VisualApp` or `ItemToItemVisualApp` class instead.

        Parameters
        ----------
        reco : tp.Union[pd.DataFrame, TablesDict]
            Recommendations from different models in a form of a pd.DataFrame or a dict.
            In DataFrame form model names must be specified in `Columns.Model` column. In dict form
            model names are supposed to be dict keys.
        item_data : pd.DataFrame
            Data for items that is used for visualisation in both interactions and recommendations
            widgets.
        selected_requests : tp.Optional[tp.Dict[tp.Hashable, ExternalId]], default ``None``
            Predefined requests (users or items) that will be displayed in widgets. Request names
            must be specified as keys of the dict and ids as values of the dict.
        is_u2i : bool, default ``True``
            Is this a user-to-item recommendation case (opposite to item-to-item).
        n_random_requests : int, default 0
            Number of random requests to add for visualization from targets in recommendation tables.
        interactions : tp.Optional[pd.DataFrame], default ``None``
            Table with interactions history for users. Only needed for u2i case.

        Returns
        -------
        AppDataStorage
            Data storage class for visualisation widgets.
        """
        id_col = Columns.User if is_u2i else Columns.TargetItem

        selected_requests = cls._validate_selected_requests(selected_requests, is_u2i, n_random_requests)

        if n_random_requests > 0:
            selected_requests = cls._fill_requests_with_random(selected_requests, n_random_requests, id_col, reco)

        if isinstance(reco, pd.DataFrame):
            if Columns.Model not in reco.columns:
                raise KeyError("Missing `{Columns.Model}` column in `reco` DataFrame")
            reco = cls._df_to_tables_dict(reco, Columns.Model)
        cls._check_columns_present_in_reco(reco=reco, id_col=id_col)

        if Columns.Item not in item_data:
            raise KeyError(f"Missed {Columns.Item} column in item_data")

        if interactions is not None and not is_u2i:
            raise ValueError("For i2i reco you must not specify interactions")
        if interactions is None:
            if is_u2i:
                raise ValueError("For u2i reco you must specify interactions")
            interactions = cls._prepare_interactions_for_i2i(reco=reco)

        grouped_interactions = cls._group_interactions(
            interactions=interactions,
            selected_requests=selected_requests,
            id_col=id_col,
            item_data=item_data,
        )
        grouped_reco = cls._group_reco(
            reco=reco,
            selected_requests=selected_requests,
            id_col=id_col,
            item_data=item_data,
        )
        return cls(
            id_col=id_col,
            is_u2i=is_u2i,
            selected_requests=selected_requests,
            grouped_interactions=grouped_interactions,
            grouped_reco=grouped_reco,
        )

    @classmethod
    def _validate_selected_requests(
        cls, selected_requests: tp.Optional[tp.Dict[tp.Hashable, ExternalId]], is_u2i: bool, n_random_requests: int
    ) -> tp.Dict[tp.Hashable, ExternalId]:
        if not selected_requests:
            if n_random_requests == 0:
                requests = "users" if is_u2i else "items"
                raise ValueError(f"Please specify `n_random_{requests}` > 0 or provide `selected_{requests}`")
            return {}
        return selected_requests

    @property
    def request_names(self) -> tp.List[tp.Hashable]:
        """Names of selected requests for comparison"""
        return list(self.selected_requests.keys())

    @property
    def model_names(self) -> tp.List[tp.Hashable]:
        """Names of recommendation models for comparison"""
        return list(self.grouped_reco.keys())

    @classmethod
    def _fill_requests_with_random(
        cls,
        selected_requests: tp.Dict[tp.Hashable, ExternalId],
        n_random_requests: int,
        id_col: str,
        reco: TablesDict,
    ) -> tp.Dict[tp.Hashable, ExternalId]:

        # Leave only those ids that were not predefined by user
        all_ids = [model_reco[id_col].unique() for model_reco in reco.values()]
        unique_ids = pd.unique(np.hstack(all_ids))
        selected_ids = np.array(list(selected_requests.values()))
        selected_mask = fast_isin(unique_ids, selected_ids)
        selecting_from = unique_ids[~selected_mask]

        num_selecting = min(len(selecting_from), n_random_requests)
        new_ids = np.random.choice(selecting_from, num_selecting, replace=False)
        res = selected_requests.copy()
        new_requests: tp.Dict[tp.Hashable, ExternalId] = {f"random_{i+1}": new_id for i, new_id in enumerate(new_ids)}
        res.update(new_requests)
        return res

    @classmethod
    def _group_interactions(
        cls,
        interactions: pd.DataFrame,
        selected_requests: tp.Dict[tp.Hashable, ExternalId],
        id_col: str,
        item_data: tp.Optional[pd.DataFrame] = None,
    ) -> TablesDict:
        selected_interactions = interactions[interactions[id_col].isin(selected_requests.values())]
        if item_data is not None:
            selected_interactions = selected_interactions.merge(item_data, how="left", on="item_id")
        prepared_interactions = {}
        for request_name, request_id in selected_requests.items():
            prepared_interactions[request_name] = selected_interactions[
                selected_interactions[id_col] == request_id
            ].drop(columns=[id_col])
        return prepared_interactions

    @classmethod
    def _group_reco(
        cls,
        reco: TablesDict,
        selected_requests: tp.Dict[tp.Hashable, ExternalId],
        id_col: str,
        item_data: tp.Optional[pd.DataFrame] = None,
        drop_na_reco_cols: bool = False,
    ) -> tp.Dict[tp.Hashable, TablesDict]:
        prepared_reco = {}
        for model_name, model_reco in reco.items():
            selected_reco = model_reco[model_reco[id_col].isin(selected_requests.values())]
            prepared_model_reco = {}
            for request_name, request_id in selected_requests.items():
                request_reco = (
                    selected_reco[selected_reco[id_col] == request_id].drop(columns=[id_col]).reset_index(drop=True)
                )
                if drop_na_reco_cols:
                    request_reco = request_reco.dropna(axis=1, how="all")
                if item_data is not None:
                    request_reco = item_data.merge(
                        request_reco,
                        how="right",
                        on="item_id",
                        suffixes=["_item", "_reco"],
                    )
                prepared_model_reco[request_name] = request_reco
            prepared_reco[model_name] = prepared_model_reco
        return prepared_reco

    @classmethod
    def _ungroup_reco(
        cls,
        grouped_reco: tp.Dict[tp.Hashable, TablesDict],
        selected_requests: tp.Dict[tp.Hashable, ExternalId],
        id_col: str,
    ) -> pd.DataFrame:
        res = []
        for model_name, prepared_model_reco in grouped_reco.items():
            for request_name, request_reco in prepared_model_reco.items():
                df = request_reco.copy()
                df[id_col] = selected_requests[request_name]
                df[Columns.Model] = model_name
                res.append(df)
        return pd.concat(res, axis=0, sort=False).reset_index(drop=True)

    @classmethod
    def _ungroup_interactions(
        cls,
        grouped_interactions: TablesDict,
        selected_requests: tp.Dict[tp.Hashable, ExternalId],
        id_col: str,
    ) -> pd.DataFrame:
        res = []
        for request_name, request_interactions in grouped_interactions.items():
            df = request_interactions.copy()
            df[id_col] = selected_requests[request_name]
            res.append(df)
        return pd.concat(res, axis=0, sort=False).reset_index(drop=True)

    @classmethod
    def _check_columns_present_in_reco(cls, reco: TablesDict, id_col: str) -> None:
        required_columns = {id_col, Columns.Item}
        for model_name, model_reco in reco.items():
            actual_columns = set(model_reco.columns)
            if not actual_columns >= required_columns:
                raise KeyError(f"Missed columns {required_columns - actual_columns} in {model_name} recommendations df")

    @classmethod
    def _prepare_interactions_for_i2i(cls, reco: TablesDict) -> pd.DataFrame:
        request_ids = set()
        for reco_df in reco.values():
            request_ids.update(set(reco_df[Columns.TargetItem].unique()))
        interactions = pd.DataFrame({Columns.TargetItem: list(request_ids), Columns.Item: list(request_ids)})
        return interactions

    @classmethod
    def _df_to_tables_dict(cls, df: pd.DataFrame, key_col: str) -> TablesDict:
        res = {}
        for key, grouped_df in df.groupby(key_col):
            res[key] = grouped_df.drop(columns=[key_col]).reset_index(drop=True)
        return res

    def save(self, folder_name: str, overwrite: bool = False) -> None:
        """Save stored data for `VisualApp` widgets. This method is not meant to be used
        directly. Use `VisualApp` or `ItemToItemVisualApp` class methods instead.

        Parameters
        ----------
        folder_name : str
            Destination folder for data.
        overwrite : bool, default ``False``
            Allow to overwrite in the folder files if they already exist.
        """
        interactions_df = self._ungroup_interactions(
            grouped_interactions=self.grouped_interactions, selected_requests=self.selected_requests, id_col=self.id_col
        )
        reco_df = self._ungroup_reco(
            grouped_reco=self.grouped_reco, selected_requests=self.selected_requests, id_col=self.id_col
        )
        requests_df = pd.Series(self.selected_requests, name=REQUEST_IDS_COL)

        Path(folder_name).mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        interactions_df.to_csv(Path(folder_name, StorageFiles.Interactions), index=False, mode=mode)
        reco_df.to_csv(Path(folder_name, StorageFiles.Recommendations), index=False, mode=mode)
        requests_df.to_csv(Path(folder_name, StorageFiles.Requests), index_label=REQUEST_NAMES_COL, mode=mode)

    @classmethod
    def load(cls, folder_name: str) -> "AppDataStorage":
        r"""Load prepared data for VisualApp widgets. This method is not meant to be used
        directly. Use `VisualApp` or `ItemToItemVisualApp` class methods instead.

        Parameters
        ----------
        folder_name : str
            Folder where data was saved earlier.

        Returns
        -------
        AppDataStorage
            Data storage class for visualisation widgets.
        """
        interactions = pd.read_csv(Path(folder_name, StorageFiles.Interactions))
        reco = pd.read_csv(Path(folder_name, StorageFiles.Recommendations))
        selected_requests_df = pd.read_csv(Path(folder_name, StorageFiles.Requests), index_col=REQUEST_NAMES_COL)
        selected_requests = selected_requests_df[REQUEST_IDS_COL].to_dict()

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

        grouped_interactions = cls._group_interactions(
            interactions=interactions, selected_requests=selected_requests, id_col=id_col
        )

        reco_dict = cls._df_to_tables_dict(reco, Columns.Model)
        # We need to drop na cols from reco dfs because they could have appeared during pd.concat when saving
        grouped_reco = cls._group_reco(
            reco=reco_dict, selected_requests=selected_requests, id_col=id_col, drop_na_reco_cols=True
        )

        return cls(
            selected_requests=selected_requests,
            is_u2i=is_u2i,
            id_col=id_col,
            grouped_interactions=grouped_interactions,
            grouped_reco=grouped_reco,
        )


class VisualAppBase:
    """
    Jupyter widgets app for recommendations visualization and models comparison.
    Warning! This is a base class.
    Do not create instances of this class directly. Use derived classes `construct` methods instead.
    """

    def __init__(
        self,
        data_storage: AppDataStorage,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 50,
    ) -> None:
        self.data_storage = data_storage
        self.rows_limit = rows_limit
        self.formatters = formatters if formatters is not None else {}

        if min_width <= MIN_WIDTH_LIMIT:
            raise ValueError(f"`min_width` must be greater then {MIN_WIDTH_LIMIT}. {min_width} specified")
        self.min_width = min_width
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

    def _display_recommendations(self, request_name: str, model_name: str) -> None:
        """Display recommended items for `request_name` from model `model_name`"""
        items_tab = widgets.Tab()
        df = self.data_storage.grouped_reco[model_name][request_name]
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
            description="Target:",
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
        reco_output = widgets.interactive_output(
            self._display_recommendations, {"request_name": request_name_selection, "model_name": model_selection}
        )

        display(
            widgets.VBox(
                [
                    request_name_selection,
                    request_id_output,
                    interactions_output,
                    model_selection,
                    model_name_output,
                    reco_output,
                ]
            )
        )

    def save(self, folder_name: str, overwrite: bool = False) -> None:
        """Save stored data to re-create widgets when necessary. Use `VisualAppBase.load`
        class method for re-creation or any other child classes (`VisualApp`, `ItemToItemVisualApp`).

        Parameters
        ----------
        folder_name : str
            Destination folder for data.
        overwrite : bool, default ``False``
            Allow to overwrite in the folder files if they already exist.
        """
        self.data_storage.save(folder_name, overwrite)

    @classmethod
    def load(
        cls: tp.Type[VisualAppT],
        folder_name: str,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 100,
    ) -> VisualAppT:
        """Create widgets from data that was processed and saved earlier.

        Parameters
        ----------
        folder_name : str
            Destination folder for data.
        auto_display : bool, optional, default ``True``
            Display widgets right after initialization.
        formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional, default ``None``
            Formatter functions to apply to columns elements in the sections of interactions and
            recommendations. Keys of the dict must be columns names (item_data, interactions and
            recommendations columns can be specified here). Values bust be functions that will be
            applied to corresponding columns elements. The result of each function must be a unicode
            string that represents html code. Formatters can be used to format text, create links
            and display images with html.
        rows_limit : int, optional, default 20
            Maximum number of rows to display in the sections of interactions and recommendations.
        min_width : int, optional, default 100
            Minimum column width in pixels for dataframe columns in widgets output. Must be greater
            then 10.

        Returns
        -------
        VisualAppBase
            Jupyter widgets for recommendations visualization.
        """
        data_storage = AppDataStorage.load(folder_name=folder_name)

        return cls(
            data_storage=data_storage,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
            min_width=min_width,
        )


class VisualApp(VisualAppBase):
    r"""
    Jupyter widgets app for user-to-item recommendations visualization and models comparison.
    Do not create instances of this class directly. Use `VisualApp.construct` method instead.
    """

    @classmethod
    def construct(
        cls,
        reco: tp.Union[pd.DataFrame, TablesDict],
        interactions: pd.DataFrame,
        item_data: pd.DataFrame,
        selected_users: tp.Optional[tp.Dict[tp.Hashable, ExternalId]] = None,
        n_random_users: int = 0,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 100,
    ) -> "VisualApp":
        r"""
        Construct visualization app for classic user-to-item recommendations.

        This will process raw data and create Jupyter widgets for visual analysis and comparison of
        different models. Created app outputs both interactions history of the selected
        users and their recommended items from different models along with explicit items data.

        Model names for comparison will be listed from the `reco` dictionary keys or `reco` dataframe
        `Columns.Model` values depending on the format provided.

        Users for comparison can be predefined or random.
        For predefined users pass `selected_users` dict with user "names" as keys and user ids as
        values.
        For random users pass `n_random_users` number greater then 0.
        You must specify at least one of the above or provide both.

        Optionally provide `formatters` to process dataframe columns values to desired html outputs.

        Parameters
        ----------
        reco : tp.Union[pd.DataFrame, TablesDict]
            Recommendations from different models in a form of a pd.DataFrame or a dict. In the dict
            form model names are supposed to be dict keys, and recommendations from different models are
            supposed to be pd.DataFrames as dict values.
            In the DataFrame form all recommendations must be specified in one DataFrame with
            `Columns.Model` column to separate different models.
            Other required columns for both forms are:
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
            Data for items that is used for visualisation in both interactions and recommendations widgets.
            Supposed to be in form of a pandas DataFrame with columns:
                - `Columns.Item` - item id
                - Any other columns with item data (e.g. name, category, popularity, image link)
        selected_users : tp.Optional[tp.Dict[tp.Hashable, ExternalId]], default ``None``
            Predefined users that will be displayed in widgets. User names must be specified as keys
            of the dict and user ids as values of the dict. Must be provided if `n_random_users` = 0.
        n_random_users : int, default 0
            Number of random users to add for visualization from users in recommendation tables. Must
            be greater then 0 if `selected_users` are not specified.
        auto_display : bool, optional, default ``True``
            Display widgets right after initialization.
        formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional, default ``None``
            Formatter functions to apply to columns elements in the sections of interactions and
            recommendations. Keys of the dict must be columns names (item_data, interactions and
            recommendations columns can be specified here). Values bust be functions that will be
            applied to corresponding columns elements. The result of each function must be a unicode
            string that represents html code. Formatters can be used to format text, create links
            and display images with html.
        rows_limit : int, optional, default 20
            Maximum number of rows to display in the sections of interactions and recommendations.
        min_width : int, optional, default 100
            Minimum column width in pixels for dataframe columns in widgets output. Must be greater then
            10.

        Examples
        --------
        Providing reco as TablesDict

        >>> reco = {
        ...     "model_1": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
        ...     "model_2": pd.DataFrame({Columns.User: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]})
        ... }
        >>>
        >>> item_data = pd.DataFrame({
        ...     Columns.Item: [3, 4, 5, 6, 7, 8],
        ...     "feature_1": ["one", "two", "three", "five", "one", "two"]
        ... })
        >>>
        >>> interactions = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
        >>> selected_users = {"user_one": 1}
        >>>
        >>> app = VisualApp.construct(
        ...     reco=reco,
        ...     item_data=item_data,
        ...     interactions=interactions,
        ...     selected_users=selected_users,
        ...     auto_display=False
        ... )

        Providing reco as pd.DataFrame and adding `formatters`

        >>> reco = pd.DataFrame({
        ...     Columns.User: [1, 2, 1, 2],
        ...     Columns.Item: [3, 4, 5, 6],
        ...     Columns.Model: ["model_1", "model_1", "model_2", "model_2"]
        ... })
        >>>
        >>> item_data = pd.DataFrame({
        ...     Columns.Item: [3, 4, 5, 6, 7, 8],
        ...     "feature_1": ["one", "two", "three", "five", "one", "two"]
        ... })
        >>>
        >>> interactions = pd.DataFrame({Columns.User: [1, 1, 2], Columns.Item: [3, 7, 8]})
        >>> selected_users = {"user_one": 1}
        >>>
        >>> formatters = {"item_id": lambda x: f"<b>{x}</b>"}
        >>>
        >>> app = VisualApp.construct(
        ...     reco=reco,
        ...     item_data=item_data,
        ...     interactions=interactions,
        ...     selected_users=selected_users,
        ...     formatters=formatters,
        ...     auto_display=False
        ... )
        """
        data_storage = AppDataStorage.from_raw(
            interactions=interactions,
            reco=reco,
            selected_requests=selected_users,
            item_data=item_data,
            is_u2i=True,
            n_random_requests=n_random_users,
        )
        return cls(
            data_storage=data_storage,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
            min_width=min_width,
        )


class ItemToItemVisualApp(VisualAppBase):
    r"""
    Jupyter widgets app for item-to-item recommendations visualization and models comparison.
    Do not create instances of this class directly. Use `ItemToItemVisualApp.construct` method
    instead.
    """

    @classmethod
    def construct(
        cls,
        reco: tp.Union[pd.DataFrame, TablesDict],
        item_data: pd.DataFrame,
        selected_items: tp.Optional[tp.Dict[tp.Hashable, ExternalId]] = None,
        n_random_items: int = 0,
        auto_display: bool = True,
        formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        rows_limit: int = 20,
        min_width: int = 100,
    ) -> "ItemToItemVisualApp":
        r"""
        Construct visualization widgets for item-to-item recommendations.

        This will process raw data and create Jupyter widgets for visual analysis and comparison of
        different models. Created app outputs both target item data and recommended items data from
        different models for all of the selected items.

        Model names for comparison will be listed from the `reco` dictionary keys or `reco` dataframe
        `Columns.Model` values depending on the format provided.

        Target items for comparison can be predefined or random.
        For predefined items pass `selected_items` dict with item "names" as keys and item ids as
        values.
        For random target items pass `n_random_items` number greater then 0.
        You must specify at least one of the above or provide both.

        Optionally provide `formatters` to process dataframe columns values to desired html outputs.

        Parameters
        ----------
        reco : tp.Union[pd.DataFrame, TablesDict]
            Recommendations from different models in a form of a pd.DataFrame or a dict. In the dict
            form model names are supposed to be dict keys, and recommendations from different models are
            supposed to be pd.DataFrames as dict values.
            In the DataFrame form all recommendations must be specified in one DataFrame with
            `Columns.Model` column to separate different models.
            Other required columns for both forms are:
                - `Columns.TargetItem` - target item id
                - `Columns.Item` - recommended item id
                - Any other columns that you wish to display in widgets (e.g. rank or score)
            The original order of the rows will be preserved. Keep in mind to sort the rows correctly
            before visualizing. The most intuitive way is to sort by rank in ascending order.
        item_data : pd.DataFrame
            Data for items that is used for visualisation in both interactions and recommendations widgets.
            Supposed to be in form of a pandas DataFrame with columns:
                - `Columns.Item` - item id
                - Any other columns with item data (e.g. name, category, popularity, image link)
        selected_items :  tp.Optional[tp.Dict[tp.Hashable, ExternalId]], default ``None``
            Predefined items that will be displayed in widgets. Item names must be specified as keys
            of the dict and item ids as values of the dict.  Must be provided if `n_random_items` = 0.
        n_random_items : int, default 0
            Number of random items to add for visualization from target items in recommendation tables.
            Must be greater then 0 if `selected_items` are not specified.
        auto_display : bool, optional, default ``True``
            Display widgets right after initialization.
        formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional, default ``None``
            Formatter functions to apply to columns elements in the sections of interactions and
            recommendations. Keys of the dict must be columns names (item_data, interactions and
            recommendations columns can be specified here). Values bust be functions that will be
            applied to corresponding columns elements. The result of each function must be a unicode
            string that represents html code. Formatters can be used to format text, create links
            and display images with html.
        rows_limit : int, optional, default 20
            Maximum number of rows to display in the sections of interactions and recommendations.
        min_width : int, optional, default 100
            Minimum column width in pixels for dataframe columns in widgets output. Must be greater then
            10.

        Examples
        --------
        Providing reco as TablesDict

        >>> reco = {
        ...     "model_1": pd.DataFrame({Columns.TargetItem: [1, 2], Columns.Item: [3, 4], Columns.Score: [0.99, 0.9]}),
        ...     "model_2": pd.DataFrame({Columns.TargetItem: [1, 2], Columns.Item: [5, 6], Columns.Rank: [1, 1]})
        ... }
        >>>
        >>> item_data = pd.DataFrame({
        ...     Columns.Item: [3, 4, 5, 6, 1, 2],
        ...     "feature_1": ["one", "two", "three", "five", "one", "two"]
        ... })
        >>>
        >>> selected_items = {"item_one": 1}
        >>>
        >>> app = ItemToItemVisualApp.construct(
        ...     reco=reco,
        ...     item_data=item_data,
        ...     selected_items=selected_items,
        ...     auto_display=False
        ... )

        Providing reco as pd.DataFrame and adding `formatters`

        >>> reco = pd.DataFrame({
        ...     Columns.TargetItem: [1, 2, 1, 2],
        ...     Columns.Item: [3, 4, 5, 6],
        ...     Columns.Model: ["model_1", "model_1", "model_2", "model_2"]
        ... })
        >>>
        >>> item_data = pd.DataFrame({
        ...     Columns.Item: [3, 4, 5, 6, 1, 2],
        ...     "feature_1": ["one", "two", "three", "five", "one", "two"]
        ... })
        >>>
        >>> selected_items = {"item_one": 1}
        >>> formatters = {"item_id": lambda x: f"<b>{x}</b>"}
        >>>
        >>> app = ItemToItemVisualApp.construct(
        ...     reco=reco,
        ...     item_data=item_data,
        ...     selected_items=selected_items,
        ...     formatters=formatters,
        ...     auto_display=False
        ... )
        """
        data_storage = AppDataStorage.from_raw(
            reco=reco,
            selected_requests=selected_items,
            item_data=item_data,
            is_u2i=False,
            n_random_requests=n_random_items,
        )
        return cls(
            data_storage=data_storage,
            auto_display=auto_display,
            formatters=formatters,
            rows_limit=rows_limit,
            min_width=min_width,
        )
