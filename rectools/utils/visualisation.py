# pylint: disable=too-many-branches

import os
import typing as tp

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets.widgets import widget_selectioncontainer

from rectools import Columns

# import warnings


# # Prevent widgets output from being collapsed
# style = """
#     <style>
#        .jupyter-widgets-output-area .output_scroll {
#             height: unset !important;
#             border-radius: unset !important;
#             -webkit-box-shadow: unset !important;
#             box-shadow: unset !important;
#         }
#         .jupyter-widgets-output-area  {
#             height: auto !important;
#         }
#     </style>
#     """
# display(widgets.HTML(style))

SHOWCASE_FOLDER_NAME = "showcase"

# warnings.filterwarnings("ignore")
# pd.set_option("display.max_colwidth", -1)


class FileNames:
    """Fixed names for saving data files"""

    Interactions = "interactions.csv"
    Recos = "recos.csv"
    RequestsDict = "requests_dict.csv"
    Items = "items_data.csv"


class ItemTypes:
    """Fixed names for item types"""

    Viewed = "viewed"
    Recommended = "recommended"


class ShowcaseDataStorage:
    """
    Helper class to hold all data for showcase purposes.
    - Holds info about interactions, recommendations and item_data
    - Holds `requests_dict`
    - Supports adding random ids to `requests_dict`
    - Supports saving and loading data
    - Supports removing exceeding data that is not needed to display requests from `requests_dict` and their items.
    """

    def __init__(  # noqa: C901
        self,
        recos: pd.DataFrame,
        is_u2i: bool = True,
        requests_dict: tp.Optional[tp.Dict[str, tp.Any]] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        item_data: tp.Optional[pd.DataFrame] = None,
        n_add_random_requests: int = 0,
        remove_exceeding_data: tp.Optional[bool] = True,
    ) -> None:

        # Check correct Columns in recos
        if is_u2i:
            assert Columns.User in recos.columns
            request_colname = Columns.User
        else:
            assert Columns.TargetItem in recos.columns
            request_colname = Columns.TargetItem

        # Check for Model column in recos
        if Columns.Model not in recos.columns:
            if "model_name" in recos.columns:
                recos.rename(columns={"model_name": Columns.Model}, inplace=True)
            else:
                recos[Columns.Model] = "Reco_Model"

        # Check sorting in recos
        if Columns.Rank in recos:
            recos = recos.sort_values(by=[request_colname, Columns.Rank])

        # Check params
        if requests_dict is None and n_add_random_requests == 0:
            raise ValueError("Please either set `n_add_random_requests` > 0 or provide `requests_dict`")

        if interactions is None:
            if is_u2i:
                raise ValueError("For u2i reco showcase you must specify interactions")
            request_ids = recos[Columns.TargetItem].unique()
            interactions = pd.DataFrame({Columns.TargetItem: request_ids, Columns.Item: request_ids})

        if item_data is None:
            raise ValueError("You must specify item_data for showcase")

        self.requests_dict = requests_dict if requests_dict is not None else {}
        self.interactions = interactions
        self.is_u2i = is_u2i
        self.request_colname = request_colname
        self.recos = recos
        self.model_names = recos[Columns.Model].unique()
        self.item_data = item_data
        self.exceeding_data_removed = False
        if n_add_random_requests > 0:
            self.update_requests_with_random(n=n_add_random_requests)

        if remove_exceeding_data:
            self.remove_exceeding_data()

    def get_relevant_items(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        inter_items = self.interactions[Columns.Item].unique()
        recos_items = self.recos[Columns.Item].unique()
        all_items = np.union1d(inter_items, recos_items)
        if not self.is_u2i:
            request_items = self.recos[Columns.TargetItem].unique()
            all_items = np.union1d(all_items, request_items)
        return all_items

    def get_request_names(self) -> tp.List[str]:
        """_summary_

        Returns
        -------
        tp.List[str]
            _description_
        """
        return [*self.requests_dict.keys()]

    def get_request_idx(self) -> tp.List[str]:
        """_summary_

        Returns
        -------
        tp.List[str]
            _description_
        """
        return [*self.requests_dict.values()]

    def get_viewed_items_for_request(self, request_id: tp.Any) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        request_id : tp.Any
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        request_interactions = self.interactions[self.interactions[self.request_colname] == request_id]
        return request_interactions[Columns.Item].unique()

    def get_recos_for_request(self, request_id: tp.Any, model_name: str) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        request_id : tp.Any
            _description_
        model_name : str
            _description_

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if model_name not in self.model_names:
            raise ValueError(f"{model_name} not in model names: {self.model_names}")
        model_recos = self.recos[
            (self.recos[Columns.Model] == model_name) & (self.recos[self.request_colname] == request_id)
        ]
        return model_recos[Columns.Item].unique()

    def update_requests_with_random(self, n: int = 10) -> None:
        """_summary_

        Parameters
        ----------
        n : int, optional
            _description_, by default 10

        Raises
        ------
        TypeError
            _description_
        """
        if self.exceeding_data_removed:
            raise TypeError("Not possible to select more requests since exceeding data was removed")
        all_requests = self.recos[self.request_colname].unique()
        new_idx = np.random.choice(all_requests, size=n, replace=False)
        new_requests_dict = {f"random_{i}": new_idx[i] for i in range(n)}
        self.requests_dict.update(new_requests_dict)

    def remove_exceeding_data(self) -> None:
        """_summary_

        Raises
        ------
        TypeError
            _description_
        """
        relevant_requests = self.get_request_idx()
        self.interactions = self.interactions[self.interactions[self.request_colname].isin(relevant_requests)].copy()
        self.recos = self.recos[self.recos[self.request_colname].isin(relevant_requests)].copy()
        relevant_items = self.get_relevant_items()
        if isinstance(self.item_data, pd.DataFrame):
            self.item_data = self.item_data[self.item_data[Columns.Item].isin(relevant_items)].copy()
        else:
            raise TypeError("Item data was not specified")
        self.exceeding_data_removed = True

    def save_data(
        self,
        name: str,
        showcase_folder_name: str = SHOWCASE_FOLDER_NAME,
        force_overwrite: tp.Optional[bool] = False,
    ) -> None:
        """
        Save data for Showcase in csv format
        Name can be generetated automatically if `date` in `recos` Columns.
        """
        if not os.path.exists(showcase_folder_name):
            os.mkdir(showcase_folder_name)
        data_folder_name = os.path.join(showcase_folder_name, name)
        if os.path.exists(data_folder_name):
            if not force_overwrite:
                raise ValueError(f"file {data_folder_name} already exists. Specify `force_overwrite=True` to overwrite")
        else:
            os.mkdir(data_folder_name)
        self.interactions.to_csv(os.path.join(data_folder_name, FileNames.Interactions), index=False)
        self.recos.to_csv(os.path.join(data_folder_name, FileNames.Recos), index=False)
        if isinstance(self.item_data, pd.DataFrame):
            self.item_data.to_csv(os.path.join(data_folder_name, FileNames.Items), index=False)
        # else:
        #     raise TypeError("Item data was not specified")

        pd.DataFrame(
            {
                "request_name": self.requests_dict.keys(),
                "request_id": self.requests_dict.values(),
            }
        ).to_csv(os.path.join(data_folder_name, FileNames.RequestsDict), index=False)

    @classmethod
    def load_data(cls, name: str, showcase_folder_name: str = SHOWCASE_FOLDER_NAME) -> "ShowcaseDataStorage":
        """Load Showcase from data in csv format"""
        data_folder_name = os.path.join(showcase_folder_name, name)
        interactions = pd.read_csv(
            os.path.join(data_folder_name, FileNames.Interactions)
        )  # TODO: what if it is not specified
        recos = pd.read_csv(os.path.join(data_folder_name, FileNames.Recos))
        item_data = pd.read_csv(os.path.join(data_folder_name, FileNames.Items))
        requests_dict = pd.read_csv(
            os.path.join(data_folder_name, FileNames.RequestsDict),
            header=None,
            index_col=0,
        )[1].to_dict()

        if Columns.User not in recos.columns and Columns.TargetItem not in recos.columns:
            raise ValueError(f"{Columns.User} and {Columns.TargetItem} are both not in recos Columns")

        is_u2i = Columns.User in recos.columns

        showcase_data_storage = ShowcaseDataStorage(
            interactions=interactions,
            recos=recos,
            requests_dict=requests_dict,
            item_data=item_data,
            is_u2i=is_u2i,
        )
        return showcase_data_storage


class Showcase(ShowcaseDataStorage):
    """
    Main class for recommendations visualization.
    - Provides visual information about requests in `requests_dict` and recos
    """

    def __init__(
        self,
        recos: pd.DataFrame,
        item_data: pd.DataFrame,
        is_u2i: bool = True,
        requests_dict: tp.Optional[tp.Dict[str, tp.Any]] = None,
        interactions: tp.Optional[pd.DataFrame] = None,
        n_add_random_requests: int = 0,
        remove_exceeding_data: tp.Optional[bool] = True,
        auto_display: bool = True,
        reco_cols: tp.Optional[list] = None,
        item_data_cols: tp.Optional[list] = None,
        item_df_renaming: tp.Optional[tp.Dict[str, str]] = None,
        item_df_formatters: tp.Optional[tp.Dict[str, tp.Callable]] = None,
    ) -> None:
        """Showcase is an interactive app for Jupyter notebook to visualise recos and easily
        compare different algorithms

        Parameters
        ----------
        recos : pd.DataFrame
            DataFrame with recommendations. Should have Columns ['user_id', 'item_id'] for u2i case and
            ['request_item_id', 'item_id'] for i2i case. If recos are made from different models,
            model names should be specified in 'model' column. Any additional Columns ('rank', 'score')
            are allowed and can be displayed.
        item_data : pd.DataFrame
            DataFrame with all info about items. Should contain column ['item_id']. All other Columns are custom
        is_u2i : bool, optional
            Should be set to 'True' in case of recommending items to users. Should be set to False
            in case of recommending items to items. By default True
        requests_dict : tp.Optional[tp.Dict[str, tp.Any]], optional
            Predefined request ids to display. User ids in case of u2i recos. Keys are names of users
            (e.g. 'western_lover'), values are user ids. By default None
        interactions : tp.Optional[pd.DataFrame], optional
            DataFrame with users interactions history for u2i recos. Should have Columns ['user_id', 'item_id'].
            For i2i recos should be set to None. By default None
        n_add_random_requests : int, optional
            Number of random requests to display recos for. In case of u2i this is a number of random users
            to show recos for. By default 0
        remove_exceeding_data : tp.Optional[bool], optional
            Remove all data from all dataframes which is not needed. Only data necessary for selected requests
            will remain. By default True
        auto_display : bool, optional
            Display widgets app after initialisation, by default True
        reco_cols : tp.Optional[list], optional
            Columns to show in app from recos dataframe ('rank', 'score' or any other), by default None
        item_data_cols : tp.Optional[list], optional
            Columns to show in app from item_data dataframe, by default None
        item_df_renaming : tp.Optional[tp.Dict[str, str]], optional
            Renaming of Columns if necessary (reco_cols and item_data_cols), by default None
        item_df_formatters : tp.Optional[tp.Dict[str, tp.Callable]], optional
            Dict {column name: a} where a is a function called with the value of an individual cell in item_df
            when it is displayed. Can be used to display images, links and make any other transformations
            from DataFrame values to actual html code for display. By default None
        """
        super().__init__(
            interactions=interactions,
            recos=recos,
            requests_dict=requests_dict,
            item_data=item_data,
            n_add_random_requests=n_add_random_requests,
            remove_exceeding_data=remove_exceeding_data,
            is_u2i=is_u2i,
        )
        self.reco_cols = reco_cols
        self.item_data_cols = item_data_cols if item_data_cols is not None else []
        if Columns.Item not in self.item_data_cols:
            self.item_data_cols.append(Columns.Item)
        self.item_df_renaming = item_df_renaming if item_df_renaming is not None else {}
        self.item_data_formatters = item_df_formatters if item_df_formatters is not None else {}
        if auto_display:
            self.display()

    def _get_html_repr(
        self,
        items_list: np.ndarray,
        request_id: tp.Optional[int],
        model_name: tp.Optional[str],
    ) -> str:
        """Return html representation of info about items in `items_list` in string format"""
        if len(items_list) > 0:
            if isinstance(self.item_data, pd.DataFrame):
                item_df = pd.DataFrame(items_list, columns=[Columns.Item])
                item_df = item_df.join(self.item_data.set_index(Columns.Item), on=Columns.Item, how="left")
                if request_id is not None and self.reco_cols is not None:
                    # request_id is provided only for display with self.reco_cols
                    request_recos = self.recos[
                        (self.recos[self.request_colname] == request_id) & (self.recos[Columns.Model] == model_name)
                    ]
                    request_recos.set_index(Columns.Item, inplace=True)
                    request_recos = request_recos[self.reco_cols]
                    item_df = item_df.join(request_recos, on=Columns.Item, how="left")
                    item_df_columns = self.item_data_cols + self.reco_cols
                else:
                    item_df_columns = self.item_data_cols
            else:
                raise TypeError("Item data was not specified")
            item_df = item_df[item_df_columns]
            item_df.rename(columns=self.item_df_renaming, inplace=True)
            html_repr = (
                item_df.to_html(
                    escape=False,
                    index=False,
                    formatters=self.item_data_formatters,
                    max_rows=20,
                    border=0,
                )
                .replace("<td>", '<td align="center">')
                .replace("<th>", '<th style="text-align: center; min-width: 100px;">')
            )
            return html_repr
        return "No items"

    def _get_items_tab(
        self,
        items_list: np.ndarray,
        title: str,
        request_id: tp.Optional[int],
        model_name: tp.Optional[str],
    ) -> widget_selectioncontainer.Tab:
        """Get visual Tab with info about items in `items_list`"""
        items_tab = widgets.Tab()
        items_tab.children = [widgets.HTML(value=self._get_html_repr(items_list, request_id, model_name))]
        items_tab.set_title(index=0, title=title)
        return items_tab

    def _display_tab_for_request(self, request_name: str, items_type: str, model_name: str = "") -> None:
        """
        Display visual Tab with info about items for `request_name` depending on `items_type` from possible
        options: `viewed` or `recos`
        """
        request_id = self.requests_dict[request_name]
        if items_type == ItemTypes.Viewed:
            items_list = self.get_viewed_items_for_request(request_id)
        elif items_type == ItemTypes.Recommended:
            items_list = self.get_recos_for_request(request_id, model_name)
        else:
            raise ValueError(f"Unknown items_type: {items_type}")
        if self.reco_cols is not None and items_type == ItemTypes.Recommended:
            display(
                self._get_items_tab(
                    items_list,
                    title=items_type,
                    request_id=request_id,
                    model_name=model_name,
                )
            )
        else:
            display(self._get_items_tab(items_list, title=items_type, request_id=None, model_name=None))

    def _display_viewed(self, request_name: str) -> None:
        """Display viewed items for `request_name`"""
        self._display_tab_for_request(request_name, items_type=ItemTypes.Viewed)

    def _display_recos(self, request_name: str, model_name: str) -> None:
        """Display recommended items for `request_name` from model `model_name`"""
        self._display_tab_for_request(request_name, items_type=ItemTypes.Recommended, model_name=model_name)

    def _display_request_id(self, request_name: str) -> None:
        """Display request_id for `request_name`"""
        request_id = self.requests_dict[request_name]
        display(widgets.HTML(value=f"{self.request_colname} {request_id}"))

    def _display_model_name(self, model_name: str) -> None:
        """Display model_name"""
        display(widgets.HTML(value=f"Model name: {model_name}"))

    def display(self) -> None:
        """Display Showcase widget"""
        request = widgets.ToggleButtons(
            options=self.get_request_names(),
            description=f"Select request {self.request_colname}:",
            disabled=False,
            button_style="warning",
        )
        request_id_out = widgets.interactive_output(self._display_request_id, {"request_name": request})
        viewed_out = widgets.interactive_output(self._display_viewed, {"request_name": request})
        model = widgets.ToggleButtons(
            options=self.model_names,
            description="Select model:",
            disabled=False,
            button_style="success",
        )
        model_name_out = widgets.interactive_output(self._display_model_name, {"model_name": model})
        recos_out = widgets.interactive_output(self._display_recos, {"request_name": request, "model_name": model})

        display(widgets.VBox([request, request_id_out, viewed_out, model, model_name_out, recos_out]))
