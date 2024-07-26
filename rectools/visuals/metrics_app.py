#  Copyright 2024 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import typing as tp
from functools import lru_cache

import ipywidgets as widgets
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from IPython.display import display

from rectools import Columns

WIDGET_WIDTH = 800
WIDGET_HEIGHT = 500
TOP_CHART_MARGIN = 20
DEFAULT_LEGEND_TITLE = "model name"
NAN_COLOR = "grey"
META_MODEL_SEP = " ❘ "
META_MODEL_SEP_REPLACEMENT = " ❘; "


class MetricsApp:
    """
    Jupyter widgets app for metric visualization and comparison.
    Do not create instances of this class directly. Use `MetricsApp.construct` method instead.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metric_names: tp.List[str],
        meta_names: tp.List[str],
        show_legend: bool = True,
        auto_display: bool = True,
        scatter_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        # `self.data` - merged data created from `models_metrics` and `models_metadata`
        # Required columns: `Columns.Models` and `Columns.Split`
        # Any other columns should be in `metric_names` or `meta_names` list
        # Columns from the `metric_names` list should be numeric type
        # Columns from the `meta_names` list could be any type
        self.data = data
        self.metric_names = metric_names
        self.meta_names = meta_names
        self.show_legend = show_legend
        self.auto_display = auto_display
        self.scatter_kwargs = scatter_kwargs if scatter_kwargs is not None else {}
        self.fig = go.Figure()

        if self.auto_display:
            self.display()

    @classmethod
    def construct(
        cls,
        models_metrics: pd.DataFrame,
        models_metadata: tp.Optional[pd.DataFrame] = None,
        show_legend: bool = True,
        auto_display: bool = True,
        scatter_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> "MetricsApp":
        r"""
        Construct interactive widget for metric-to-metric trade-off analysis.

        Parameters
        ----------
        models_metrics : pd.DataFrame
            A pandas DataFrame containing metrics for visualization. Required columns:
                - `Columns.Models` - model names
                - `Columns.Split` - fold number
                - Any other numeric columns which represent metric values
        models_metadata: tp.Optional[pd.DataFrame], optional, default None
            An optional pandas DataFrame containing any models metadata (hyperparameters, training info, etc.).
            Used for alternative ways of coloring scatterplot points.
            Required columns:
                - `Columns.Model` - model names
                - Any other columns with additional information
        show_legend : bool, default True
            Specifies whether to display the chart legend.
        auto_display : bool, default True
            Automatically displays the widgets immediately after initialization.
        scatter_kwargs : tp.Optional[tp.Dict[str, tp.Any]], optional, default None
            Additional arguments for `plotly.express.scatter`

        Returns
        -------
        MetricsApp
            An instance of `MetricsApp`, providing interactive Jupyter widget for metric visualization.

        Examples
        --------
        Create interactive widget

        >>> metrics_df = pd.DataFrame(
        ...    {
        ...        Columns.Model: ["Model1", "Model2", "Model1", "Model2", "Model1", "Model2"],
        ...        Columns.Split: [0, 0, 1, 1, 2, 2],
        ...        "prec@10": [0.031, 0.025, 0.027, 0.21, 0.031, 0.033],
        ...        "recall@10": [0.041, 0.045, 0.055, 0.08, 0.036, 0.021],
        ...        "novelty@10": [2.6, 11.3, 4.3, 9.8, 3.3, 11.2],
        ...    })
        >>> # Optional metainfo about models
        >>> metadata_df = pd.DataFrame(
        ...    {
        ...        Columns.Model: ["Model1", "Model2"],
        ...        "factors": [64, 32],
        ...        "regularization": [0.05, 0.05],
        ...        "alpha": [2.0, 0.5],
        ...    })
        >>> app = MetricsApp.construct(
        ...    models_metrics=metrics_df,
        ...    models_metadata=metadata_df,
        ...    show_legend=True,
        ...    auto_display=False,
        ...    scatter_kwargs={"width": 800, "height": 600})

        Get plotly chart from the current widget state

        >>> fig = app.fig
        >>> fig = fig.update_layout(title="Metrics comparison")
        """
        cls._validate_models_metrics_base(models_metrics)
        cls._validate_models_metrics_split(models_metrics)
        if models_metadata is None:
            models_metadata = models_metrics[Columns.Model].drop_duplicates().to_frame()
        cls._validate_models_metadata(models_metadata)

        merged_data = models_metrics.merge(models_metadata, on=Columns.Model, how="left")
        merged_data = merged_data.replace(META_MODEL_SEP, META_MODEL_SEP_REPLACEMENT, regex=True)

        metric_names = [col for col in models_metrics.columns if col not in {Columns.Split, Columns.Model}]
        meta_names = [col for col in models_metadata.columns if col != Columns.Model]

        return cls(merged_data, metric_names, meta_names, show_legend, auto_display, scatter_kwargs)

    @property
    @lru_cache
    def model_names(self) -> tp.List[str]:
        """Sorted list of model names from `models_metrics`."""
        return sorted(self.data[Columns.Model].unique())

    @property
    @lru_cache
    def fold_ids(self) -> tp.Optional[tp.List[int]]:
        """Sorted list of fold identifiers from the `models_metrics`."""
        if Columns.Split in self.data.columns:
            return sorted(self.data[Columns.Split].unique())
        return None

    @staticmethod
    def _validate_models_metrics_base(models_metrics: pd.DataFrame) -> None:
        metric_columns = list(set(models_metrics.columns) - {Columns.Model, Columns.Split})
        if Columns.Model not in models_metrics.columns:
            raise KeyError("Missing `Model` column in `metrics_data` DataFrame")
        if not metric_columns:
            raise KeyError("`metrics_data` DataFrame assumed to have at least one metric column")
        if models_metrics[Columns.Model].isnull().any():
            raise ValueError("Found NaN values in `Model` column of `metrics_data`")
        if Columns.Split in models_metrics.columns and models_metrics[Columns.Split].isnull().any():
            raise ValueError("Found NaN values in `Split` column of `metrics_data`")
        if Columns.Split not in models_metrics.columns and models_metrics[Columns.Model].nunique() != len(
            models_metrics
        ):
            raise ValueError("Each `Model` value in the `metrics_data` DataFrame must be unique")
        if len(models_metrics[metric_columns].select_dtypes(include="number").columns) != len(metric_columns):
            raise ValueError("All metrics columns should be numeric")

    @staticmethod
    def _validate_models_metrics_split(models_metrics: pd.DataFrame) -> None:
        if Columns.Split not in models_metrics.columns:
            return
        # Validate that each model have same folds names
        splits = models_metrics.groupby(Columns.Model)[Columns.Split].apply(frozenset)
        splits_set = set(splits)
        if len(splits_set) > 1:
            raise ValueError(f"All models must have the same splits. But now they are different: {splits_set}")
        # Validate that each row have unique model and folds names
        if models_metrics.duplicated(subset=[Columns.Model, Columns.Split], keep=False).any():
            raise ValueError("Each pair of `Model` and `Split` values in the `metrics_data` DataFrame must be unique")

    @staticmethod
    def _validate_models_metadata(models_metadata: pd.DataFrame) -> None:
        if Columns.Model not in models_metadata.columns:
            raise KeyError("Missing `Model` column in `models_metadata` DataFrame")
        if models_metadata[Columns.Model].isnull().any():
            raise ValueError("Found NaN values in `Model` column")
        if models_metadata[Columns.Model].nunique() != len(models_metadata):
            raise ValueError("`Model` values of `models_metadata` should be unique`")

    @lru_cache
    def _make_chart_data_fold(self, fold_number: int) -> pd.DataFrame:
        return self.data[self.data[Columns.Split] == fold_number].reset_index(drop=True)

    @lru_cache
    def _make_chart_data_avg(self) -> pd.DataFrame:
        avg_data = self.data.groupby(Columns.Model).agg(
            {
                **{metric: "mean" for metric in self.metric_names},
                **{meta: "first" for meta in self.meta_names},
            }
        )
        avg_data = avg_data.reset_index()
        return avg_data

    @staticmethod
    @lru_cache
    def _split_to_meta_and_model(raw_string: str, sep: str = META_MODEL_SEP) -> tp.Tuple[str, str]:
        splitted_row = raw_string.split(sep, 1)
        if len(splitted_row) > 1:
            meta_value, model_name = splitted_row
            return meta_value, model_name
        return "", raw_string

    def _create_chart_figure(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        legend_title: str,
    ) -> go.Figure:  # pragma: no cover
        scatter_kwargs = {
            "width": WIDGET_WIDTH,
            "height": WIDGET_HEIGHT,
        }
        scatter_kwargs.update(self.scatter_kwargs)

        data = data.sort_values(by=color_col, ascending=True)
        data[color_col] = data[color_col].astype(str)  # to treat colors values as categorical

        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            symbol=Columns.Model,
            **scatter_kwargs,
        )
        # Set custom legend for meta info coloring
        if color_col != Columns.Model:
            for trace, meta_value, model_name in zip(fig.data, data[color_col], data[Columns.Model]):
                trace.name = f"{meta_value}{META_MODEL_SEP}{model_name}"
        fig.update_layout(margin={"t": TOP_CHART_MARGIN}, legend_title=legend_title, showlegend=self.show_legend)
        fig.update_coloraxes(showscale=False)
        return fig

    def _update_figure_widget(
        self,
        fig_widget: go.FigureWidget,
        metric_x: widgets.Dropdown,
        metric_y: widgets.Dropdown,
        use_avg: widgets.Checkbox,
        fold_i: widgets.Dropdown,
        meta_feature: widgets.Dropdown,
        use_meta: widgets.Checkbox,
    ) -> None:  # pragma: no cover
        chart_data = self._create_chart_data(use_avg, fold_i)
        color_col = meta_feature.value if use_meta.value else Columns.Model

        # Save dots symbols from the previous widget state
        # Remove metainfo from trace name. Thus we guarantee to map with traces from previous state
        model_name_to_symbol = {
            self._split_to_meta_and_model(trace.name)[1]: trace.marker.symbol for trace in self.fig.data
        }
        legend_title = (
            f"{meta_feature.value}{META_MODEL_SEP}{DEFAULT_LEGEND_TITLE}" if use_meta.value else DEFAULT_LEGEND_TITLE
        )
        self.fig = self._create_chart_figure(chart_data, metric_x.value, metric_y.value, color_col, legend_title)

        for trace in self.fig.data:
            model_name = self._split_to_meta_and_model(trace.name)[1]
            trace.marker.symbol = model_name_to_symbol[model_name]

        chart_data = chart_data.set_index(Columns.Model)
        with fig_widget.batch_update():
            for idx, trace in enumerate(self.fig.data):
                model_name = self._split_to_meta_and_model(trace.name)[1]
                if color_col != Columns.Model and pd.isna(chart_data.at[model_name, color_col]):
                    trace.marker.color = NAN_COLOR
                fig_widget.data[idx].x = trace.x
                fig_widget.data[idx].y = trace.y
                fig_widget.data[idx].marker.color = trace.marker.color
                fig_widget.data[idx].marker.symbol = trace.marker.symbol
                fig_widget.data[idx].text = trace.text
                fig_widget.data[idx].name = trace.name
                fig_widget.data[idx].legendgroup = trace.legendgroup
                fig_widget.data[idx].hoverinfo = trace.hoverinfo
                fig_widget.data[idx].hovertemplate = trace.hovertemplate

            fig_widget.layout = self.fig.layout
        self.fig.layout.margin = None  # keep separate chart non-truncated

    def _update_fold_visibility(self, use_avg: widgets.Checkbox, fold_i: widgets.Dropdown) -> None:
        fold_i.layout.visibility = "hidden" if use_avg.value else "visible"

    def _update_meta_visibility(self, use_meta: widgets.Checkbox, meta_feature: widgets.Dropdown) -> None:
        meta_feature.layout.visibility = "hidden" if not use_meta.value else "visible"

    def _create_chart_data(
        self, use_avg: widgets.Checkbox, fold_i: widgets.Dropdown
    ) -> pd.DataFrame:  # pragma: no cover
        if use_avg.value or fold_i.value is None:
            return self._make_chart_data_avg()
        return self._make_chart_data_fold(fold_i.value)

    def display(self) -> None:
        """Display MetricsApp widget"""
        metric_x = widgets.Dropdown(description="Metric X:", value=self.metric_names[0], options=self.metric_names)
        metric_y = widgets.Dropdown(
            description="Metric Y:",
            value=self.metric_names[min(1, len(self.metric_names) - 1)],
            options=self.metric_names,
        )
        use_avg = widgets.Checkbox(description="Average folds", value=True)
        fold_i = widgets.Dropdown(
            description="Fold number:",
            value=self.fold_ids[0] if self.fold_ids is not None else None,
            options=self.fold_ids if self.fold_ids is not None else [],
        )
        use_meta = widgets.Checkbox(description="Use metadata", value=False)
        meta_feature = widgets.Dropdown(
            description="Color by:",
            value=self.meta_names[0] if self.meta_names else None,
            options=self.meta_names,
        )

        # Initialize go.FigureWidget initial chart state
        if not self.fig.data:
            chart_data = self._create_chart_data(use_avg, fold_i)
            legend_title = f"{meta_feature.value}, {DEFAULT_LEGEND_TITLE}" if use_meta.value else DEFAULT_LEGEND_TITLE
            self.fig = self._create_chart_figure(
                chart_data, metric_x.value, metric_y.value, Columns.Model, legend_title
            )
            fig_widget = go.FigureWidget(data=self.fig.data, layout=self.fig.layout)

        def update(event: tp.Callable[..., tp.Any]) -> None:  # pragma: no cover
            self._update_figure_widget(fig_widget, metric_x, metric_y, use_avg, fold_i, meta_feature, use_meta)
            self._update_fold_visibility(use_avg, fold_i)
            self._update_meta_visibility(use_meta, meta_feature)

        metric_x.observe(update, "value")
        metric_y.observe(update, "value")
        use_avg.observe(update, "value")
        fold_i.observe(update, "value")
        use_meta.observe(update, "value")
        meta_feature.observe(update, "value")

        tab = widgets.Tab()

        metrics_vbox = widgets.VBox([widgets.HBox([metric_x, metric_y])])

        if self.meta_names:
            metadata_vbox = widgets.VBox([widgets.HBox([use_meta, meta_feature])])
            if self.fold_ids:
                metrics_vbox = widgets.VBox([widgets.HBox([use_avg, fold_i]), widgets.HBox([metric_x, metric_y])])
            tab.children = [metrics_vbox, metadata_vbox]
            tab.set_title(0, "Metrics")
            tab.set_title(1, "Metadata")
        else:
            if self.fold_ids:
                metrics_vbox = widgets.VBox([widgets.HBox([use_avg, fold_i]), widgets.HBox([metric_x, metric_y])])
            tab.children = [metrics_vbox]
            tab.set_title(0, "Metrics")

        display(widgets.VBox([tab, fig_widget]))

        self._update_fold_visibility(use_avg, fold_i)
        self._update_meta_visibility(use_meta, meta_feature)
        self._update_figure_widget(fig_widget, metric_x, metric_y, use_avg, fold_i, meta_feature, use_meta)
