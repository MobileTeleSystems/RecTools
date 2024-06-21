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
LEGEND_TITLE = "Models"


class MetricsApp:
    """
    Jupyter widgets app for metric visualization and comparison.
    Do not create instances of this class directly. Use `MetricsApp.construct` method instead.
    """

    def __init__(
        self,
        models_metrics: pd.DataFrame,
        models_metadata: tp.Optional[pd.DataFrame] = None,
        show_legend: bool = True,
        auto_display: bool = True,
        scatter_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        self.models_metrics = models_metrics
        self._validate_models_metrics()

        if models_metadata is not None:
            self.models_metadata: pd.DataFrame = models_metadata
        else:
            self.models_metadata = models_metrics[Columns.Model].drop_duplicates().to_frame()
        self._validate_models_metadata()

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
                - `Columns.Models` - model names
                - Any other columns with additional information
        show_legend : bool, default True
            Specifies whether to display the chart legend.
        auto_display : bool, default True
            Automatically displays the widgets immediately after initialization.
        scatter_kwargs : tp.Optional[tp.Dict[str, tp.Any]], optional, default None
            Additional arguments from `plotly.express.scatter`

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
        return cls(models_metrics, models_metadata, show_legend, auto_display, scatter_kwargs)

    @property
    @lru_cache
    def metric_names(self) -> tp.List[str]:
        """List of metric column names from the `models_metrics`."""
        non_metric_columns = {Columns.Split, Columns.Model}
        return [col for col in self.models_metrics.columns if col not in non_metric_columns]

    @property
    @lru_cache
    def meta_names(self) -> tp.List[str]:
        """List of metadata columns names from `models_metadata`"""
        return [col for col in self.models_metadata.columns if col != Columns.Model]

    @property
    @lru_cache
    def model_names(self) -> tp.List[str]:
        """Sorted list of model names from `models_metrics`."""
        return sorted(self.models_metrics[Columns.Model].unique())

    @property
    def fold_ids(self) -> tp.List[int]:
        """Sorted list of fold identifiers from the `models_metrics`."""
        return sorted(self.models_metrics[Columns.Split].unique())

    def _validate_models_metrics(self) -> None:
        if not isinstance(self.models_metrics, pd.DataFrame):
            raise ValueError("Incorrect input type. `metrics_data` should be a DataFrame")
        if Columns.Split not in self.models_metrics.columns:
            raise KeyError("Missing `Split` column in `metrics_data` DataFrame")
        if Columns.Model not in self.models_metrics.columns:
            raise KeyError("Missing `Model`` column in `metrics_data` DataFrame")
        if len(self.models_metrics.columns) < 3:
            raise KeyError("`metrics_data` DataFrame assumed to have at least one metric column")

    def _validate_models_metadata(self) -> None:
        if not isinstance(self.models_metadata, pd.DataFrame):
            raise ValueError("Incorrect input type. `models_metadata` should be a DataFrame")
        if Columns.Model not in self.models_metadata.columns:
            raise KeyError("Missing `Model`` column in `models_metadata` DataFrame")
        if self.models_metadata[Columns.Model].nunique() != len(self.models_metadata):
            raise ValueError("Found ambiguous values in `Model` column of `models_metadata`")

    @lru_cache
    def _make_chart_data(self, fold_number: int) -> pd.DataFrame:
        return (
            self.models_metrics[self.models_metrics[Columns.Split] == fold_number]
            .drop(columns=Columns.Split)
            .merge(self.models_metadata, on=Columns.Model, how="left")
        )

    @lru_cache
    def _make_chart_data_avg(self) -> pd.DataFrame:
        return (
            self.models_metrics.drop(columns=Columns.Split)
            .groupby(Columns.Model, sort=False)
            .mean()
            .reset_index(drop=False)
            .merge(self.models_metadata, on=Columns.Model, how="left")
        )

    def _create_chart(self, data: pd.DataFrame, metric_x: str, metric_y: str, color: str) -> go.Figure:
        scatter_kwargs = {
            "width": WIDGET_WIDTH,
            "height": WIDGET_HEIGHT,
        }
        scatter_kwargs.update(self.scatter_kwargs)
        fig = px.scatter(
            data,
            x=metric_x,
            y=metric_y,
            color=color,
            category_orders={color: sorted(data[color].unique())},
            symbol=Columns.Model,
            **scatter_kwargs,
        )
        layout_params = {
            "margin": {"t": TOP_CHART_MARGIN},
            "legend_title": LEGEND_TITLE,
            "showlegend": self.show_legend,
        }
        fig.update_layout(layout_params)
        return fig

    def _update_chart(
        self,
        fig_widget: go.FigureWidget,
        metric_x: widgets.Dropdown,
        metric_y: widgets.Dropdown,
        use_avg: widgets.Checkbox,
        fold_i: widgets.Dropdown,
        meta_feature: widgets.Dropdown,
        use_meta: widgets.Checkbox,
    ) -> None:  # pragma: no cover
        data = self._make_chart_data_avg() if use_avg.value else self._make_chart_data(fold_i.value)
        color_clmn = meta_feature.value if use_meta.value else Columns.Model

        # Ensuring that the color mapping treats the data as categorical
        if use_meta.value:
            data[color_clmn] = data[color_clmn].astype(str)

        scatter = self._create_chart(data, metric_x.value, metric_y.value, color_clmn)
        with fig_widget.batch_update():
            for i, trace in enumerate(scatter.data):
                fig_widget.data[i].x = trace.x
                fig_widget.data[i].y = trace.y
                fig_widget.data[i].marker = trace.marker

        fig_widget.layout.update(scatter.layout)
        self.fig.layout.margin = None  # keep separate chart non-truncated

    def _update_fold_visibility(self, use_avg: widgets.Checkbox, fold_i: widgets.Dropdown) -> None:
        fold_i.layout.visibility = "hidden" if use_avg.value else "visible"

    def _update_meta_visibility(self, use_meta: widgets.Checkbox, meta_feature: widgets.Dropdown) -> None:
        meta_feature.layout.visibility = "hidden" if not use_meta.value else "visible"

    def display(self) -> None:
        """Display MetricsApp widget"""
        metric_x = widgets.Dropdown(description="Metric X:", value=self.metric_names[0], options=self.metric_names)
        metric_y = widgets.Dropdown(description="Metric Y:", value=self.metric_names[-1], options=self.metric_names)
        use_avg = widgets.Checkbox(description="Avg folds", value=True)
        fold_i = widgets.Dropdown(description="Fold number:", value=self.fold_ids[0], options=self.fold_ids)
        use_meta = widgets.Checkbox(description="Use metadata colors", value=False)
        meta_feature = widgets.Dropdown(
            description="Feature:", value=self.meta_names[0] if self.meta_names else None, options=self.meta_names
        )

        data = self._make_chart_data_avg() if use_avg.value else self._make_chart_data(fold_i.value)
        self.fig = self._create_chart(data, metric_x.value, metric_y.value, Columns.Model)
        fig_widget = go.FigureWidget(data=self.fig.data, layout=self.fig.layout)

        def update(event: tp.Callable[..., tp.Any]) -> None:  # pragma: no cover
            self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i, meta_feature, use_meta)
            self._update_fold_visibility(use_avg, fold_i)
            self._update_meta_visibility(use_meta, meta_feature)

        metric_x.observe(update, "value")
        metric_y.observe(update, "value")
        use_avg.observe(update, "value")
        fold_i.observe(update, "value")
        use_meta.observe(update, "value")
        meta_feature.observe(update, "value")

        tab = widgets.Tab()

        if self.meta_names:
            tab.children = [
                widgets.VBox(
                    [
                        widgets.HBox([use_avg, fold_i]),
                        widgets.HBox([metric_x, metric_y]),
                    ]
                ),
                widgets.VBox([widgets.HBox([use_meta, meta_feature])]),
            ]
            tab.set_title(0, "Data & Metrics")
            tab.set_title(1, "Metadata colors")
        else:
            tab.children = [
                widgets.VBox(
                    [
                        widgets.HBox([use_avg, fold_i]),
                        widgets.HBox([metric_x, metric_y]),
                    ]
                )
            ]
            tab.set_title(0, "Data & Metrics")

        display(widgets.VBox([tab, fig_widget]))

        self._update_fold_visibility(use_avg, fold_i)
        self._update_meta_visibility(use_meta, meta_feature)
        self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i, meta_feature, use_meta)
