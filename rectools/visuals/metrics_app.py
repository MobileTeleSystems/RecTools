import typing as tp

import ipywidgets as widgets
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from IPython.display import display

from rectools import Columns

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600


class MetricsApp:
    """
    Jupyter widgets app for metric visualization and comparison.
    Do not create instances of this class directly. Use `MetricsApp.construct` method instead.
    """

    def __init__(
        self,
        models_metrics: pd.DataFrame,
        show_legend: bool = True,
        auto_display: bool = True,
        plotly_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        self.models_metrics = models_metrics
        self.show_legend = show_legend
        self.auto_display = auto_display
        self.plotly_kwargs = plotly_kwargs if plotly_kwargs is not None else {}

        self.fig = go.Figure()

        self._validate_input_data()
        if self.auto_display:
            self.display()

    @classmethod
    def construct(
        cls,
        models_metrics: pd.DataFrame,
        show_legend: bool = True,
        auto_display: bool = True,
        plotly_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> "MetricsApp":
        """Construct a widget-based application for visualizing metrics from the provided dataframe.

        Parameters
        ----------
        models_metrics : pd.DataFrame
            A pandas DataFrame containing metrics for visualization. Required columns:
                - `Columns.Models` - model names
                - `Columns.Split` - fold number
                - Any other numeric columns which represent metric values
        show_legend : bool, default True
            Specifies whether to display the chart legend.
        auto_display : bool, default True
            Automatically displays the widgets immediately after initialization.
        plotly_kwargs : tp.Optional[tp.Dict[str, tp.Any]], optional, default None
            Additional arguments for customizing the `px.scatter` plot.

        Returns
        -------
        MetricsApp
            An instance of `MetricsApp`, providing interactive Jupyter widget for metric visualization.
        """
        return cls(models_metrics, show_legend, auto_display, plotly_kwargs)

    def _validate_input_data(self) -> None:
        if not isinstance(self.models_metrics, pd.DataFrame):
            raise ValueError("Incorrect input type. `metrics_data` should be a DataFrame")
        if Columns.Split not in self.models_metrics.columns:
            raise KeyError("Missing 'Split' column in `metrics_data` DataFrame")
        if Columns.Model not in self.models_metrics.columns:
            raise KeyError("Missing 'Model' column in `metrics_data` DataFrame")
        if len(self.models_metrics.columns) < 3:
            raise KeyError("`metrics_data` DataFrame assumed to have at least one metric column")

    @property
    def metric_names(self) -> tp.List[str]:
        """List of metric column names from the `models_metrics`."""
        non_metric_columns = {Columns.Split, Columns.Model}
        return [col for col in self.models_metrics.columns if col not in non_metric_columns]

    @property
    def fold_ids(self) -> tp.List[int]:
        """Sorted list of unique fold identifiers from the `models_metrics`."""
        return sorted(self.models_metrics[Columns.Split].unique())

    @property
    def n_folds(self) -> int:
        """Total number of unique folds available in the `models_metrics`."""
        return len(self.fold_ids)

    def _make_chart_data(self, fold_number: int) -> pd.DataFrame:
        return self.models_metrics[self.models_metrics[Columns.Split] == fold_number].drop(columns=Columns.Split)

    def _make_chart_data_avg(self) -> pd.DataFrame:
        return self.models_metrics.drop(columns=Columns.Split).groupby(Columns.Model, sort=False).mean().reset_index()

    def _toggle_fold_number_visibility(self, fold: widgets.Dropdown, use_avg: widgets.Checkbox) -> None:
        fold.layout.visibility = "hidden" if use_avg.value else "visible"

    def _update_chart(
        self,
        scatter_chart: widgets.Output,
        metric_x: widgets.Dropdown,
        metric_y: widgets.Dropdown,
        use_avg: widgets.Checkbox,
        fold: widgets.Dropdown,
    ) -> None:
        with scatter_chart:
            scatter_chart.clear_output(wait=True)
            chart_data = self._make_chart_data_avg() if use_avg.value else self._make_chart_data(fold.value)
            self.fig = px.scatter(
                data_frame=chart_data,
                x=metric_x.value,
                y=metric_y.value,
                color=Columns.Model,
                width=self.plotly_kwargs.get("width", DEFAULT_WIDTH),
                height=self.plotly_kwargs.get("height", DEFAULT_HEIGHT),
                **self.plotly_kwargs,
            )
            self.fig.layout.update(showlegend=self.show_legend)
            self.fig.show()

    def _attach_callbacks(
        self,
        scatter_chart: widgets.Output,
        metric_x: widgets.Dropdown,
        metric_y: widgets.Dropdown,
        use_avg: widgets.Checkbox,
        fold: widgets.Dropdown,
    ) -> None:
        metric_x.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")
        metric_y.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")
        use_avg.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")
        use_avg.observe(lambda upd: self._toggle_fold_number_visibility(fold, use_avg), "value")
        fold.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")

    def display(self) -> None:
        """Display full MetricsApp widget"""
        scatter_chart = widgets.Output()
        metric_x = widgets.Dropdown(description="Metric X:", options=self.metric_names, value=self.metric_names[0])
        metric_y = widgets.Dropdown(description="Metric Y:", options=self.metric_names, value=self.metric_names[-1])
        use_avg = widgets.Checkbox(description="Avg folds", value=True)
        fold = widgets.Dropdown(description="Fold number:", options=self.fold_ids, value=self.fold_ids[0])

        self._attach_callbacks(scatter_chart, metric_x, metric_y, use_avg, fold)

        # trigger first chart update
        self._toggle_fold_number_visibility(fold, use_avg)
        self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold)

        container_folds = widgets.HBox(children=[use_avg, fold])
        container_metrics = widgets.HBox(children=[metric_x, metric_y])
        display(widgets.VBox([container_folds, container_metrics, scatter_chart]))
