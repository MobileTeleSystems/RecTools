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
        df_metrics_data: pd.DataFrame,
        show_legend: bool = True,
        auto_display: bool = True,
        plotly_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        self.metrics_data = df_metrics_data
        self.show_legend = show_legend
        self.auto_display = auto_display
        self.default_plotly_kwargs = {"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT}
        self.plotly_kwargs = {**self.default_plotly_kwargs, **(plotly_kwargs if plotly_kwargs is not None else {})}

        self.fig = go.Figure()

        self._validate_input_data()
        if self.auto_display:
            self.display()

    @classmethod
    def construct(
        cls,
        df_metrics_data: pd.DataFrame,
        show_legend: bool = True,
        auto_display: bool = True,
        plotly_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> "MetricsApp":
        """TODO: add docstring"""
        return cls(df_metrics_data, show_legend, auto_display, plotly_kwargs)

    def _validate_input_data(self) -> None:
        if not isinstance(self.metrics_data, pd.DataFrame):
            raise ValueError("Incorrect input type. `metrics_data` should be a DataFrame")
        if Columns.Split not in self.metrics_data.columns:
            raise KeyError("Missing 'Split' column in `metrics_data` DataFrame")
        if Columns.Model not in self.metrics_data.columns:
            raise KeyError("Missing 'Model' column in `metrics_data` DataFrame")
        if len(self.metrics_data.columns) < 3:
            raise KeyError("`metrics_data` DataFrame assumed to have at least one metric column")

    @property
    def metric_names(self) -> tp.List[str]:
        """TODO: add docstring"""
        non_metric_columns = {Columns.Split, Columns.Model}
        return [col for col in self.metrics_data.columns if col not in non_metric_columns]

    @property
    def n_folds(self) -> int:
        """TODO: add docstring"""
        return self.metrics_data[Columns.Split].nunique()

    def _make_chart_data(self, fold_number: int) -> pd.DataFrame:
        return self.metrics_data[self.metrics_data[Columns.Split] == fold_number].drop(columns=Columns.Split)

    def _make_chart_data_avg(self) -> pd.DataFrame:
        return self.metrics_data.drop(columns=Columns.Split).groupby(Columns.Model, sort=False).mean().reset_index()

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
                chart_data, x=metric_x.value, y=metric_y.value, color=Columns.Model, **self.plotly_kwargs
            )
            self.fig.layout.update(showlegend=self.show_legend)
            self.fig.show()

    def _toggle_fold_number_visibility(self, fold: widgets.Dropdown, use_avg: widgets.Checkbox) -> None:
        fold.layout.visibility = "hidden" if use_avg.value else "visible"

    def display(self) -> None:
        """TODO: add dosctring"""
        metric_x = widgets.Dropdown(description="Metric X:", options=self.metric_names, value=self.metric_names[0])
        metric_y = widgets.Dropdown(description="Metric Y:", options=self.metric_names, value=self.metric_names[-1])
        use_avg = widgets.Checkbox(description="Avg folds", value=True)
        fold = widgets.Dropdown(description="Fold number:", options=list(range(self.n_folds)), value=0)
        scatter_chart = widgets.Output()

        metric_x.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")
        metric_y.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")
        use_avg.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")
        use_avg.observe(lambda upd: self._toggle_fold_number_visibility(fold, use_avg), "value")
        fold.observe(lambda upd: self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold), "value")

        container_metrics = widgets.HBox(children=[metric_x, metric_y])
        container_folds = widgets.HBox(children=[use_avg, fold])
        scatter_chart = widgets.Output()

        # trigger first chart update
        self._toggle_fold_number_visibility(fold, use_avg)
        self._update_chart(scatter_chart, metric_x, metric_y, use_avg, fold)

        display(widgets.VBox([container_folds, container_metrics, scatter_chart]))
