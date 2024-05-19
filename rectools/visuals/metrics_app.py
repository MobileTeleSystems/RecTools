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

    def _update_fig(self, metric_x: str, metric_y: str, avg_folds: bool, fold_number: tp.Optional[int]) -> None:
        if avg_folds:
            chart_data = self._make_chart_data_avg()
        elif fold_number is not None:
            chart_data = self._make_chart_data(fold_number)
        else:
            raise ValueError("`fold_number` must be provided when `avg_folds` is False")

        self.fig = px.scatter(
            chart_data,
            x=metric_x,
            y=metric_y,
            color=Columns.Model,
            **self.plotly_kwargs,
        )
        self.fig.layout.update(showlegend=self.show_legend)

    def display(self) -> None:
        """TODO: add dosctring"""
        metric_x = widgets.Dropdown(description="Metric X:", value=self.metric_names[0], options=self.metric_names)
        metric_y = widgets.Dropdown(description="Metric Y:", value=self.metric_names[-1], options=self.metric_names)
        container_metrics = widgets.HBox(children=[metric_x, metric_y])

        use_avg = widgets.Checkbox(description="Avg folds", value=True)
        fold_number = widgets.Dropdown(description="Fold number:", value=0, options=list(range(self.n_folds)))

        def toggle_fold_number_visibility(*args: tp.Any) -> None:
            fold_number.layout.visibility = "hidden" if use_avg.value else "visible"

        use_avg.observe(toggle_fold_number_visibility, "value")
        toggle_fold_number_visibility()

        container_folds = widgets.HBox(children=[use_avg, fold_number])
        scatter_chart = widgets.Output()

        def update_chart(*args: tp.Any) -> None:
            with scatter_chart:
                scatter_chart.clear_output(wait=True)
                self._update_fig(metric_x.value, metric_y.value, use_avg.value, fold_number.value)
                self.fig.show()

        metric_x.observe(update_chart, "value")
        metric_y.observe(update_chart, "value")
        use_avg.observe(update_chart, "value")
        fold_number.observe(update_chart, "value")

        update_chart()
        display(widgets.VBox([container_folds, container_metrics, scatter_chart]))
