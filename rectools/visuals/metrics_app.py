import typing as tp

import ipywidgets as widgets
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

from rectools import Columns


class MetricsApp:
    """
    Jupyter widgets app for metric visualization and comparison.
    Do not create instances of this class directly. Use `MetricsApp.construct` method instead.
    """

    def __init__(
        self,
        df_metrics_data: pd.DataFrame,
        chart_width: tp.Union[float, int] = 800,
        chart_height: tp.Union[float, int] = 600,
        show_legend: bool = False,
        auto_display: bool = True,
        color_discrete_sequence: tp.Optional[tp.List[str]] = None,
    ):
        self.metrics_data = df_metrics_data
        self.chart_width = chart_width
        self.chart_height = chart_height
        self.show_legend = show_legend
        self.auto_display = auto_display
        self.color_discrete_sequence = (
            px.colors.qualitative.Plotly if color_discrete_sequence is None else color_discrete_sequence
        )

        self._validate_input_data()
        self._validate_input_patameters()
        if self.auto_display:
            self.display()

    @classmethod
    def construct(
        cls,
        df_metrics_data: pd.DataFrame,
        chart_width: tp.Union[float, int] = 800,
        chart_height: tp.Union[float, int] = 600,
        show_legend: bool = False,
        auto_display: bool = True,
        color_discrete_sequence: tp.Optional[tp.List[str]] = None,
    ) -> "MetricsApp":
        """TODO: add docstring"""
        return cls(df_metrics_data, chart_width, chart_height, show_legend, auto_display, color_discrete_sequence)

    def _validate_input_data(self) -> None:
        if not isinstance(self.metrics_data, pd.DataFrame):
            raise ValueError("Incorrect input type. `metrics_data` should be a DataFrame")
        if Columns.Split not in self.metrics_data.columns:
            raise KeyError("Missing 'Split' column in `metrics_data` DataFrame")
        if Columns.Model not in self.metrics_data.columns:
            raise KeyError("Missing 'Model' column in `metrics_data` DataFrame")
        if len(self.metrics_data.columns) < 3:
            raise KeyError("`metrics_data` DataFrame assumed to have at least one metric column")

    def _validate_input_patameters(self) -> None:
        if self.chart_width < 10:
            raise ValueError(
                "Incorrect `chart_width` value. `chart_width` should be a float or int in the interval [10, inf]"
            )
        if self.chart_height < 10:
            raise ValueError(
                "Incorrect `chart_height` value. `chart_height` should be a float or int in the interval [10, inf]"
            )

    def _get_metric_names(self) -> tp.List[str]:
        non_metric_columns = {Columns.Split, Columns.Model}
        return [col for col in self.metrics_data.columns if col not in non_metric_columns]

    def _get_folds_number(self) -> int:
        return self.metrics_data[Columns.Split].nunique()

    def _make_chart_data(self, fold_number: int) -> pd.DataFrame:
        return self.metrics_data[self.metrics_data[Columns.Split] == fold_number].drop(columns=Columns.Split)

    def _make_chart_data_avg(self) -> pd.DataFrame:
        return self.metrics_data.drop(columns=Columns.Split).groupby(Columns.Model, sort=False).mean().reset_index()

    def _create_chart(self, avg_folds: bool, fold_number: int, metric_x: str, metric_y: str) -> go.Figure:
        chart_data = self._make_chart_data_avg() if avg_folds else self._make_chart_data(fold_number)
        fig = px.scatter(
            chart_data,
            x=metric_x,
            y=metric_y,
            color=Columns.Model,
            width=self.chart_width,
            height=self.chart_height,
            color_discrete_sequence=self.color_discrete_sequence,
        )
        fig.layout.update(showlegend=self.show_legend)
        return fig

    def display(self) -> None:
        """TODO: add dosctring"""
        metrics_list = self._get_metric_names()
        n_splits = self._get_folds_number()

        metric_x = widgets.Dropdown(description="Metric X:", value=metrics_list[0], options=metrics_list)
        metric_y = widgets.Dropdown(description="Metric Y:", value=metrics_list[-1], options=metrics_list)
        container_metrics = widgets.HBox(children=[metric_x, metric_y])

        use_avg = widgets.Checkbox(description="Avg folds", value=True)
        fold_number = widgets.Dropdown(description="Fold number:", value=0, options=list(range(n_splits)))
        container_folds = widgets.HBox(children=[use_avg, fold_number])

        scatter_chart = widgets.Output()

        def update_chart(*args: tp.Any) -> None:
            with scatter_chart:
                scatter_chart.clear_output(wait=True)
                fig = self._create_chart(use_avg.value, fold_number.value, metric_x.value, metric_y.value)
                fig.show()

        metric_x.observe(update_chart, "value")
        metric_y.observe(update_chart, "value")
        use_avg.observe(update_chart, "value")
        fold_number.observe(update_chart, "value")

        update_chart()
        display(widgets.VBox([container_folds, container_metrics, scatter_chart]))
