import typing as tp

import ipywidgets as widgets
import pandas as pd
import plotly.graph_objs as go
from IPython.display import display

from rectools import Columns

WIDGET_WIDTH = 800
WIDGET_HEIGHT = 500
CHART_MARGIN = 20


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
    ):
        self.models_metrics = models_metrics
        self.show_legend = show_legend
        self.auto_display = auto_display
        self.fig = go.Figure()

        self._validate_input_data()

    @classmethod
    def construct(
        cls,
        models_metrics: pd.DataFrame,
        show_legend: bool = True,
        auto_display: bool = True,
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

        Returns
        -------
        MetricsApp
            An instance of `MetricsApp`, providing interactive Jupyter widget for metric visualization.
        """
        app = cls(models_metrics, show_legend, auto_display)
        if auto_display:
            app.display()
        return app

    @property
    def metric_names(self) -> tp.List[str]:
        """List of metric column names from the `models_metrics`."""
        non_metric_columns = {Columns.Split, Columns.Model}
        return [col for col in self.models_metrics.columns if col not in non_metric_columns]

    @property
    def model_names(self) -> tp.List[str]:
        """Sorted list of model names from `models_metrics`."""
        return sorted(self.models_metrics[Columns.Model].unique())

    @property
    def fold_ids(self) -> tp.List[int]:
        """Sorted list of fold identifiers from the `models_metrics`."""
        return sorted(self.models_metrics[Columns.Split].unique())

    @property
    def n_folds(self) -> int:
        """Total number of unique folds available in the `models_metrics`."""
        return len(self.fold_ids)

    def _validate_input_data(self) -> None:
        if not isinstance(self.models_metrics, pd.DataFrame):
            raise ValueError("Incorrect input type. `metrics_data` should be a DataFrame")
        if Columns.Split not in self.models_metrics.columns:
            raise KeyError("Missing `Split` column in `metrics_data` DataFrame")
        if Columns.Model not in self.models_metrics.columns:
            raise KeyError("Missing `Model`` column in `metrics_data` DataFrame")
        if len(self.models_metrics.columns) < 3:
            raise KeyError("`metrics_data` DataFrame assumed to have at least one metric column")

    def _make_chart_data(self, fold_number: int) -> pd.DataFrame:
        return self.models_metrics[self.models_metrics[Columns.Split] == fold_number].drop(columns=Columns.Split)

    def _make_chart_data_avg(self) -> pd.DataFrame:
        return self.models_metrics.drop(columns=Columns.Split).groupby(Columns.Model, sort=False).mean().reset_index()

    def display(self) -> None:
        """Display MetricsApp widget"""
        metric_x = widgets.Dropdown(description="Metric X:", value=self.metric_names[0], options=self.metric_names)
        metric_y = widgets.Dropdown(description="Metric Y:", value=self.metric_names[-1], options=self.metric_names)
        use_avg = widgets.Checkbox(description="Avg folds", value=False)
        fold_i = widgets.Dropdown(description="Fold number:", value=0, options=list(range(self.n_folds)))
        fig_widget = go.FigureWidget(
            layout={"width": WIDGET_WIDTH, "height": WIDGET_HEIGHT, "margin": {"t": CHART_MARGIN}}
        )

        def update_fold_visibility(*args: tp.Any) -> None:
            fold_i.layout.visibility = "hidden" if use_avg.value else "visible"

        def update_chart(*args: tp.Any) -> None:
            data = self._make_chart_data_avg() if use_avg.value else self._make_chart_data(fold_i.value)
            with fig_widget.batch_update():
                fig_widget.data = []
                for model in self.model_names:
                    df = data[data[Columns.Model] == model]
                    htemplate = f"<b>{model}</b><br>{metric_x.value}: %{{x}}<br>{metric_y.value}: %{{y}}<extra></extra>"
                    fig_widget.add_scatter(
                        x=df[metric_x.value], y=df[metric_y.value], mode="markers", name=model, hovertemplate=htemplate
                    )
                fig_widget.layout.xaxis.title = metric_x.value
                fig_widget.layout.yaxis.title = metric_y.value
            self.fig = go.Figure(fig_widget.data)

        metric_x.observe(update_chart, "value")
        metric_y.observe(update_chart, "value")
        use_avg.observe(update_chart, "value")
        use_avg.observe(update_fold_visibility, "value")
        fold_i.observe(update_chart, "value")

        display(
            widgets.VBox(
                [widgets.HBox(children=[use_avg, fold_i]), widgets.HBox(children=[metric_x, metric_y]), fig_widget]
            )
        )
        update_chart()
