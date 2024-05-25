import typing as tp
from functools import lru_cache

import ipywidgets as widgets
import pandas as pd
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
        show_legend: bool = True,
        auto_display: bool = True,
        layout_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        self.models_metrics = models_metrics
        self.show_legend = show_legend
        self.auto_display = auto_display
        self.layout_kwargs = layout_kwargs if layout_kwargs is not None else {}
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
        layout_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
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
        show_legend : bool, default True
            Specifies whether to display the chart legend.
        auto_display : bool, default True
            Automatically displays the widgets immediately after initialization.
         layout_kwargs : tp.Optional[tp.Dict[str, tp.Any]], optional, default None
            Additional arguments from `plotly.graph_objects.Layout`

        Returns
        -------
        MetricsApp
            An instance of `MetricsApp`, providing interactive Jupyter widget for metric visualization.

        Examples
        --------
        Create interactive widget
        >>> example_df = pd.DataFrame(
        ...    {
        ...        Columns.Model: ["Model1", "Model2", "Model1", "Model2", "Model1", "Model2"],
        ...        Columns.Split: [0, 0, 1, 1, 2, 2],
        ...        "prec@10": [0.031, 0.025, 0.027, 0.21, 0.031, 0.033],
        ...        "recall@10": [0.041, 0.045, 0.055, 0.08, 0.036, 0.021],
        ...        "novelty@10": [2.6, 11.3, 4.3, 9.8, 3.3, 11.2],
        ...    })
        >>> app = MetricsApp.construct(
        ...    models_metrics=example_df,
        ...    show_legend=True,
        ...    auto_display=False,
        ...    layout_kwargs={"width": 800, "height": 600})

        Get plotly chart from widget state
        >>> fig = app.fig
        >>> fig = fig.update_layout(
        ...    title="Metrics comparison",
        ...    margin=None,
        ... )
        """
        return cls(models_metrics, show_legend, auto_display, layout_kwargs)

    @property
    @lru_cache
    def metric_names(self) -> tp.List[str]:
        """List of metric column names from the `models_metrics`."""
        non_metric_columns = {Columns.Split, Columns.Model}
        return [col for col in self.models_metrics.columns if col not in non_metric_columns]

    @property
    @lru_cache
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

    @lru_cache
    def _make_chart_data(self, fold_number: int) -> pd.DataFrame:
        return self.models_metrics[self.models_metrics[Columns.Split] == fold_number].drop(columns=Columns.Split)

    @lru_cache
    def _make_chart_data_avg(self) -> pd.DataFrame:
        return self.models_metrics.drop(columns=Columns.Split).groupby(Columns.Model, sort=False).mean().reset_index()

    def _update_chart(
        self,
        fig_widget: go.FigureWidget,
        metric_x: widgets.Dropdown,
        metric_y: widgets.Dropdown,
        use_avg: widgets.Checkbox,
        fold_i: widgets.Dropdown,
    ) -> None:
        data = self._make_chart_data_avg() if use_avg.value else self._make_chart_data(fold_i.value)
        existing_traces = {trace.name: trace for trace in fig_widget.data}
        with fig_widget.batch_update():
            for model in self.model_names:
                df = data[data[Columns.Model] == model]
                hover = f"<b>{model}</b><br>{metric_x.value}: %{{x}}<br>{metric_y.value}: %{{y}}<extra></extra>"
                if model in existing_traces:
                    trace = existing_traces[model]
                    trace.x = df[metric_x.value]
                    trace.y = df[metric_y.value]
                    trace.hovertemplate = hover
                else:
                    fig_widget.add_scatter(
                        x=df[metric_x.value],
                        y=df[metric_y.value],
                        mode="markers",
                        name=model,
                        hovertemplate=hover,
                        showlegend=self.show_legend,
                    )
            fig_widget.layout.xaxis.title = metric_x.value
            fig_widget.layout.yaxis.title = metric_y.value
        self.fig = go.Figure(data=fig_widget.data, layout=fig_widget.layout)
        self.fig.layout.margin = None

    def _update_fold_visibility(self, use_avg: widgets.Checkbox, fold_i: widgets.Dropdown) -> None:
        fold_i.layout.visibility = "hidden" if use_avg.value else "visible"

    def display(self) -> None:
        """Display MetricsApp widget"""
        metric_x = widgets.Dropdown(description="Metric X:", value=self.metric_names[0], options=self.metric_names)
        metric_y = widgets.Dropdown(description="Metric Y:", value=self.metric_names[-1], options=self.metric_names)
        use_avg = widgets.Checkbox(description="Avg folds", value=True)
        fold_i = widgets.Dropdown(description="Fold number:", value=self.fold_ids[0], options=self.fold_ids)

        layout_params = {
            "width": WIDGET_WIDTH,
            "height": WIDGET_HEIGHT,
            "margin": {"t": TOP_CHART_MARGIN},
            "legend_title": LEGEND_TITLE,
        }
        layout_params.update(self.layout_kwargs)
        fig_widget = go.FigureWidget(layout=layout_params)

        metric_x.observe(lambda upd: self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i), "value")
        metric_y.observe(lambda upd: self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i), "value")
        use_avg.observe(lambda upd: self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i), "value")
        use_avg.observe(lambda upd: self._update_fold_visibility(use_avg, fold_i), "value")
        fold_i.observe(lambda upd: self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i), "value")

        display(widgets.VBox([widgets.HBox([use_avg, fold_i]), widgets.HBox([metric_x, metric_y]), fig_widget]))
        self._update_fold_visibility(use_avg, fold_i)
        self._update_chart(fig_widget, metric_x, metric_y, use_avg, fold_i)
