import typing as tp

import pandas as pd
import plotly.express as px
import pytest

from rectools import Columns
from rectools.visuals.metrics_app import MetricsApp

DF_METRICS = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2", "Model1", "Model2", "Model1", "Model2"],
        Columns.Split: [0, 0, 1, 1, 2, 2],
        "prec@10": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "recall": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
)

DF_METRICS_EMPTY = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2", "Model1", "Model2", "Model1", "Model2"],
        Columns.Split: [0, 0, 1, 1, 2, 2],
    }
)

DF_METRICS_NO_MODEL = pd.DataFrame(
    {
        Columns.Split: [0, 0, 1, 1, 2, 2],
        "prec@10": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "recall": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
)


class TestMetricsApp:
    @pytest.mark.parametrize("chart_width", (10, 100.0, 1000))
    @pytest.mark.parametrize("chart_height", (10, 100.0, 1000))
    @pytest.mark.parametrize("show_legend", (True, False))
    @pytest.mark.parametrize(
        "color_discrete_sequence",
        (
            None,
            px.colors.qualitative.Plotly,
            ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b"],
        ),
    )
    def test_correct_run(
        self,
        chart_width: tp.Union[float, int],
        chart_height: tp.Union[float, int],
        show_legend: bool,
        color_discrete_sequence: tp.Optional[tp.List[str]],
    ) -> None:
        MetricsApp.construct(
            df_metrics_data=DF_METRICS,
            chart_width=chart_width,
            chart_height=chart_height,
            show_legend=show_legend,
            auto_display=False,
            color_discrete_sequence=color_discrete_sequence,
        )

    def test_incorrect_width(self) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(
                df_metrics_data=DF_METRICS,
                chart_width=0,
                auto_display=False,
            )

    def test_incorrect_height(self) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(
                df_metrics_data=DF_METRICS,
                chart_height=0,
                auto_display=False,
            )

    def test_no_metric_columns(self) -> None:
        with pytest.raises(KeyError):
            MetricsApp.construct(
                df_metrics_data=DF_METRICS_EMPTY,
            )

    def test_no_base_columns(self) -> None:
        with pytest.raises(KeyError):
            MetricsApp.construct(
                df_metrics_data=DF_METRICS_NO_MODEL,
            )
