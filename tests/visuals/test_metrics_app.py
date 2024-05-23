import typing as tp
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rectools import Columns
from rectools.visuals.metrics_app import MetricsApp

DF_METRICS = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2", "Model1", "Model2", "Model1", "Model2"],
        Columns.Split: [0, 0, 1, 1, 2, 2],
        "prec@10": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "recall@5": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }
)


class TestMetricsApp:
    @pytest.mark.parametrize("show_legend", (True, False))
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize(
        "layout_kwargs",
        (
            None,
            {"width": None, "height": None, "title": None},
            {"width": 0, "height": 0, "title": "some text"},
            {"width": 800, "height": 600},
        ),
    )
    def test_happy_path(
        self,
        show_legend: bool,
        auto_display: bool,
        layout_kwargs: tp.Optional[tp.Dict[str, tp.Any]],
    ) -> None:
        with patch("rectools.visuals.metrics_app.MetricsApp.display", MagicMock()):
            MetricsApp.construct(
                models_metrics=DF_METRICS,
                show_legend=show_legend,
                auto_display=auto_display,
                layout_kwargs=layout_kwargs,
            )

    def test_no_metric_columns(self) -> None:
        with pytest.raises(KeyError):
            MetricsApp.construct(
                models_metrics=pd.DataFrame(
                    {
                        Columns.Model: ["Model1", "Model2", "Model1", "Model2", "Model1", "Model2"],
                        Columns.Split: [0, 0, 1, 1, 2, 2],
                    }
                ),
            )

    def test_no_model_column(self) -> None:
        with pytest.raises(KeyError):
            MetricsApp.construct(
                models_metrics=pd.DataFrame(
                    {
                        Columns.Split: [0, 0, 1, 1, 2, 2],
                        "prec@10": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        "recall": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    }
                ),
            )
