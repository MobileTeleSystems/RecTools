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

DF_METAINFO = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2"],
        "param1": [1, None],
        "param2": [None, 2],
        "param3": [None, None],
        "param4": [1, 2],
    }
)


class TestMetricsApp:
    @pytest.mark.parametrize("show_legend", (True, False))
    @pytest.mark.parametrize("auto_display", (True, False))
    @pytest.mark.parametrize(
        "scatter_kwargs",
        (
            None,
            {"width": None, "height": None, "title": None},
            {"width": 0, "height": 0, "title": "some text"},
            {"width": 800, "height": 600},
        ),
    )
    @pytest.mark.parametrize(
        "model_metadata",
        (
            None,
            DF_METAINFO,
            pd.DataFrame({Columns.Model: ["Model1", "Model2"]}),
        ),
    )
    def test_happy_path(
        self,
        model_metadata: tp.Optional[pd.DataFrame],
        show_legend: bool,
        auto_display: bool,
        scatter_kwargs: tp.Optional[tp.Dict[str, tp.Any]],
    ) -> None:
        with patch("rectools.visuals.metrics_app.MetricsApp.display", MagicMock()):
            app = MetricsApp.construct(
                models_metrics=DF_METRICS,
                models_metadata=model_metadata,
                show_legend=show_legend,
                auto_display=auto_display,
                scatter_kwargs=scatter_kwargs,
            )
            _ = app.fig

    @pytest.mark.parametrize("model_metadata", (None, DF_METAINFO))
    def test_display(
        self,
        model_metadata: tp.Optional[pd.DataFrame],
    ) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=model_metadata,
            show_legend=True,
            auto_display=False,
        )
        app.display()

    def test_models_metrics_is_not_dataframe(self) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(models_metrics=1)

    @pytest.mark.parametrize("column", (Columns.Model, Columns.Split, "metric"))
    def test_missed_models_metrics_column(self, column: str) -> None:
        models_metrics = pd.DataFrame(
            {
                Columns.Model: ["Model1", "Model2"],
                Columns.Split: [0, 0],
                "metric": [0.1, 0.2],
            }
        )
        models_metrics.drop(columns=column, inplace=True)
        with pytest.raises(KeyError):
            MetricsApp.construct(models_metrics=models_metrics)

    def test_models_metadata_is_not_dataframe(self) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(models_metrics=DF_METRICS, models_metadata=1)

    def test_models_metadata_missed_model_column(self) -> None:
        with pytest.raises(KeyError):
            MetricsApp.construct(
                models_metrics=DF_METRICS,
                models_metadata=DF_METAINFO.drop(columns=Columns.Model),
            )

    def test_models_metadata_ambiguous_model_column(self) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(
                models_metrics=DF_METRICS,
                models_metadata=pd.DataFrame({Columns.Model: ["Model1", "Model2", "Model2"]}),
            )

    def test_model_names(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        expected_model_names = sorted(DF_METRICS[Columns.Model].unique())
        assert app.model_names == expected_model_names, f"Expected {expected_model_names}, but got {app.model_names}"

    @pytest.mark.parametrize("fold_number", DF_METRICS[Columns.Split].unique())
    def test_make_chart_data(self, fold_number: int) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        chart_data = app._make_chart_data(fold_number)  # pylint: disable=protected-access
        expected_data = (
            DF_METRICS[DF_METRICS[Columns.Split] == fold_number]
            .drop(columns=Columns.Split)
            .merge(DF_METAINFO, on=Columns.Model, how="left")
        )
        pd.testing.assert_frame_equal(chart_data, expected_data)

    @pytest.mark.parametrize("fold_number", DF_METRICS[Columns.Split].unique())
    def test_make_chart_data_no_metadata(self, fold_number: int) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=None,
            auto_display=False,
        )
        chart_data = app._make_chart_data(fold_number)  # pylint: disable=protected-access
        expected_data = (
            DF_METRICS[DF_METRICS[Columns.Split] == fold_number].drop(columns=Columns.Split).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(chart_data, expected_data)

    def test_make_chart_data_avg(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        chart_data_avg = app._make_chart_data_avg()  # pylint: disable=protected-access
        expected_data_avg = (
            DF_METRICS.drop(columns=Columns.Split)
            .groupby(Columns.Model, sort=False)
            .mean()
            .reset_index(drop=False)
            .merge(DF_METAINFO, on=Columns.Model, how="left")
        )
        pd.testing.assert_frame_equal(chart_data_avg, expected_data_avg)

    def test_make_chart_data_avg_no_metadata(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=None,
            auto_display=False,
        )
        chart_data_avg = app._make_chart_data_avg()  # pylint: disable=protected-access
        expected_data_avg = (
            DF_METRICS.drop(columns=Columns.Split)
            .groupby(Columns.Model, sort=False)
            .mean()
            .reset_index(drop=False)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(chart_data_avg, expected_data_avg)
