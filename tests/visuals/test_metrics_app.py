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

import pandas as pd
import pytest

from rectools import Columns
from rectools.visuals.metrics_app import MetricsApp

DF_METRICS = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2", "Model1", "Model2"],
        Columns.Split: [0, 0, 1, 1],
        "prec@10": [0.1, 0.2, 0.3, 0.4],
        "recall@5": [0.5, 0.6, 0.7, 0.8],
    }
)

DF_METAINFO = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2"],
        "param1": [1, None],
        "param2": [None, "some text"],
        "param3": [None, None],
        "param4": [1, 2.0],
    }
)

DF_MERGED = pd.DataFrame(
    {
        Columns.Model: ["Model1", "Model2", "Model1", "Model2"],
        Columns.Split: [0, 0, 1, 1],
        "prec@10": [0.1, 0.2, 0.3, 0.4],
        "recall@5": [0.5, 0.6, 0.7, 0.8],
        "param1": [1, None, 1, None],
        "param2": [None, "some text", None, "some text"],
        "param3": [None, None, None, None],
        "param4": [1, 2.0, 1, 2.0],
    }
)


class TestMetricsApp:

    # -------------------------------------------Test happy paths------------------------------------------- #

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
        "models_metrics",
        (
            DF_METRICS,
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model2"],
                    "prec@10": [0.1, 0.2],
                    "recall@5": [0.5, 0.6],
                }
            ),
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
        models_metrics: pd.DataFrame,
        model_metadata: tp.Optional[pd.DataFrame],
        show_legend: bool,
        auto_display: bool,
        scatter_kwargs: tp.Optional[tp.Dict[str, tp.Any]],
    ) -> None:
        app = MetricsApp.construct(
            models_metrics=models_metrics,
            models_metadata=model_metadata,
            show_legend=show_legend,
            auto_display=auto_display,
            scatter_kwargs=scatter_kwargs,
        )
        app.display()

    # -------------------------------------Test metrics data validation------------------------------------- #

    @pytest.mark.parametrize(
        "models_metrics",
        (
            pd.DataFrame(
                {
                    Columns.Split: [0, 0, 1, 1],
                    "prec@10": [0.1, 0.2, 0.3, 0.4],
                }
            ),
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model2", "Model1", "Model2"],
                    Columns.Split: [0, 0, 1, 1],
                }
            ),
        ),
    )
    def test_missed_models_metrics_column(self, models_metrics: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            MetricsApp.construct(models_metrics=models_metrics)

    @pytest.mark.parametrize(
        "models_metrics",
        (
            pd.DataFrame(
                {
                    Columns.Model: [None, "Model2"],
                    "metric": [0.1, None],
                }
            ),
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model2"],
                    Columns.Split: [None, 0],
                    "metric": [0.1, None],
                }
            ),
        ),
    )
    def test_models_metrics_has_nan(self, models_metrics: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(models_metrics=models_metrics)

    @pytest.mark.parametrize(
        "models_metrics",
        (
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model1"],
                    "metric": [0.1, 0.2],
                }
            ),
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model1"],
                    Columns.Split: [0, 0],
                    "metric": [0.1, 0.2],
                }
            ),
        ),
    )
    def test_models_metrics_has_non_unique_models_names(self, models_metrics: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(models_metrics=models_metrics)

    @pytest.mark.parametrize(
        "models_metrics",
        (
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model2", "Model1", "Model2"],
                    Columns.Split: [0, 0, 1, 2],
                    "prec@10": [0.1, 0.2, 0.3, 0.4],
                    "recall@5": [0.5, 0.6, 0.7, 0.8],
                }
            ),
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model2", "Model1"],
                    Columns.Split: [0, 0, 1],
                    "prec@10": [0.1, 0.2, 0.3],
                    "recall@5": [0.5, 0.6, 0.7],
                }
            ),
        ),
    )
    def test_model_metrics_has_non_consistent_folds(self, models_metrics: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(models_metrics=models_metrics)

    @pytest.mark.parametrize(
        "models_metrics",
        (
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model1"],
                    "metric": ["0.1", 0.2],
                }
            ),
            pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model1"],
                    Columns.Split: [0, 0],
                    "metric1": [0.1, 0.2],
                    "metric2": ["0.3", "0.4"],
                }
            ),
        ),
    )
    def test_models_metrics_has_non_numeric_metric_columns(self, models_metrics: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            MetricsApp.construct(models_metrics=models_metrics)

    # --------------------------------------Test meta data validation--------------------------------------- #

    def test_models_metadata_missed_model_column(self) -> None:
        with pytest.raises(KeyError, match="Missing `Model` column in `models_metadata` DataFrame"):
            MetricsApp.construct(
                models_metrics=DF_METRICS,
                models_metadata=DF_METAINFO.drop(columns=Columns.Model),
            )

    def test_models_metadata_ambiguous_model_column(self) -> None:
        with pytest.raises(ValueError, match="`Model` values of `models_metadata` should be unique`"):
            MetricsApp.construct(
                models_metrics=DF_METRICS,
                models_metadata=pd.DataFrame({Columns.Model: ["Model1", "Model2", "Model2"]}),
            )

    def test_models_metadata_has_nan_in_models_names(self) -> None:
        with pytest.raises(ValueError, match="Found NaN values in `Model` column"):
            MetricsApp.construct(
                models_metrics=DF_METRICS,
                models_metadata=pd.DataFrame({Columns.Model: ["Model1", "Model2", None]}),
            )

    # -------------------------------------------Test properties-------------------------------------------- #

    def test_model_names(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        expected_model_names = ["Model1", "Model2"]
        assert app.model_names == expected_model_names, f"Expected {expected_model_names}, but got {app.model_names}"

    def test_folds_ids(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        expected_fold_ids = [0, 1]
        assert app.fold_ids == expected_fold_ids, f"Expected {expected_fold_ids}, but got {app.fold_ids}"

    def test_folds_ids_no_split_column(self) -> None:
        app = MetricsApp.construct(
            models_metrics=pd.DataFrame(
                {
                    Columns.Model: ["Model1", "Model2"],
                    "prec@10": [0.2, 0.3],
                }
            ),
            auto_display=False,
        )
        expected_fold_ids = None
        assert app.fold_ids == expected_fold_ids, f"Expected {expected_fold_ids}, but got {app.fold_ids}"

    # ----------------------------------------Test data aggregators----------------------------------------- #

    @pytest.mark.parametrize("fold_number", DF_METRICS[Columns.Split].unique())
    def test_make_chart_data_fold(self, fold_number: int) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        chart_data = app._make_chart_data_fold(fold_number)  # pylint: disable=protected-access
        expected_data = DF_MERGED[DF_MERGED[Columns.Split] == fold_number].reset_index(drop=True)
        pd.testing.assert_frame_equal(chart_data, expected_data)

    @pytest.mark.parametrize("fold_number", DF_METRICS[Columns.Split].unique())
    def test_make_chart_data_fold_no_metadata(self, fold_number: int) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=None,
            auto_display=False,
        )
        chart_data = app._make_chart_data_fold(fold_number)  # pylint: disable=protected-access
        expected_data = DF_METRICS[DF_METRICS[Columns.Split] == fold_number].reset_index(drop=True)
        pd.testing.assert_frame_equal(chart_data, expected_data)

    def test_make_chart_data_avg(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=DF_METAINFO,
            auto_display=False,
        )
        chart_data = app._make_chart_data_avg()  # pylint: disable=protected-access
        expected_data = pd.DataFrame(
            {
                Columns.Model: ["Model1", "Model2"],
                "prec@10": [0.2, 0.3],
                "recall@5": [0.6, 0.7],
                "param1": [1, None],
                "param2": [None, "some text"],
                "param3": [None, None],
                "param4": [1, 2.0],
            }
        )
        pd.testing.assert_frame_equal(chart_data, expected_data)

    def test_make_chart_data_avg_no_metadata(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=None,
            auto_display=False,
        )
        chart_data = app._make_chart_data_avg()  # pylint: disable=protected-access
        expected_data = pd.DataFrame(
            {
                Columns.Model: ["Model1", "Model2"],
                "prec@10": [0.2, 0.3],
                "recall@5": [0.6, 0.7],
            }
        )
        pd.testing.assert_frame_equal(chart_data, expected_data)

    # -----------------------------------------Test helper methods------------------------------------------ #

    def test_split_to_meta_and_model_with_meta(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=None,
            auto_display=False,
        )
        sep = ","
        test_string = "10,random"
        expected_result = ("10", "random")
        assert app._split_to_meta_and_model(test_string, sep) == expected_result  # pylint: disable=protected-access

    def test_split_to_meta_and_model_without_meta(self) -> None:
        app = MetricsApp.construct(
            models_metrics=DF_METRICS,
            models_metadata=None,
            auto_display=False,
        )
        sep = ","
        test_string = "random"
        expected_result = ("", "random")
        assert app._split_to_meta_and_model(test_string, sep) == expected_result  # pylint: disable=protected-access
