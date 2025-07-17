import re
import typing as tp

import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset.context import get_context


class TestContextPreprocessor:
    @pytest.fixture
    def context_to_filter(self) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                [0, 0, 2, "2021-09-01", 1],
                [4, 2, 1, "2021-09-02", 1],
                [2, 1, 1, "2021-09-02", 1],
                [2, 2, 1, "2021-09-03", 1],
                [3, 2, 4, "2021-09-03", 1],
                [3, 3, 5, "2021-09-03", 1],
                [3, 4, 1, "2021-09-04", 1],
                [1, 2, 1, "2021-09-04", 1],
                [3, 1, 1, "2021-09-05", 1],
                [4, 2, 1, "2021-09-05", 1],
                [3, 3, 1, "2021-09-06", 1],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime, "extra"],
        )
        return df

    @pytest.mark.parametrize(
        "expected_columns, expected_context",
        (
            (
                [Columns.User, Columns.Datetime, Columns.Weight, "extra"],
                pd.DataFrame(
                    [
                        [0, 2, "2021-09-01", 1],
                        [1, 1, "2021-09-04", 1],
                        [2, 1, "2021-09-02", 1],
                        [3, 4, "2021-09-03", 1],
                        [4, 1, "2021-09-02", 1],
                    ],
                    columns=[Columns.User, Columns.Weight, Columns.Datetime, "extra"],
                ).astype({Columns.Datetime: "datetime64[ns]"}),
            ),
        ),
    )
    def test_get_context(
        self,
        context_to_filter: pd.DataFrame,
        expected_columns: tp.List[str],
        expected_context: pd.DataFrame,
    ) -> None:

        actual = get_context(context_to_filter).reset_index(drop=True)
        assert Columns.Item not in actual.columns
        assert pd.api.types.is_datetime64_any_dtype(actual[Columns.Datetime])
        assert set(actual.columns.tolist()) == set(expected_columns)
        pd.testing.assert_frame_equal(actual, expected_context)

    def test_wrong_datetime(
        self,
        context_to_filter: pd.DataFrame,
    ) -> None:
        context_to_filter.loc[0, [Columns.Datetime]] = "incorrect type"
        error_match = f"Column '{Columns.Datetime}' must be convertible to 'datetime64' type"
        with pytest.raises(TypeError, match=re.escape(error_match)):
            get_context(context_to_filter)
