#  Copyright 2022 MTS (Mobile Telesystems)
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
import warnings

import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics.base import MetricAtK


class SomeMetric(MetricAtK):
    def calc(
        self, reco: pd.DataFrame, interactions: pd.DataFrame, prev_interactions: pd.DataFrame, ref_reco: pd.DataFrame
    ) -> None:
        self._check(reco, interactions, prev_interactions, ref_reco)


class TestMetricAtK:
    @pytest.fixture
    def data(self) -> tp.Dict[str, pd.DataFrame]:
        reco = pd.DataFrame(
            [[10, 100, 1], [10, 200, 2], [20, 200, 1]],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        interactions = pd.DataFrame(
            [[10, 100], [10, 200], [20, 200]],
            columns=[Columns.User, Columns.Item],
        )
        prev_interactions = pd.DataFrame(
            [[10, 100], [10, 200], [20, 200]],
            columns=[Columns.User, Columns.Item],
        )
        ref_reco = pd.DataFrame(
            [[10, 100, 1], [10, 300, 2], [20, 200, 1]],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        return {
            "reco": reco,
            "interactions": interactions,
            "prev_interactions": prev_interactions,
            "ref_reco": ref_reco,
        }

    @pytest.mark.parametrize("table", ("reco", "interactions", "prev_interactions", "ref_reco"))
    @pytest.mark.parametrize("column", (Columns.User, Columns.Item, Columns.Rank))
    def test_check_columns(self, data: tp.Dict[str, pd.DataFrame], table: str, column: str) -> None:
        if column not in data[table]:
            return
        metric = SomeMetric(1)
        data[table].drop(columns=column, inplace=True)
        with pytest.raises(KeyError) as e:
            metric.calc(**data)
        err_text = e.value.args[0]
        assert table in err_text.lower()
        assert column in err_text.lower()

    @pytest.mark.parametrize("table", ("reco", "ref_reco"))
    def test_check_rank_type(self, data: tp.Dict[str, pd.DataFrame], table: str) -> None:
        data[table][Columns.Rank] = data[table][Columns.Rank].astype(float)
        metric = SomeMetric(1)
        with warnings.catch_warnings(record=True) as w:
            metric.calc(**data)
            assert len(w) == 1
            for phrase in (Columns.Rank, table, "dtype", "integer"):
                assert phrase in str(w[-1].message)

    @pytest.mark.parametrize("table", ("reco", "ref_reco"))
    def test_check_min_rank(self, data: tp.Dict[str, pd.DataFrame], table: str) -> None:
        data[table][Columns.Rank] = data[table][Columns.Rank].map({1: 3, 2: 2})
        metric = SomeMetric(1)
        with warnings.catch_warnings(record=True) as w:
            metric.calc(**data)
            assert len(w) == 1
            for phrase in (Columns.Rank, table, "min value", "1"):
                assert phrase in str(w[-1].message)
