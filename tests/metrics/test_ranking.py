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

# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics.ranking import MAP, MRR, NDCG

EMPTY_INTERACTIONS = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)


class TestMAP:
    @pytest.mark.parametrize(
        "k,divide_by_k,expected_ap",
        (
            (1, False, [0, 0, 1 / 6, 1, 1 / 3, 0, 0, 0]),
            (3, False, [0, 1 / 3, 1 / 6 * (1 / 1 + 2 / 3), 1, 1, 0, 1 * (1 / 2), 0]),
            (1, True, [0, 0, 1, 1, 1, 0, 0, 0]),
            (3, True, [0, 1 / 9, 1 / 3 * (1 / 1 + 2 / 3), 1 / 3, 1, 0, 1 / 3 * (1 / 2), 0]),
        ),
    )
    def test_calc(self, k: int, divide_by_k: bool, expected_ap: tp.List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 7, 8, 9],
                Columns.Item: [1, 3, 1, 1, 2, 3, 4, 5, 1, 1, 2, 3, 1, 2, 1],
                Columns.Rank: [9, 1, 3, 1, 3, 5, 7, 9, 1, 1, 2, 3, 2, 1, 1],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                Columns.Item: [1, 2, 1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 1, 1, 1],
            }
        )

        metric = MAP(k=k, divide_by_k=divide_by_k)
        expected_metric_per_user = pd.Series(
            expected_ap,
            index=pd.Series([1, 2, 3, 4, 5, 6, 7, 8], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert metric.calc(reco, interactions) == expected_metric_per_user.mean()

    @pytest.mark.parametrize("metric", (MAP(k=3), MAP(k=3, divide_by_k=True)))
    def test_when_no_interactions(self, metric: MAP) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))

    def test_when_duplicates_in_interactions(self) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 3, 1, 2, 3],
                Columns.Rank: [1, 2, 3, 1, 2, 3],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 1, 1, 2, 3],
            }
        )
        metric = MAP(k=3)
        expected_metric_per_user = pd.Series(
            [3.5 / 3, 3 / 3],
            index=pd.Series([1, 2], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)


class TestNDCG:

    _idcg_at_3 = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)

    @pytest.mark.parametrize(
        "k,expected_ndcg",
        (
            (1, [0, 0, 1, 1, 0]),
            (3, [0, 0, 1, 1 / _idcg_at_3, 0.5 / _idcg_at_3]),
        ),
    )
    def test_calc(self, k: int, expected_ndcg: tp.List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6],
                Columns.Item: [1, 2, 1, 2, 3, 1, 1, 2, 3, 5, 1],
                Columns.Rank: [9, 1, 1, 2, 3, 1, 3, 7, 9, 1, 1],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5],
                Columns.Item: [1, 1, 1, 2, 3, 1, 1, 2, 3, 4],
            }
        )

        metric = NDCG(k=k)
        expected_metric_per_user = pd.Series(
            expected_ndcg,
            index=pd.Series([1, 2, 3, 4, 5], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert np.allclose(metric.calc(reco, interactions), expected_metric_per_user.mean())

    def test_when_no_interactions(self) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        metric = NDCG(k=3)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))


class TestMRR:
    @pytest.mark.parametrize(
        "k,expected_mrr",
        (
            (1, [0, 0, 1, 1, 0]),
            (3, [0, 0, 1, 1, 1 / 3]),
        ),
    )
    def test_calc(self, k: int, expected_mrr: tp.List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5],
                Columns.Item: [1, 2, 1, 2, 3, 1, 1, 2, 3, 5],
                Columns.Rank: [9, 1, 1, 2, 3, 1, 3, 7, 9, 1],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 2, 3, 3, 3, 4, 5, 5, 5, 5],
                Columns.Item: [1, 1, 1, 2, 3, 1, 1, 2, 3, 4],
            }
        )

        metric = MRR(k=k)
        expected_metric_per_user = pd.Series(
            expected_mrr,
            index=pd.Series([1, 2, 3, 4, 5], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert np.allclose(metric.calc(reco, interactions), expected_metric_per_user.mean())

    def test_when_no_interactions(self) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        metric = MRR(k=3)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))

    def test_when_duplicates_in_interactions(self) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 3, 1, 2, 3],
                Columns.Rank: [1, 2, 3, 4, 5, 6],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 1, 1, 2, 3],
            }
        )
        metric = MRR(k=3)
        expected_metric_per_user = pd.Series(
            [1, 0],
            index=pd.Series([1, 2], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
