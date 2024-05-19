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

# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics.auc import PAUC, PAP

EMPTY_INTERACTIONS = pd.DataFrame(columns=[Columns.User, Columns.Item], dtype=int)


class TestPAUC:
    @pytest.mark.parametrize(
        "k, insufficient_cases, expected_pauc, expected_users",
        (
            (1, "don't check", [0, 0, 1, 1, 0], [1, 2, 3, 4, 5]),
            (3, "don't check", [0, 0, 1, 1, 1 / 12], [1, 2, 3, 4, 5]),
            (1, "exclude", [0, 0, 1, 1, 0], [1, 2, 3, 4, 5]),
            (3, "exclude", [0, 1, 1, 1 / 12], [1, 3, 4, 5]),  # user 2 was excluded
        ),
    )
    def test_calc(
        self, k: int, insufficient_cases: str, expected_pauc: tp.List[float], expected_users: tp.List[int]
    ) -> None:
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

        metric = PAUC(k=k, insufficient_cases=insufficient_cases)
        expected_metric_per_user = pd.Series(
            expected_pauc,
            index=pd.Series(expected_users, name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert np.allclose(metric.calc(reco, interactions), expected_metric_per_user.mean())

    def test_raise_when_insufficient_recs(self) -> None:
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
        metric = PAUC(k=3, insufficient_cases="raise")
        with pytest.raises(ValueError):
            metric.calc(reco, interactions)

    @pytest.mark.parametrize("insufficient_cases", ["raise", "exclude", "don't check"])
    def test_when_no_interactions(self, insufficient_cases: str) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        metric = PAUC(k=3, insufficient_cases=insufficient_cases)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))

    @pytest.mark.parametrize("k", (1, 3))
    @pytest.mark.parametrize("insufficient_cases", ("raise", "exclude", "don't check"))
    def test_when_duplicates_in_interactions_sufficient(self, k: int, insufficient_cases: str) -> None:
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
        metric = PAUC(k=k, insufficient_cases=insufficient_cases)
        expected_metric_per_user = pd.Series(
            [1, 0],
            index=pd.Series([1, 2], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)

    #     # this one is actually useful
    @pytest.mark.parametrize(
        "k, insufficient_cases, expected_pauc, expected_users",
        (
            (1, "don't check", [2 / 3, 0], [1, 2]),
            (1, "exclude", [2 / 3, 0], [1, 2]),
        ),
    )
    def test_when_duplicates_in_interactions_insufficient(
        self, k: int, insufficient_cases: str, expected_pauc: tp.List[int], expected_users: tp.List[int]
    ) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 3, 1, 2, 3],
                Columns.Rank: [1, 2, 3, 4, 5, 6],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2, 1],
                Columns.Item: [1, 2, 1, 1, 2, 3, 10],  # last positive is not in reco
            }
        )
        metric = PAUC(k=k, insufficient_cases=insufficient_cases)
        expected_metric_per_user = pd.Series(
            expected_pauc,
            index=pd.Series(expected_users, name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)



class TestPAP:
    @pytest.mark.parametrize(
        "k, insufficient_cases, expected_pauc, expected_users",
        (
            (1, "don't check", [0, 0, 1, 1, 0], [1, 2, 3, 4, 5]),
            (3, "don't check", [0, 0, 1, 1, 1 / 9], [1, 2, 3, 4, 5]),
            (1, "exclude", [0, 0, 1, 1, 0], [1, 2, 3, 4, 5]),
            (3, "exclude", [0, 1, 1, 1 / 9], [1, 3, 4, 5]),  # user 2 was excluded
        ),
    )
    def test_calc(
        self, k: int, insufficient_cases: str, expected_pauc: tp.List[float], expected_users: tp.List[int]
    ) -> None:
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

        metric = PAP(k=k, insufficient_cases=insufficient_cases)
        expected_metric_per_user = pd.Series(
            expected_pauc,
            index=pd.Series(expected_users, name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)
        assert np.allclose(metric.calc(reco, interactions), expected_metric_per_user.mean())

    def test_raise_when_insufficient_recs(self) -> None:
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
        metric = PAP(k=3, insufficient_cases="raise")
        with pytest.raises(ValueError):
            metric.calc(reco, interactions)

    @pytest.mark.parametrize("insufficient_cases", ["raise", "exclude", "don't check"])
    def test_when_no_interactions(self, insufficient_cases: str) -> None:
        reco = pd.DataFrame([[1, 1, 1], [2, 1, 1]], columns=[Columns.User, Columns.Item, Columns.Rank])
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        metric = PAP(k=3, insufficient_cases=insufficient_cases)
        pd.testing.assert_series_equal(metric.calc_per_user(reco, EMPTY_INTERACTIONS), expected_metric_per_user)
        assert np.isnan(metric.calc(reco, EMPTY_INTERACTIONS))

    @pytest.mark.parametrize("k", (1, 3))
    @pytest.mark.parametrize("insufficient_cases", ("raise", "exclude", "don't check"))
    def test_when_duplicates_in_interactions_sufficient(self, k: int, insufficient_cases: str) -> None:
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
        metric = PAP(k=k, insufficient_cases=insufficient_cases)
        expected_metric_per_user = pd.Series(
            [1, 0],
            index=pd.Series([1, 2], name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)

    #     # this one is actually useful
    @pytest.mark.parametrize(
        "k, insufficient_cases, expected_pauc, expected_users",
        (
            (1, "don't check", [1, 0], [1, 2]),
            (1, "exclude", [1, 0], [1, 2]),
        ),
    )
    def test_when_duplicates_in_interactions_insufficient(
        self, k: int, insufficient_cases: str, expected_pauc: tp.List[int], expected_users: tp.List[int]
    ) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2],
                Columns.Item: [1, 2, 3, 1, 2, 3],
                Columns.Rank: [1, 2, 3, 4, 5, 6],
            }
        )
        interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 2, 1],
                Columns.Item: [1, 2, 1, 1, 2, 3, 10],  # last positive is not in reco
            }
        )
        metric = PAP(k=k, insufficient_cases=insufficient_cases)
        expected_metric_per_user = pd.Series(
            expected_pauc,
            index=pd.Series(expected_users, name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric.calc_per_user(reco, interactions), expected_metric_per_user)


# # # TODO: exclude -> nan?


# # # TODO: exclude -> nan?
