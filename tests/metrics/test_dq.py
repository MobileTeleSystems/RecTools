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

import pandas as pd

from rectools import Columns
from rectools.metrics import CoveredUsers, SufficientReco, UnrepeatedReco


class TestSufficientReco:
    def setup_method(self) -> None:
        self.metric = SufficientReco(k=2, deep=False)
        self.deep_metric = SufficientReco(k=2, deep=True)
        self.reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 3, 4],
                Columns.Item: [1, 2, 3, 1, 2, 1, 1],
                Columns.Rank: [1, 2, 3, 1, 1, 3, 2],
            }
        )

    def test_calc_deep(self) -> None:
        expected_metric_per_user = pd.Series(
            [1.0, 1.0, 0.0, 0.5],
            index=pd.Series([1, 2, 3, 4], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.deep_metric.calc_per_user(self.reco), expected_metric_per_user)
        assert self.deep_metric.calc(self.reco) == expected_metric_per_user.mean()

    def test_calc_default(self) -> None:
        expected_metric_per_user = pd.Series(
            [1, 1, 0, 0],
            index=pd.Series([1, 2, 3, 4], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(self.reco), expected_metric_per_user)
        assert self.metric.calc(self.reco) == expected_metric_per_user.mean()


class TestUnrepeatedReco:
    def setup_method(self) -> None:
        self.metric = UnrepeatedReco(k=4, deep=False)
        self.deep_metric = UnrepeatedReco(k=4, deep=True)
        self.reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                Columns.Item: [1, 2, 1, 1, 3, 4, 1, 2, 2, 1, 5],
                Columns.Rank: [1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5],
            }
        )

    def test_calc_deep(self) -> None:
        expected_metric_per_user = pd.Series(
            [1.0, 0.75, 0.5],
            index=pd.Series([1, 2, 3], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.deep_metric.calc_per_user(self.reco), expected_metric_per_user)
        assert self.deep_metric.calc(self.reco) == expected_metric_per_user.mean()

    def test_calc_default(self) -> None:
        expected_metric_per_user = pd.Series(
            [1, 0, 0],
            index=pd.Series([1, 2, 3], name=Columns.User),
        )
        pd.testing.assert_series_equal(self.metric.calc_per_user(self.reco), expected_metric_per_user)
        assert self.metric.calc(self.reco) == expected_metric_per_user.mean()


class TestCoveredUsers:
    def setup_method(self) -> None:
        self.metric = CoveredUsers(k=4)
        self.reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
                Columns.Item: [1, 2, 1, 1, 3, 1, 2, 2, 1, 5],
                Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
            }
        )
        self.interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 10, 11],
                Columns.Item: [1, 2, 1, 1, 3, 1, 2, 2, 1, 5, 5, 5],
            }
        )

    def test_calc(self) -> None:
        expected_metric_per_user = pd.Series([1, 1, 1, 0, 0], index=pd.Series([1, 2, 3, 10, 11], name=Columns.User))
        pd.testing.assert_series_equal(
            self.metric.calc_per_user(self.reco, self.interactions), expected_metric_per_user
        )
        assert self.metric.calc(self.reco, self.interactions) == expected_metric_per_user.mean()
