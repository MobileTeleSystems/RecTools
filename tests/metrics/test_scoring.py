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

import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics import (
    MAP,
    MRR,
    NDCG,
    Accuracy,
    AvgRecPopularity,
    IntraListDiversity,
    MeanInvUserFreq,
    PairwiseHammingDistanceCalculator,
    Precision,
    Recall,
    Serendipity,
    calc_metrics,
)
from rectools.metrics.base import MetricAtK


class TestCalcMetrics:  # pylint: disable=attribute-defined-outside-init
    def setup(self) -> None:
        self.reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 5],
                Columns.Item: [1, 2, 1, 1, 1],
                Columns.Rank: [1, 2, 1, 1, 2],
            }
        )
        self.interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 4, 5],
                Columns.Item: [1, 2, 2, 1, 1],
            }
        )
        self.prev_interactions = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2],
                Columns.Item: [1, 2, 1, 1],
            }
        )
        features_df = pd.DataFrame(
            [
                [1, 0, 0],
                [2, 0, 1],
            ],
            columns=[Columns.Item, "feature_1", "feature_2"],
        ).set_index(Columns.Item)
        self.calculator = PairwiseHammingDistanceCalculator(features_df)
        self.catalog = list(range(10))

    def test_success(self) -> None:
        metrics = {
            "prec@1": Precision(k=1),
            "prec@2": Precision(k=2),
            "recall@1": Recall(k=1),
            "accuracy@1": Accuracy(k=1),
            "map@1": MAP(k=1),
            "map@2": MAP(k=2),
            "ndcg@1": NDCG(k=1, log_base=3),
            "mrr@1": MRR(k=1),
            "miuf": MeanInvUserFreq(k=3),
            "arp": AvgRecPopularity(k=2),
            "ild": IntraListDiversity(k=3, distance_calculator=self.calculator),
            "serendipity": Serendipity(k=3),
            "custom": MetricAtK(k=1),
        }
        with pytest.warns(UserWarning, match="Custom metrics are not supported"):
            actual = calc_metrics(metrics, self.reco, self.interactions, self.prev_interactions, self.catalog)
        expected = {
            "prec@1": 0.25,
            "prec@2": 0.375,
            "recall@1": 0.125,
            "accuracy@1": 0.825,
            "map@1": 0.125,
            "map@2": 0.375,
            "ndcg@1": 0.25,
            "mrr@1": 0.25,
            "miuf": 0.125,
            "arp": 2.75,
            "ild": 0.25,
            "serendipity": 0,
        }
        assert actual == expected

    @pytest.mark.parametrize(
        "metric,arg_names",
        (
            (Precision(k=1), ["reco"]),
            (MAP(k=1), ["reco"]),
            (MeanInvUserFreq(k=1), ["reco"]),
            (AvgRecPopularity(k=1), ["reco"]),
            (Serendipity(k=1), ["reco"]),
            (Serendipity(k=1), ["reco", "interactions"]),
            (Serendipity(k=1), ["reco", "interactions", "prev_interactions"]),
        ),
    )
    def test_raises(self, metric: MetricAtK, arg_names: tp.List[str]) -> None:
        kwargs = {name: getattr(self, name) for name in arg_names}
        with pytest.raises(ValueError):
            calc_metrics({"m": metric}, **kwargs)
