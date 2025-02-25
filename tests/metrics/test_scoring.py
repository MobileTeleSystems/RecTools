#  Copyright 2022-2025 MTS (Mobile Telesystems)
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
    MCC,
    MRR,
    NDCG,
    PAP,
    Accuracy,
    AvgRecPopularity,
    CatalogCoverage,
    CoveredUsers,
    DebiasConfig,
    F1Beta,
    HitRate,
    Intersection,
    IntraListDiversity,
    MeanInvUserFreq,
    PairwiseHammingDistanceCalculator,
    PartialAUC,
    Precision,
    Recall,
    Serendipity,
    SufficientReco,
    UnrepeatedReco,
    calc_metrics,
    debias_interactions,
)
from rectools.metrics.base import MetricAtK


class TestCalcMetrics:  # pylint: disable=attribute-defined-outside-init
    def setup_method(self) -> None:
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
        self.ref_recos = {
            "one": pd.DataFrame(
                {
                    Columns.User: [1, 1, 2, 3, 5],
                    Columns.Item: [1, 3, 1, 1, 2],
                    Columns.Rank: [1, 2, 1, 3, 2],
                }
            ),
            "two": pd.DataFrame(
                {
                    Columns.User: [1, 1, 2, 3, 5],
                    Columns.Item: [1, 2, 1, 1, 1],
                    Columns.Rank: [1, 2, 3, 1, 1],
                }
            ),
        }

    def test_success(self) -> None:
        metrics = {
            "prec@1": Precision(k=1),
            "prec@2": Precision(k=2),
            "rprec@2": Precision(k=2, r_precision=True),
            "recall@1": Recall(k=1),
            "accuracy@1": Accuracy(k=1),
            "hitrate@1": HitRate(k=1),
            "map@1": MAP(k=1),
            "map@2": MAP(k=2),
            "ndcg@1": NDCG(k=1, log_base=3),
            "pauc@1": PartialAUC(k=1),
            "pauc@2": PartialAUC(k=2),
            "pap@1": PAP(k=1),
            "pap@2": PAP(k=2),
            "mrr@1": MRR(k=1),
            "miuf": MeanInvUserFreq(k=3),
            "arp": AvgRecPopularity(k=2),
            "ild": IntraListDiversity(k=3, distance_calculator=self.calculator),
            "serendipity": Serendipity(k=3),
            "intersection": Intersection(k=2, ref_k=2),
            "custom": MetricAtK(k=1),
            "sufficient": SufficientReco(k=2),
            "unrepeated": UnrepeatedReco(k=2),
            "covered_users": CoveredUsers(k=2),
            "catalog_coverage": CatalogCoverage(k=2, normalize=True),
        }
        with pytest.warns(UserWarning, match="Custom metrics are not supported"):
            actual = calc_metrics(
                metrics, self.reco, self.interactions, self.prev_interactions, self.catalog, self.ref_recos
            )
        expected = {
            "prec@1": 0.25,
            "prec@2": 0.375,
            "rprec@2": 0.5,
            "recall@1": 0.125,
            "accuracy@1": 0.825,
            "hitrate@1": 0.25,
            "map@1": 0.125,
            "map@2": 0.375,
            "ndcg@1": 0.25,
            "pauc@1": 0.25,
            "pauc@2": 0.375,
            "pap@1": 0.25,
            "pap@2": 0.375,
            "mrr@1": 0.25,
            "miuf": 0.125,
            "arp": 2.75,
            "ild": 0.25,
            "serendipity": 0,
            "intersection_one": 0.375,
            "intersection_two": 0.75,
            "sufficient": 0.25,
            "unrepeated": 1,
            "covered_users": 0.75,
            "catalog_coverage": 0.2,
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
            (PAP(k=1), ["reco"]),
            (PartialAUC(k=1), ["reco"]),
            (Intersection(k=1), ["reco"]),
            (CoveredUsers(k=1), ["reco"]),
            (CatalogCoverage(k=1), ["reco"]),
        ),
    )
    def test_raises(self, metric: MetricAtK, arg_names: tp.List[str]) -> None:
        kwargs = {name: getattr(self, name) for name in arg_names}
        with pytest.raises(ValueError):
            calc_metrics({"m": metric}, **kwargs)

    def test_success_debias(self) -> None:
        debias_config = DebiasConfig(iqr_coef=1.5, random_state=32)
        debiased_metrics = {
            "debiased_precision@3": Precision(k=3, debias_config=debias_config),
            "debiased_rprecision@3": Precision(k=3, r_precision=True, debias_config=debias_config),
            "debiased_recall@3": Recall(k=3, debias_config=debias_config),
            "debiased_f1beta@3": F1Beta(k=3, debias_config=debias_config),
            "debiased_accuracy@3": Accuracy(k=3, debias_config=debias_config),
            "debiased_mcc@3": MCC(k=3, debias_config=debias_config),
            "debiased_hitrate@3": HitRate(k=3, debias_config=debias_config),
            "debiased_map@1": MAP(k=1, debias_config=debias_config),
            "debiased_map@3": MAP(k=3, debias_config=debias_config),
            "debiased_ndcg@3": NDCG(k=3, debias_config=debias_config),
            "debiased_mrr@3": MRR(k=3, debias_config=debias_config),
            "debiased_pap@3": PAP(k=3, debias_config=debias_config),
            "debiased_partauc@3": PartialAUC(k=3, debias_config=debias_config),
        }
        metrics = {
            "debiased_precision@3": Precision(k=3),
            "debiased_rprecision@3": Precision(k=3, r_precision=True),
            "debiased_recall@3": Recall(k=3),
            "debiased_f1beta@3": F1Beta(k=3),
            "debiased_accuracy@3": Accuracy(k=3),
            "debiased_mcc@3": MCC(k=3),
            "debiased_hitrate@3": HitRate(k=3),
            "debiased_map@1": MAP(k=1),
            "debiased_map@3": MAP(k=3),
            "debiased_ndcg@3": NDCG(k=3),
            "debiased_mrr@3": MRR(k=3),
            "debiased_pap@3": PAP(k=3),
            "debiased_partauc@3": PartialAUC(k=3),
        }

        debiased_interactions = debias_interactions(self.interactions, config=debias_config)

        actual = calc_metrics(
            metrics=debiased_metrics,
            reco=self.reco,
            interactions=self.interactions,
            catalog=self.catalog,
        )
        expected = calc_metrics(
            metrics=metrics,
            reco=self.reco,
            interactions=debiased_interactions,
            catalog=self.catalog,
        )
        assert actual == expected
