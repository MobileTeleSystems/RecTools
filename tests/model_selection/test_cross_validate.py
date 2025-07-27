#  Copyright 2023-2024 MTS (Mobile Telesystems)
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

import os
import typing as tp

import pandas as pd
import pytest
import torch
from implicit.als import AlternatingLeastSquares
from pytorch_lightning import seed_everything

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset
from rectools.dataset.context import get_context
from rectools.metrics import Intersection, Precision, Recall, calc_metrics
from rectools.metrics.base import MetricAtK
from rectools.model_selection import LastNSplitter, cross_validate
from rectools.models import ImplicitALSWrapperModel, PopularModel, RandomModel
from rectools.models.base import ModelBase
from rectools.models.nn.transformers.hstu import HSTUModel

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
a = pytest.approx


class TestCrossValidate:
    def setup_method(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)
        interactions_df = pd.DataFrame(
            [
                [10, 11, 1, 101],
                [10, 12, 1, 102],
                [10, 11, 1, 103],
                [20, 12, 1, 101],
                [20, 11, 1, 102],
                [20, 14, 1, 103],
                [30, 11, 1, 101],
                [30, 12, 1, 102],
                [40, 11, 1, 101],
                [40, 12, 1, 102],
            ],
            columns=Columns.Interactions,
        )
        self.dataset = Dataset.construct(interactions_df)

        user_feature_df = pd.DataFrame(
            [
                [10, 0.5, 100],
                [30, 0.5, 300],
                [40, 0.5, 400],
                [20, 0.5, 200],
            ],
            columns=["id", "f1", "f2"],
        )
        item_feature_df = pd.DataFrame(
            [
                [14, "f1", "x"],
                [14, "f2", 1],
                [11, "f1", "y"],
                [11, "f2", 2],
            ],
            columns=["id", "feature", "value"],
        )
        self.featured_dataset = Dataset.construct(
            interactions_df=interactions_df,
            user_features_df=user_feature_df,
            make_dense_user_features=True,
            item_features_df=item_feature_df,
            cat_item_features=["f1"],
        )

        self.metrics: tp.Dict[str, MetricAtK] = {
            "precision@2": Precision(2),
            "recall@1": Recall(1),
        }
        self.metrics_intersection: tp.Dict[str, MetricAtK] = {
            "precision@2": Precision(2),
            "recall@1": Recall(1),
            "intersection": Intersection(1),
        }

        self.models: tp.Dict[str, ModelBase] = {
            "popular": PopularModel(),
            "random": RandomModel(random_state=42),
        }

    @pytest.mark.parametrize(
        "items_to_recommend, expected_metrics",
        (
            (
                None,
                [
                    {"model": "popular", "i_split": 0, "precision@2": 0.5, "recall@1": 0.5},
                    {"model": "random", "i_split": 0, "precision@2": 0.5, "recall@1": 0.0},
                    {"model": "popular", "i_split": 1, "precision@2": 0.375, "recall@1": 0.25},
                    {"model": "random", "i_split": 1, "precision@2": 0.375, "recall@1": 0.5},
                ],
            ),
            (
                [11, 14],
                [
                    {"model": "popular", "i_split": 0, "precision@2": 0.25, "recall@1": 0.5},
                    {"model": "random", "i_split": 0, "precision@2": 0.25, "recall@1": 0.5},
                    {"model": "popular", "i_split": 1, "precision@2": 0.125, "recall@1": 0.25},
                    {"model": "random", "i_split": 1, "precision@2": 0.125, "recall@1": 0.25},
                ],
            ),
        ),
    )
    @pytest.mark.parametrize("prefer_warm_inference_over_cold", (True, False))
    def test_happy_path(
        self,
        items_to_recommend: tp.Optional[ExternalIds],
        expected_metrics: tp.List[tp.Dict[str, tp.Any]],
        prefer_warm_inference_over_cold: bool,
    ) -> None:
        splitter = LastNSplitter(n=1, n_splits=2, filter_cold_items=False, filter_already_seen=False)

        actual = cross_validate(
            dataset=self.dataset,
            splitter=splitter,
            metrics=self.metrics,
            models=self.models,
            k=2,
            filter_viewed=False,
            items_to_recommend=items_to_recommend,
            prefer_warm_inference_over_cold=prefer_warm_inference_over_cold,
        )

        expected = {
            "splits": [
                {
                    "i_split": 0,
                    "test": 2,
                    "test_items": 2,
                    "test_users": 2,
                    "train": 2,
                    "train_items": 2,
                    "train_users": 2,
                },
                {
                    "i_split": 1,
                    "test": 4,
                    "test_items": 3,
                    "test_users": 4,
                    "train": 6,
                    "train_items": 2,
                    "train_users": 4,
                },
            ],
            "metrics": expected_metrics,
        }

        assert actual == expected

    @pytest.mark.parametrize("prefer_warm_inference_over_cold", (True, False))
    def test_happy_path_with_features(self, prefer_warm_inference_over_cold: bool) -> None:
        splitter = LastNSplitter(n=1, n_splits=2, filter_cold_items=False, filter_already_seen=False)

        models: tp.Dict[str, ModelBase] = {
            "als": ImplicitALSWrapperModel(AlternatingLeastSquares(factors=2, iterations=2, random_state=42)),
        }

        actual = cross_validate(
            dataset=self.featured_dataset,
            splitter=splitter,
            metrics=self.metrics,
            models=models,
            k=2,
            filter_viewed=False,
            prefer_warm_inference_over_cold=prefer_warm_inference_over_cold,
        )

        expected = {
            "splits": [
                {
                    "i_split": 0,
                    "test": 2,
                    "test_items": 2,
                    "test_users": 2,
                    "train": 2,
                    "train_items": 2,
                    "train_users": 2,
                },
                {
                    "i_split": 1,
                    "test": 4,
                    "test_items": 3,
                    "test_users": 4,
                    "train": 6,
                    "train_items": 2,
                    "train_users": 4,
                },
            ],
            "metrics": [
                {"model": "als", "i_split": 0, "precision@2": 0.5, "recall@1": 0.0},
                {"model": "als", "i_split": 1, "precision@2": 0.375, "recall@1": 0.25},
            ],
        }

        assert actual == expected

    @pytest.mark.parametrize(
        "ref_models,validate_ref_models,expected_metrics",
        (
            (
                ["popular"],
                False,
                [
                    {"model": "random", "i_split": 0, "precision@2": 0.5, "recall@1": 0.0, "intersection_popular": 0.5},
                    {
                        "model": "random",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.5,
                        "intersection_popular": 0.75,
                    },
                ],
            ),
            (
                ["popular"],
                True,
                [
                    {
                        "model": "popular",
                        "i_split": 0,
                        "precision@2": 0.5,
                        "recall@1": 0.5,
                        "intersection_popular": 1.0,
                    },
                    {"model": "random", "i_split": 0, "precision@2": 0.5, "recall@1": 0.0, "intersection_popular": 0.5},
                    {
                        "model": "popular",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.25,
                        "intersection_popular": 1.0,
                    },
                    {
                        "model": "random",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.5,
                        "intersection_popular": 0.75,
                    },
                ],
            ),
            (
                ["random"],
                False,
                [
                    {"model": "popular", "i_split": 0, "precision@2": 0.5, "recall@1": 0.5, "intersection_random": 0.5},
                    {
                        "model": "popular",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.25,
                        "intersection_random": 0.75,
                    },
                ],
            ),
            (
                ["random"],
                True,
                [
                    {"model": "popular", "i_split": 0, "precision@2": 0.5, "recall@1": 0.5, "intersection_random": 0.5},
                    {"model": "random", "i_split": 0, "precision@2": 0.5, "recall@1": 0.0, "intersection_random": 1.0},
                    {
                        "model": "popular",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.25,
                        "intersection_random": 0.75,
                    },
                    {
                        "model": "random",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.5,
                        "intersection_random": 1.0,
                    },
                ],
            ),
            (["random", "popular"], False, []),
            (
                ["random", "popular"],
                True,
                [
                    {
                        "model": "popular",
                        "i_split": 0,
                        "precision@2": 0.5,
                        "recall@1": 0.5,
                        "intersection_random": 0.5,
                        "intersection_popular": 1.0,
                    },
                    {
                        "model": "random",
                        "i_split": 0,
                        "precision@2": 0.5,
                        "recall@1": 0.0,
                        "intersection_random": 1.0,
                        "intersection_popular": 0.5,
                    },
                    {
                        "model": "popular",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.25,
                        "intersection_random": 0.75,
                        "intersection_popular": 1.0,
                    },
                    {
                        "model": "random",
                        "i_split": 1,
                        "precision@2": 0.375,
                        "recall@1": 0.5,
                        "intersection_random": 1.0,
                        "intersection_popular": 0.75,
                    },
                ],
            ),
        ),
    )
    def test_happy_path_with_intersection(
        self,
        ref_models: tp.Optional[tp.List[str]],
        validate_ref_models: bool,
        expected_metrics: tp.List[tp.Dict[str, tp.Any]],
    ) -> None:
        splitter = LastNSplitter(n=1, n_splits=2, filter_cold_items=False, filter_already_seen=False)

        actual = cross_validate(
            dataset=self.dataset,
            splitter=splitter,
            metrics=self.metrics_intersection,
            models=self.models,
            k=2,
            filter_viewed=False,
            ref_models=ref_models,
            validate_ref_models=validate_ref_models,
        )

        expected = {
            "splits": [
                {
                    "i_split": 0,
                    "test": 2,
                    "test_items": 2,
                    "test_users": 2,
                    "train": 2,
                    "train_items": 2,
                    "train_users": 2,
                },
                {
                    "i_split": 1,
                    "test": 4,
                    "test_items": 3,
                    "test_users": 4,
                    "train": 6,
                    "train_items": 2,
                    "train_users": 4,
                },
            ],
            "metrics": expected_metrics,
        }

        assert actual == expected

    @pytest.mark.parametrize(
        "as_ref_model,expected_metrics",
        (
            (
                False,
                [
                    {"model": "hstu", "i_split": 0, "precision@2": 0.375, "recall@1": 0.0},
                ],
            ),
            (
                True,
                [
                    {"model": "hstu", "i_split": 0, "precision@2": 0.375, "recall@1": 0.0},
                ],
            ),
        ),
    )
    def test_context_preprocessing(
        self,
        as_ref_model: bool,
        expected_metrics: tp.List[tp.Dict[str, tp.Any]],
    ) -> None:
        splitter = LastNSplitter(n=1, n_splits=1, filter_cold_items=False, filter_already_seen=False)
        models = {"hstu": HSTUModel()}
        ref_models = []
        if as_ref_model:
            ref_models = ["hstu"]
        actual_cv = cross_validate(
            dataset=self.dataset,
            splitter=splitter,
            metrics=self.metrics,
            models=models,  # type: ignore[arg-type]
            k=2,
            filter_viewed=False,
            ref_models=ref_models,
            validate_ref_models=True,
        )
        hstu_no_envelope = HSTUModel()
        (train_ids, test_ids, split_info) = next(splitter.split(self.dataset.interactions, collect_fold_stats=False))
        train = self.dataset.filter_interactions(
            row_indexes_to_keep=train_ids,
            keep_external_ids=True,
        )
        test = self.dataset.interactions.df.loc[test_ids]
        test[Columns.User] = self.dataset.user_id_map.convert_to_external(test[Columns.User])
        test[Columns.Item] = self.dataset.item_id_map.convert_to_external(test[Columns.Item])
        test_users = test[Columns.User].unique()
        train_external = train.get_raw_interactions()

        hstu_no_envelope.fit(train)
        reco = hstu_no_envelope.recommend(
            users=test_users,
            dataset=train,
            k=2,
            filter_viewed=False,
            on_unsupported_targets="warn",
            context=get_context(test),
        )
        metric_values = calc_metrics(
            self.metrics,
            reco=reco,
            interactions=test,
            prev_interactions=train_external,
            catalog=train_external[Columns.Item].unique(),
        )
        res_no_env = {"model": "hstu", "i_split": split_info["i_split"]}
        res_no_env.update(metric_values)
        metrics_no_env = [res_no_env]
        assert actual_cv["metrics"] == expected_metrics
        assert actual_cv["metrics"] == metrics_no_env
