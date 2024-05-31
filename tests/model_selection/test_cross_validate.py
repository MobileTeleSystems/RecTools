# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest
from implicit.als import AlternatingLeastSquares
from scipy import sparse

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset, DenseFeatures, SparseFeatures
from rectools.metrics import Intersection, Precision, Recall
from rectools.metrics.base import MetricAtK
from rectools.model_selection import LastNSplitter, cross_validate
from rectools.model_selection.cross_validate import _gen_2x_internal_ids_dataset
from rectools.models import ImplicitALSWrapperModel, PopularModel, RandomModel
from rectools.models.base import ModelBase
from tests.testing_utils import assert_sparse_matrix_equal

a = pytest.approx


class TestGen2xInternalIdsDataset:
    def setup_method(self) -> None:
        self.interactions_internal_df = pd.DataFrame(
            [
                [0, 0, 1, 101],
                [0, 1, 1, 102],
                [0, 0, 1, 103],
                [3, 0, 1, 101],
                [3, 2, 1, 102],
            ],
            columns=Columns.Interactions,
        ).astype({Columns.Datetime: "datetime64[ns]", Columns.Weight: float})

        self.expected_interactions_2x_internal_df = pd.DataFrame(
            [
                [0, 0, 1, 101],
                [0, 1, 1, 102],
                [0, 0, 1, 103],
                [1, 0, 1, 101],
                [1, 2, 1, 102],
            ],
            columns=Columns.Interactions,
        ).astype({Columns.Datetime: "datetime64[ns]", Columns.Weight: float})

    @pytest.mark.parametrize("prefer_warm_inference_over_cold", (True, False))
    def test_without_features(self, prefer_warm_inference_over_cold: bool) -> None:
        dataset = _gen_2x_internal_ids_dataset(
            self.interactions_internal_df, None, None, prefer_warm_inference_over_cold
        )

        np.testing.assert_equal(dataset.user_id_map.external_ids, np.array([0, 3]))
        np.testing.assert_equal(dataset.item_id_map.external_ids, np.array([0, 1, 2]))
        pd.testing.assert_frame_equal(dataset.interactions.df, self.expected_interactions_2x_internal_df)
        assert dataset.user_features is None
        assert dataset.item_features is None

    @pytest.mark.parametrize(
        "prefer_warm_inference_over_cold, expected_user_ids, expected_item_ids",
        (
            (False, [0, 3], [0, 1, 2]),
            (True, [0, 3, 1, 2], [0, 1, 2, 3]),
        ),
    )
    def test_with_features(
        self, prefer_warm_inference_over_cold: bool, expected_user_ids: tp.List[int], expected_item_ids: tp.List[int]
    ) -> None:
        user_features = DenseFeatures(
            values=np.array([[1, 10], [2, 20], [3, 30], [4, 40]]),
            names=("f1", "f2"),
        )
        item_features = SparseFeatures(
            values=sparse.csr_matrix(
                [
                    [3.2, 0, 1],
                    [2.4, 2, 0],
                    [0.0, 0, 1],
                    [1.0, 5, 1],
                ],
            ),
            names=(("f1", None), ("f2", 100), ("f2", 200)),
        )

        dataset = _gen_2x_internal_ids_dataset(
            self.interactions_internal_df, user_features, item_features, prefer_warm_inference_over_cold
        )

        np.testing.assert_equal(dataset.user_id_map.external_ids, np.array(expected_user_ids))
        np.testing.assert_equal(dataset.item_id_map.external_ids, np.array(expected_item_ids))
        pd.testing.assert_frame_equal(dataset.interactions.df, self.expected_interactions_2x_internal_df)

        assert dataset.user_features is not None and dataset.item_features is not None  # for mypy
        np.testing.assert_equal(dataset.user_features.values, user_features.values[expected_user_ids])
        assert dataset.user_features.names == user_features.names
        assert_sparse_matrix_equal(dataset.item_features.values, item_features.values[expected_item_ids])
        assert dataset.item_features.names == item_features.names


class TestCrossValidate:
    def setup_method(self) -> None:
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

        self.models = {
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
