import typing as tp

import pandas as pd
import pytest

from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import Precision, Recall
from rectools.metrics.base import MetricAtK
from rectools.model_selection import LastNSplitter, cross_validate
from rectools.models import PopularModel, PureSVDModel


def test_happy_path() -> None:
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
    dataset = Dataset.construct(interactions_df)

    splitter = LastNSplitter(n=1, n_splits=2, filter_cold_items=False, filter_already_seen=False)

    metrics: tp.Dict[str, MetricAtK] = {
        "precision@2": Precision(2),
        "recall@1": Recall(1),
    }

    models = {
        "popular": PopularModel(),
        "pure_svd": PureSVDModel(factors=2),
    }

    actual = cross_validate(
        dataset=dataset,
        splitter=splitter,
        metrics=metrics,
        models=models,
        k=2,
        filter_viewed=False,
        items_to_recommend=None,
    )

    a = pytest.approx
    expected = {
        "splits": [
            {"i_split": 0, "test": 2, "test_items": 2, "test_users": 2, "train": 2, "train_items": 2, "train_users": 2},
            {"i_split": 1, "test": 4, "test_items": 3, "test_users": 4, "train": 6, "train_items": 2, "train_users": 4},
        ],
        "metrics": [
            {"i_split": 0, "model": "popular", "precision@2": a(0.5), "recall@1": a(0.5)},
            {"i_split": 0, "model": "pure_svd", "precision@2": a(0.5), "recall@1": a(0.0)},
            {"i_split": 1, "model": "popular", "precision@2": a(0.375), "recall@1": a(0.25)},
            {"i_split": 1, "model": "pure_svd", "precision@2": a(0.125), "recall@1": a(0.25)},
        ],
    }

    assert actual == expected
