import typing as tp

import pandas as pd
import pytest

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset
from rectools.metrics import Precision, Recall
from rectools.metrics.base import MetricAtK
from rectools.model_selection import LastNSplitter, cross_validate
from rectools.models import PopularModel, RandomModel

a = pytest.approx


@pytest.mark.parametrize(
    "items_to_recommend, expected_metrics",
    (
        (
            None,
            [
                {"model": "popular", "i_split": 0, "precision@2": 0.5, "recall@1": 0.5},
                {"model": "random", "i_split": 0, "precision@2": 0.25, "recall@1": 0.0},
                {"model": "popular", "i_split": 1, "precision@2": 0.375, "recall@1": 0.25},
                {"model": "random", "i_split": 1, "precision@2": 0.5, "recall@1": 0.5},
            ],
        ),
        (
            [11, 14],
            [
                {"model": "popular", "i_split": 0, "precision@2": 0.25, "recall@1": 0.5},
                {"model": "random", "i_split": 0, "precision@2": 0.25, "recall@1": 0.5},
                {"model": "popular", "i_split": 1, "precision@2": 0.125, "recall@1": 0.25},
                {"model": "random", "i_split": 1, "precision@2": 0.25, "recall@1": 0.0},
            ],
        ),
    ),
)
def test_happy_path(
    items_to_recommend: tp.Optional[ExternalIds], expected_metrics: tp.List[tp.Dict[str, tp.Any]]
) -> None:
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
        "random": RandomModel(random_state=42),
    }

    actual = cross_validate(
        dataset=dataset,
        splitter=splitter,
        metrics=metrics,
        models=models,
        k=2,
        filter_viewed=False,
        items_to_recommend=items_to_recommend,
    )

    expected = {
        "splits": [
            {"i_split": 0, "test": 2, "test_items": 2, "test_users": 2, "train": 2, "train_items": 2, "train_users": 2},
            {"i_split": 1, "test": 4, "test_items": 3, "test_users": 4, "train": 6, "train_items": 2, "train_users": 4},
        ],
        "metrics": expected_metrics,
    }

    assert actual == expected
