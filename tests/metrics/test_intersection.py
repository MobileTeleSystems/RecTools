from typing import List

import numpy as np
import pandas as pd
import pytest

from rectools import Columns
from rectools.metrics.intersection import Intersection


class TestIntersection:

    @pytest.mark.parametrize(
        "k,ref_k,expected_users,expected_intersection",
        (
            (2, 2, [1, 2, 4, 5], [0.0, 1.0, 1 / 2, 1 / 3]),
            (3, None, [1, 2, 4, 5], [1 / 2, 1.0, 1 / 2, 2 / 3]),
            (3, 6, [1, 2, 4, 5], [1 / 2, 1.0, 1.0, 1.0]),
        ),
    )
    def test_calc(self, k: int, ref_k: int, expected_users: List[int], expected_intersection: List[float]) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5],
                Columns.Item: [1, 2, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4],
                Columns.Rank: [3, 1, 1, 7, 5, 2, 1, 8, 1, 2, 2, 9],
            }
        )
        ref_reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5],
                Columns.Item: [1, 3, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4],
                Columns.Rank: [3, 2, 1, 1, 4, 5, 1, 2, 3, 4, 1, 7],
            }
        )

        intersection_metric = Intersection(k=k, ref_k=ref_k)

        metric_per_user = intersection_metric.calc_per_user(reco, ref_reco)
        expected_metric_per_user = pd.Series(
            expected_intersection,
            index=pd.Series(expected_users, name=Columns.User),
            dtype=float,
        )
        pd.testing.assert_series_equal(metric_per_user, expected_metric_per_user)

        metric = intersection_metric.calc(reco, ref_reco)
        assert np.allclose(metric, expected_metric_per_user.mean())

    def test_when_no_ref_reco(self) -> None:
        reco = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 2, 2, 3, 4],
                Columns.Item: [1, 2, 3, 1, 2, 1, 1],
                Columns.Rank: [1, 2, 3, 1, 2, 1, 1],
            }
        )
        empty_ref_reco = pd.DataFrame(columns=[Columns.User, Columns.Item, Columns.Rank], dtype=int)

        intersection_metric = Intersection(k=2)

        metric_per_user = intersection_metric.calc_per_user(reco, empty_ref_reco)
        expected_metric_per_user = pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        pd.testing.assert_series_equal(metric_per_user, expected_metric_per_user)

        metric = intersection_metric.calc(reco, empty_ref_reco)
        assert np.isnan(metric)
