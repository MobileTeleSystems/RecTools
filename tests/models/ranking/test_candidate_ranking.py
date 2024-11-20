import typing as tp
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRanker

from rectools import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.exceptions import NotFittedForStageError
from rectools.model_selection import TimeRangeSplitter
from rectools.models import PopularModel
from rectools.models.ranking import (
    CandidateFeatureCollector,
    CandidateGenerator,
    CandidateRankingModel,
    CatBoostReranker,
    PerUserNegativeSampler,
)


class TestPerUserNegativeSampler:
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        data = {
            Columns.User: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            Columns.Item: [101, 102, 103, 104, 201, 202, 203, 204, 301, 302, 303, 304],
            Columns.Score: [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6],
            Columns.Rank: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            Columns.Target: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
        return pd.DataFrame(data)

    @pytest.mark.parametrize("n_negatives", (1, 2))
    def test_sample_negatives(self, sample_data: pd.DataFrame, n_negatives: int) -> None:
        sampler = PerUserNegativeSampler(n_negatives=n_negatives, random_state=42)
        sampled_df = sampler.sample_negatives(sample_data)

        # Check if the resulting DataFrame has the correct columns
        assert set(sampled_df.columns) == set(sample_data.columns)

        # Check if the number of negatives per user is correct
        n_negatives_per_user = sampled_df.groupby(Columns.User)[Columns.Target].agg(lambda target: (target == 0).sum())
        assert (n_negatives_per_user == n_negatives).all()

        # Check if positives were not changed
        pd.testing.assert_frame_equal(
            sampled_df[sampled_df[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
            sample_data[sample_data[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
        )

    def test_sample_negatives_with_insufficient_negatives(self, sample_data: pd.DataFrame) -> None:
        # Modify sample_data to have insufficient negatives for user 1
        sample_data.loc[sample_data[Columns.User] == 1, Columns.Target] = [1, 0, 1, 0]

        sampler = PerUserNegativeSampler(n_negatives=3, random_state=42)
        sampled_df = sampler.sample_negatives(sample_data)

        # Check if the resulting DataFrame has the correct columns
        assert set(sampled_df.columns) == set(sample_data.columns)

        # Check if the number of negatives per user is correct
        n_negatives_per_user = sampled_df.groupby(Columns.User)[Columns.Target].agg(lambda target: (target == 0).sum())
        assert n_negatives_per_user.to_list() == [2, 3, 3]

        # Check if positives were not changed
        pd.testing.assert_frame_equal(
            sampled_df[sampled_df[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
            sample_data[sample_data[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
        )


class TestCandidateGenerator:
    @pytest.fixture
    def dataset(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [70, 11, 1, "2021-11-30"],
                [70, 12, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-30"],
                [10, 12, 1, "2021-11-29"],
                [10, 13, 9, "2021-11-28"],
                [20, 11, 1, "2021-11-27"],
                [20, 14, 2, "2021-11-26"],
                [30, 11, 1, "2021-11-24"],
                [30, 12, 1, "2021-11-23"],
                [30, 14, 1, "2021-11-23"],
                [30, 15, 5, "2021-11-21"],
                [40, 11, 1, "2021-11-20"],
                [40, 12, 1, "2021-11-19"],
            ],
            columns=Columns.Interactions,
        )
        user_id_map = IdMap.from_values([10, 20, 30, 40, 50, 60, 70, 80])
        item_id_map = IdMap.from_values([11, 12, 13, 14, 15, 16])
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map)
        return Dataset(user_id_map, item_id_map, interactions)

    @pytest.fixture
    def users(self) -> tp.List[int]:
        return [10, 20, 30]

    @pytest.fixture
    def model(self) -> PopularModel:
        return PopularModel()

    @pytest.fixture
    def generator(self, model: PopularModel) -> CandidateGenerator:
        return CandidateGenerator(model, 2, False, False)

    @pytest.mark.parametrize("for_train", (True, False))
    def test_not_fitted_errors(
        self, for_train: bool, dataset: Dataset, generator: CandidateGenerator, users: tp.List[int]
    ) -> None:
        with pytest.raises(NotFittedForStageError):
            generator.generate_candidates(users, dataset, filter_viewed=True, for_train=for_train)

    @pytest.mark.parametrize("for_train", (True, False))
    def test_not_fitted_errors_when_fitted_to_opposite_case(
        self, for_train: bool, dataset: Dataset, generator: CandidateGenerator, users: tp.List[int]
    ) -> None:
        generator.fit(dataset, for_train=not for_train)
        with pytest.raises(NotFittedForStageError):
            generator.generate_candidates(users, dataset, filter_viewed=True, for_train=for_train)

    @pytest.mark.parametrize("for_train", (True, False))
    @pytest.mark.parametrize(
        ("filter_viewed", "expected"),
        (
            (True, pd.DataFrame({Columns.User: [10, 10, 20, 20, 30], Columns.Item: [14, 15, 12, 13, 13]})),
            (False, pd.DataFrame({Columns.User: [10, 10, 20, 20, 30, 30], Columns.Item: [11, 12, 11, 12, 11, 12]})),
        ),
    )
    def test_happy_path(
        self,
        for_train: bool,
        dataset: Dataset,
        generator: CandidateGenerator,
        users: tp.List[int],
        filter_viewed: bool,
        expected: pd.DataFrame,
    ) -> None:
        generator.fit(dataset, for_train=for_train)
        actual = generator.generate_candidates(users, dataset, filter_viewed=filter_viewed, for_train=for_train)
        pd.testing.assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("keep_scores", (True, False))
    @pytest.mark.parametrize("keep_ranks", (True, False))
    def test_columns(
        self, dataset: Dataset, model: PopularModel, users: tp.List[int], keep_scores: bool, keep_ranks: bool
    ) -> None:
        generator = CandidateGenerator(model, 2, keep_ranks=keep_ranks, keep_scores=keep_scores)
        generator.fit(dataset, for_train=True)
        candidates = generator.generate_candidates(users, dataset, filter_viewed=True, for_train=True)

        columns = candidates.columns.to_list()
        assert Columns.User in columns
        assert Columns.Item in columns

        if keep_scores:
            assert Columns.Score in columns
        else:
            assert Columns.Score not in columns

        if keep_ranks:
            assert Columns.Rank in columns
        else:
            assert Columns.Rank not in columns


class TestCandidateFeatureCollector:
    def test_happy_path(self) -> None:
        feature_collector = CandidateFeatureCollector()
        candidates = pd.DataFrame(
            {
                Columns.User: [1, 1, 2, 2, 3, 3],
                Columns.Item: [10, 20, 30, 40, 50, 60],
                "some_model_rank": [1, 2, 1, 2, 1, 2],
            }
        )
        dataset = MagicMock()
        fold_info = MagicMock()
        actual = feature_collector.collect_features(candidates, dataset, fold_info)
        pd.testing.assert_frame_equal(candidates, actual)


class TestCandidateRankingModel:
    @pytest.fixture
    def dataset(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [70, 11, 1, "2021-11-30"],
                [70, 12, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-30"],
                [10, 12, 1, "2021-11-29"],
                [10, 13, 9, "2021-11-28"],
                [20, 11, 1, "2021-11-27"],
                [20, 14, 2, "2021-11-26"],
                [30, 11, 1, "2021-11-24"],
                [30, 12, 1, "2021-11-23"],
                [30, 14, 1, "2021-11-23"],
                [30, 15, 5, "2021-11-21"],
                [40, 11, 1, "2021-11-20"],
                [40, 12, 1, "2021-11-19"],
            ],
            columns=Columns.Interactions,
        )
        user_id_map = IdMap.from_values([10, 20, 30, 40, 50, 60, 70, 80])
        item_id_map = IdMap.from_values([11, 12, 13, 14, 15, 16])
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map)
        return Dataset(user_id_map, item_id_map, interactions)

    @pytest.fixture
    def users(self) -> tp.List[int]:
        return [10, 20, 30]

    @pytest.fixture
    def model(self) -> PopularModel:
        return PopularModel()

    def test_get_train_with_targets_for_reranker_happy_path(self, model: PopularModel, dataset: Dataset) -> None:
        candidate_generators = [CandidateGenerator(model, 2, False, False)]
        splitter = TimeRangeSplitter("1D", n_splits=1)
        sampler = PerUserNegativeSampler(1, 32)
        two_stage_model = CandidateRankingModel(
            candidate_generators,
            splitter,
            sampler=sampler,
            reranker=CatBoostReranker(CatBoostRanker(random_state=32, verbose=False)),
        )
        actual = two_stage_model.get_train_with_targets_for_reranker(dataset)
        expected = pd.DataFrame(
            {
                Columns.User: [10, 10],
                Columns.Item: [14, 11],
                Columns.Target: np.array([0, 1], dtype="int32"),
            }
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_recommend_happy_path(self, model: PopularModel, dataset: Dataset) -> None:
        candidate_generators = [CandidateGenerator(model, 2, True, True)]
        splitter = TimeRangeSplitter("1D", n_splits=1)
        sampler = PerUserNegativeSampler(1, 32)
        two_stage_model = CandidateRankingModel(
            candidate_generators,
            splitter,
            sampler=sampler,
            reranker=CatBoostReranker(CatBoostRanker(random_state=32, verbose=False)),
        )
        two_stage_model.fit(dataset)

        actual = two_stage_model.recommend(
            [10, 20, 30],
            dataset,
            k=3,
            filter_viewed=True,
        )
        expected = pd.DataFrame(
            {
                Columns.User: [10, 10, 20, 20, 30],
                Columns.Item: [14, 15, 12, 13, 13],
                Columns.Score: [
                    -0.1924785067861987,
                    -23.396888326641466,
                    23.396888326641466,
                    -23.396888326641466,
                    -0.1924785067861987,
                ],
                Columns.Rank: [1, 2, 1, 2, 1],
            }
        )
        pd.testing.assert_frame_equal(actual, expected, atol=0.001)
