import typing as tp
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from implicit.nearest_neighbours import CosineRecommender
from sklearn.ensemble import GradientBoostingClassifier

from rectools import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.exceptions import NotFittedForStageError
from rectools.model_selection import TimeRangeSplitter
from rectools.models import ImplicitItemKNNWrapperModel, PopularModel
from rectools.models.ranking import (
    CandidateFeatureCollector,
    CandidateGenerator,
    CandidateRankingModel,
    PerUserNegativeSampler,
    Reranker,
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

    def test_fail_if_splitter_has_more_than_one_fold(self, dataset: Dataset) -> None:
        splitter = TimeRangeSplitter("1D", n_splits=2)
        with pytest.raises(ValueError, match="Splitter must have only one fold"):
            CandidateRankingModel(
                candidate_generators=[],
                splitter=splitter,
                reranker=Reranker(GradientBoostingClassifier(random_state=123)),
            )

    def test_get_train_with_targets_for_reranker(self, model: PopularModel, dataset: Dataset) -> None:
        candidate_generators = [CandidateGenerator(model, 2, False, False)]
        splitter = TimeRangeSplitter("1D", n_splits=1)
        sampler = PerUserNegativeSampler(1, 32)
        two_stage_model = CandidateRankingModel(
            candidate_generators,
            splitter,
            sampler=sampler,
            reranker=Reranker(GradientBoostingClassifier(random_state=123)),
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

    def test_recommend(self, model: PopularModel, dataset: Dataset) -> None:
        cangen_1 = model
        cangen_2 = ImplicitItemKNNWrapperModel(CosineRecommender())

        scores_fillna_value = -100
        ranks_fillna_value = 3

        candidate_generators = [
            CandidateGenerator(cangen_1, 2, True, True, scores_fillna_value, ranks_fillna_value),
            CandidateGenerator(cangen_2, 2, True, True, scores_fillna_value, ranks_fillna_value),
        ]
        splitter = TimeRangeSplitter("1D", n_splits=1)
        sampler = PerUserNegativeSampler(1, 32)
        two_stage_model = CandidateRankingModel(
            candidate_generators,
            splitter,
            sampler=sampler,
            reranker=Reranker(GradientBoostingClassifier(random_state=123)),
        )
        two_stage_model.fit(dataset)

        actual_reco = two_stage_model.recommend(
            [10, 20, 30], dataset, k=3, filter_viewed=True, force_fit_candidate_generators=True
        )
        expected_reco = pd.DataFrame(
            {
                Columns.User: [10, 10, 20, 20, 20, 30],
                Columns.Item: [14, 15, 12, 15, 13, 13],
                Columns.Score: [
                    0.999,
                    0.412,
                    0.999,
                    0.412,
                    0.000,
                    0.999,
                ],
                Columns.Rank: [1, 2, 1, 2, 3, 1],
            }
        )
        pd.testing.assert_frame_equal(actual_reco, expected_reco, atol=0.001)

    def test_raises_warning_on_context(self, model: PopularModel, dataset: Dataset) -> None:
        two_stage_model = CandidateRankingModel(
            candidate_generators=[CandidateGenerator(model, 2, True, True)],
            splitter=TimeRangeSplitter("1D", n_splits=1),
            sampler=PerUserNegativeSampler(1, 32),
            reranker=Reranker(GradientBoostingClassifier(random_state=123)),
        )
        two_stage_model.fit(dataset)
        context = pd.DataFrame({Columns.User: [10], Columns.Datetime: ["2025-11-30"]})
        with pytest.warns(UserWarning, match="This model does not support context. It will be ignored."):
            two_stage_model.recommend([10], dataset, k=3, filter_viewed=True, context=context)


class TestReranker:
    @pytest.fixture
    def fit_kwargs(self) -> tp.Dict[str, tp.Any]:
        fit_kwargs = {"sample_weight": np.array([1, 2])}
        return fit_kwargs

    @pytest.fixture
    def model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(random_state=123)

    @pytest.fixture
    def reranker(self, model: GradientBoostingClassifier, fit_kwargs: tp.Dict[str, tp.Any]) -> Reranker:
        return Reranker(model, fit_kwargs)

    @pytest.fixture
    def candidates_with_target(self) -> pd.DataFrame:
        candidates_with_target = pd.DataFrame(
            {
                Columns.User: [10, 10],
                Columns.Item: [14, 11],
                Columns.Score: [0.1, 0.2],
                Columns.Target: np.array([0, 1], dtype="int32"),
            }
        )
        return candidates_with_target

    def test_prepare_fit_kwargs(self, reranker: Reranker, candidates_with_target: pd.DataFrame) -> None:
        expected_fit_kwargs = {
            "X": pd.DataFrame(
                {
                    Columns.Score: [0.1, 0.2],
                }
            ),
            "y": pd.Series(np.array([0, 1], dtype="int32"), name=Columns.Target),
            "sample_weight": np.array([1, 2]),
        }

        actual_fit_kwargs = reranker.prepare_fit_kwargs(candidates_with_target)
        pd.testing.assert_frame_equal(actual_fit_kwargs["X"], expected_fit_kwargs["X"])
        pd.testing.assert_series_equal(actual_fit_kwargs["y"], expected_fit_kwargs["y"])
        np.testing.assert_array_equal(actual_fit_kwargs["sample_weight"], expected_fit_kwargs["sample_weight"])

    def test_predict_scores(self, reranker: Reranker, candidates_with_target: pd.DataFrame) -> None:
        reranker.fit(candidates_with_target)
        candidates = candidates_with_target.drop(columns=Columns.Target)

        actual_predict_scores = reranker.predict_scores(candidates)
        expected_predict_scores = np.array([0.000029, 1.000000])
        np.testing.assert_allclose(actual_predict_scores, expected_predict_scores, rtol=0.015, atol=1.5e-05)

    def test_recommend(self) -> None:
        scored_pairs = pd.DataFrame(
            {
                Columns.User: [1, 1, 1, 1, 2, 2, 2],
                Columns.Item: [10, 20, 30, 40, 10, 20, 30],
                Columns.Score: [1, 4, 2, 3, 2, 3, 1],
            }
        )
        actual = Reranker.recommend(scored_pairs, 2, add_rank_col=False)
        expected = pd.DataFrame(
            {Columns.User: [1, 1, 2, 2], Columns.Item: [20, 40, 20, 10], Columns.Score: [4, 3, 3, 2]}
        )
        pd.testing.assert_frame_equal(actual, expected)
