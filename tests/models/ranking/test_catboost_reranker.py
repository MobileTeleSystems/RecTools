import typing as tp

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from implicit.nearest_neighbours import CosineRecommender
from pytest import FixtureRequest

from rectools import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.model_selection import TimeRangeSplitter
from rectools.models import ImplicitItemKNNWrapperModel, PopularModel
from rectools.models.ranking import CandidateGenerator, CandidateRankingModel, CatBoostReranker, PerUserNegativeSampler


class TestCatBoostReranker:
    @pytest.fixture
    def fit_kwargs(self) -> tp.Dict[str, tp.Any]:
        fit_kwargs = {"early_stopping_rounds": 10}
        return fit_kwargs

    @pytest.fixture
    def pool_kwargs(self) -> tp.Dict[str, tp.Any]:
        pool_kwargs = {"cat_features": ["age", "sex"]}
        return pool_kwargs

    @pytest.fixture
    def reranker_catboost_classifier(
        self, pool_kwargs: tp.Dict[str, tp.Any], fit_kwargs: tp.Dict[str, tp.Any]
    ) -> CatBoostReranker:
        return CatBoostReranker(
            CatBoostClassifier(verbose=False, random_state=123), pool_kwargs=pool_kwargs, fit_kwargs=fit_kwargs
        )

    @pytest.fixture
    def reranker_catboost_ranker(
        self, pool_kwargs: tp.Dict[str, tp.Any], fit_kwargs: tp.Dict[str, tp.Any]
    ) -> CatBoostReranker:
        return CatBoostReranker(
            CatBoostRanker(verbose=False, random_state=123), pool_kwargs=pool_kwargs, fit_kwargs=fit_kwargs
        )

    @pytest.fixture
    def candidates_with_target(self) -> pd.DataFrame:
        candidates_with_target = pd.DataFrame(
            {
                Columns.User: [10, 10],
                Columns.Item: [14, 11],
                Columns.Score: [0.1, 0.2],
                "sex": ["M", "F"],
                "age": ["18_24", "25_34"],
                Columns.Target: [0, 1],
            }
        )
        return candidates_with_target

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

    @pytest.mark.parametrize(
        "reranker_fixture, expected_training_pool",
        [
            (
                "reranker_catboost_ranker",
                Pool(
                    data=pd.DataFrame(
                        {
                            Columns.Score: [0.1, 0.2],
                            "sex": ["M", "F"],
                            "age": ["18_24", "25_34"],
                        }
                    ),
                    label=[0, 1],
                    cat_features=["age", "sex"],
                ),
            ),
            (
                "reranker_catboost_classifier",
                Pool(
                    data=pd.DataFrame(
                        {
                            Columns.Score: [0.1, 0.2],
                            "sex": ["M", "F"],
                            "age": ["18_24", "25_34"],
                        }
                    ),
                    label=[0, 1],
                    cat_features=["age", "sex"],
                    group_id=[10, 10],
                ),
            ),
        ],
    )
    def test_prepare_training_pool(
        self,
        request: FixtureRequest,
        reranker_fixture: str,
        expected_training_pool: Pool,
        candidates_with_target: pd.DataFrame,
    ) -> None:
        reranker = request.getfixturevalue(reranker_fixture)
        actual_training_pool = reranker.prepare_training_pool(candidates_with_target)

        expected_labels = expected_training_pool.get_label()
        actual_labels = actual_training_pool.get_label()
        np.testing.assert_array_equal(expected_labels, actual_labels)

        expected_cat_features = expected_training_pool.get_cat_feature_indices()
        actual_cat_features = actual_training_pool.get_cat_feature_indices()
        np.testing.assert_array_equal(expected_cat_features, actual_cat_features)

        expected_feature_names = expected_training_pool.get_feature_names()
        actual_feature_names = actual_training_pool.get_feature_names()
        np.testing.assert_array_equal(expected_feature_names, actual_feature_names)

    @pytest.mark.parametrize(
        "reranker_fixture, expected_predict_scores",
        [
            (
                "reranker_catboost_ranker",
                np.array([-23.397, 23.397]),
            ),
            (
                "reranker_catboost_classifier",
                np.array([0.334, 0.665]),
            ),
        ],
    )
    def test_predict_scores(
        self,
        request: FixtureRequest,
        reranker_fixture: str,
        expected_predict_scores: np.ndarray,
        candidates_with_target: pd.DataFrame,
    ) -> None:
        reranker = request.getfixturevalue(reranker_fixture)
        reranker.fit(candidates_with_target)

        candidates = candidates_with_target.drop(columns=Columns.Target)
        actual_predict_scores = reranker.predict_scores(candidates)
        np.testing.assert_allclose(actual_predict_scores, expected_predict_scores, atol=0.0007)

    @pytest.mark.parametrize(
        "reranker, expected_reco",
        [
            (
                CatBoostReranker(CatBoostRanker(random_state=32, verbose=False)),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20, 20, 30],
                        Columns.Item: [14, 15, 12, 15, 13, 13],
                        Columns.Score: [
                            11.909,
                            1.020,
                            23.396,
                            1.020,
                            -23.396,
                            11.909,
                        ],
                        Columns.Rank: [1, 2, 1, 2, 3, 1],
                    }
                ),
            ),
            (
                CatBoostReranker(CatBoostClassifier(random_state=32, verbose=False)),
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20, 20, 30],
                        Columns.Item: [14, 15, 12, 15, 13, 13],
                        Columns.Score: [0.588, 0.505, 0.665, 0.505, 0.334, 0.588],
                        Columns.Rank: [1, 2, 1, 2, 3, 1],
                    }
                ),
            ),
        ],
    )
    def test_recommend(self, reranker: CatBoostReranker, expected_reco: pd.DataFrame, dataset: Dataset) -> None:
        cangen_1 = PopularModel()
        cangen_2 = ImplicitItemKNNWrapperModel(CosineRecommender())

        scores_fillna_value = -100
        ranks_fillna_value = 3

        candidate_generators = [
            CandidateGenerator(cangen_1, 2, True, True, scores_fillna_value, ranks_fillna_value),
            CandidateGenerator(cangen_2, 2, True, True, scores_fillna_value, ranks_fillna_value),
        ]
        splitter = TimeRangeSplitter("1D", n_splits=1)
        sampler = PerUserNegativeSampler(1, 32)
        two_stage_model_ranker = CandidateRankingModel(
            candidate_generators,
            splitter,
            sampler=sampler,
            reranker=reranker,
        )
        two_stage_model_ranker.fit(dataset)

        actual_reco_ranker = two_stage_model_ranker.recommend(
            [10, 20, 30], dataset, k=3, filter_viewed=True, force_fit_candidate_generators=True
        )
        pd.testing.assert_frame_equal(actual_reco_ranker, expected_reco, atol=0.001)
