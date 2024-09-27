import typing as tp
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.columns import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.models.sasrec import SASRecDataPreparator, SASRecModel, SequenceDataset
from tests.models.utils import assert_second_fit_refits_model
from tests.testing_utils import assert_id_map_equal, assert_interactions_set_equal


@pytest.mark.filterwarnings("ignore::pytorch_lightning.utilities.warnings.PossibleUserWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestSASRecModel:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.fixture
    def dataset(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def dataset_hot_users_items(self, interactions_df: pd.DataFrame) -> Dataset:
        return Dataset.construct(interactions_df[:-4])

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [15, 17, 14, 13, 17, 12, 14, 13],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 14, 15, 14, 13, 12, 12, 17, 14],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_u2i(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=filter_viewed)
        actual[Columns.Item] = actual[Columns.Item].apply(int)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 30, 30, 40],
                        Columns.Item: [17, 13, 17, 13],
                        Columns.Rank: [1, 1, 2, 1],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 11, 17, 13, 11, 17, 17, 13, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        items_to_recommend = np.array([11, 13, 17])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
        )
        actual[Columns.Item] = actual[Columns.Item].apply(int)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [12, 14, 17, 14, 12, 15, 17, 12, 15],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [14, 17, 15, 12, 15, 17, 12, 15, 14],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                True,
                np.array([15, 13, 14]),
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 17, 17, 17],
                        Columns.Item: [14, 15, 13, 15, 13, 15, 14, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self, dataset: Dataset, filter_itself: bool, whitelist: tp.Optional[np.ndarray], expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset)
        target_items = np.array([12, 14, 17])
        actual = model.recommend_to_items(
            target_items=target_items,
            dataset=dataset,
            k=3,
            filter_itself=filter_itself,
            items_to_recommend=whitelist,
        )
        actual[Columns.Item] = actual[Columns.Item].apply(int)
        actual[Columns.TargetItem] = actual[Columns.TargetItem].apply(int)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, dataset_hot_users_items: Dataset) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
        )
        assert_second_fit_refits_model(model, dataset_hot_users_items, pre_fit_callback=self._seed_everything)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [14, 15, 12],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [13, 14, 15],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_recommend_for_cold_user_with_hot_item(
        self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset)
        users = np.array([20])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
        )
        actual[Columns.Item] = actual[Columns.Item].apply(int)
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20, 20],
                        Columns.Item: [15, 17, 14, 15, 12],
                        Columns.Rank: [1, 2, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 20, 20, 20],
                        Columns.Item: [13, 14, 15, 13, 14, 15],
                        Columns.Rank: [1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_warn_when_hot_user_has_cold_items_in_recommend(
        self, dataset: Dataset, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 20, 50])
        with pytest.warns(UserWarning, match="1 target users were considered cold because of missing known items"):
            actual = model.recommend(
                users=users,
                dataset=dataset,
                k=3,
                filter_viewed=filter_viewed,
                on_unsupported_targets="warn",
            )
            actual[Columns.Item] = actual[Columns.Item].apply(int)
            pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
            pd.testing.assert_frame_equal(
                actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
                actual,
            )


class TestSequenceDataset:

    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 4, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 8, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return interactions_df

    @pytest.mark.parametrize(
        "expected_sessions, expected_weights",
        (([[14, 11, 12, 13], [15, 12, 11], [11, 17], [16]], [[1, 1, 4, 1], [1, 2, 1], [1, 8], [1]]),),
    )
    def test_from_interactions(
        self, interactions_df: pd.DataFrame, expected_sessions: List[List[int]], expected_weights: List[List[float]]
    ) -> None:
        actual = SequenceDataset.from_interactions(interactions_df)
        assert len(actual.sessions) == len(expected_sessions)
        assert all(
            actual_list == expected_list for actual_list, expected_list in zip(actual.sessions, expected_sessions)
        )
        assert len(actual.weights) == len(expected_weights)
        assert all(actual_list == expected_list for actual_list, expected_list in zip(actual.weights, expected_weights))


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestSASRecDataPreparator:

    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset(self) -> Dataset:
        interactions_df = pd.DataFrame(
            [
                [10, 13, 1, "2021-11-30"],
                [10, 11, 1, "2021-11-29"],
                [10, 12, 1, "2021-11-29"],
                [30, 11, 1, "2021-11-27"],
                [30, 12, 2, "2021-11-26"],
                [30, 15, 1, "2021-11-25"],
                [40, 11, 1, "2021-11-25"],
                [40, 17, 1, "2021-11-26"],
                [50, 16, 1, "2021-11-25"],
                [10, 14, 1, "2021-11-28"],
                [10, 16, 1, "2021-11-27"],
                [20, 13, 9, "2021-11-28"],
            ],
            columns=Columns.Interactions,
        )
        return Dataset.construct(interactions_df)

    @pytest.fixture
    def data_preparator(self) -> SASRecDataPreparator:
        return SASRecDataPreparator(session_max_len=3, batch_size=4, dataloader_num_workers=0)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([30, 40, 10]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 1, 1.0, "2021-11-25"],
                            [1, 2, 1.0, "2021-11-25"],
                            [0, 3, 2.0, "2021-11-26"],
                            [1, 4, 1.0, "2021-11-26"],
                            [0, 2, 1.0, "2021-11-27"],
                            [2, 5, 1.0, "2021-11-28"],
                            [2, 2, 1.0, "2021-11-29"],
                            [2, 3, 1.0, "2021-11-29"],
                            [2, 6, 1.0, "2021-11-30"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_process_dataset_train(
        self,
        dataset: Dataset,
        data_preparator: SASRecDataPreparator,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        actual = data_preparator.process_dataset_train(dataset)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([10, 20]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 6, 1.0, "2021-11-30"],
                            [0, 2, 1.0, "2021-11-29"],
                            [0, 3, 1.0, "2021-11-29"],
                            [0, 5, 1.0, "2021-11-28"],
                            [1, 6, 9.0, "2021-11-28"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_transform_dataset_u2i(
        self,
        dataset: Dataset,
        data_preparator: SASRecDataPreparator,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        users = [10, 20, 50]
        actual = data_preparator.transform_dataset_u2i(dataset, users)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    @pytest.mark.parametrize(
        "expected_user_id_map, expected_item_id_map, expected_interactions",
        (
            (
                IdMap.from_values([10, 30, 40, 50, 20]),
                IdMap.from_values(["PAD", 15, 11, 12, 17, 14, 13]),
                Interactions(
                    pd.DataFrame(
                        [
                            [0, 6, 1.0, "2021-11-30"],
                            [0, 2, 1.0, "2021-11-29"],
                            [0, 3, 1.0, "2021-11-29"],
                            [1, 2, 1.0, "2021-11-27"],
                            [1, 3, 2.0, "2021-11-26"],
                            [1, 1, 1.0, "2021-11-25"],
                            [2, 2, 1.0, "2021-11-25"],
                            [2, 4, 1.0, "2021-11-26"],
                            [0, 5, 1.0, "2021-11-28"],
                            [4, 6, 9.0, "2021-11-28"],
                        ],
                        columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
                    ),
                ),
            ),
        ),
    )
    def test_tranform_dataset_i2i(
        self,
        dataset: Dataset,
        data_preparator: SASRecDataPreparator,
        expected_interactions: Interactions,
        expected_item_id_map: IdMap,
        expected_user_id_map: IdMap,
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        actual = data_preparator.transform_dataset_i2i(dataset)
        assert_id_map_equal(actual.user_id_map, expected_user_id_map)
        assert_id_map_equal(actual.item_id_map, expected_item_id_map)
        assert_interactions_set_equal(actual.interactions, expected_interactions)

    @pytest.mark.parametrize(
        "train_batch",
        (
            (
                [
                    torch.tensor([[5, 2, 3], [0, 1, 3], [0, 0, 2]]),
                    torch.tensor([[2, 3, 6], [0, 3, 2], [0, 0, 4]]),
                    torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
                ]
            ),
        ),
    )
    def test_get_dataloader_train(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, train_batch: List
    ) -> None:
        dataset = data_preparator.process_dataset_train(dataset)
        dataloader = data_preparator.get_dataloader_train(dataset)
        actual = next(iter(dataloader))
        for i, value in enumerate(actual):
            assert torch.equal(value, train_batch[i])

    @pytest.mark.parametrize(
        "recommend_batch",
        ((torch.tensor([[2, 3, 6], [1, 3, 2], [0, 2, 4], [0, 0, 6]])),),
    )
    def test_get_dataloader_recommend(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, recommend_batch: torch.Tensor
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataset = data_preparator.transform_dataset_i2i(dataset)
        dataloader = data_preparator.get_dataloader_recommend(dataset)
        actual = next(iter(dataloader))
        assert torch.equal(actual, recommend_batch)
