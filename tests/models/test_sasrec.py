# pylint: disable=too-many-lines

import typing as tp
from typing import List, Literal, Union

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer, seed_everything

from rectools.columns import Columns
from rectools.dataset import Dataset, IdMap, Interactions
from rectools.dataset.features import SparseFeatures
from rectools.models.sasrec import (
    CatFeaturesItemNet,
    DotProductBCEHead,
    DotProductGBCEHead,
    DotProductSoftmaxHead,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructor,
    LossName,
    SASRecDataPreparator,
    SASRecModel,
    SequenceDataset,
    SessionEncoderHeadBase,
)
from tests.models.utils import assert_second_fit_refits_model
from tests.testing_utils import assert_feature_set_equal, assert_id_map_equal, assert_interactions_set_equal

from .data import DATASET, INTERACTIONS


def assert_equal_reco_and_correct_scores_ranking(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
    pd.testing.assert_frame_equal(
        actual.sort_values([Columns.User, Columns.Score], ascending=[True, False]).reset_index(drop=True),
        actual,
    )


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

    @pytest.fixture
    def trainer(self) -> Trainer:
        return Trainer(
            max_epochs=2,
            min_epochs=2,
            deterministic=True,
            accelerator="cpu",
        )

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [17, 15, 14, 13, 17, 12, 14, 13],
                        Columns.Rank: [1, 2, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                        Columns.Item: [13, 12, 14, 12, 11, 14, 12, 17, 11],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    # TODO: tests do not pass for multiple GPUs
    def test_u2i(self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=filter_viewed)
        assert_equal_reco_and_correct_scores_ranking(actual, expected)

    @pytest.mark.parametrize("loss", ("softmax", "BCE", "gBCE", DotProductSoftmaxHead()))
    def test_losses_happy_pass(
        self,
        dataset: Dataset,
        trainer: Trainer,
        loss: Union[Literal["softmax", "BCE", "gBCE"], SessionEncoderHeadBase],
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            loss=loss,
            deterministic=True,
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 30, 40])
        actual = model.recommend(users=users, dataset=dataset, k=3, filter_viewed=False)
        expected = pd.DataFrame(
            {
                Columns.User: [10, 10, 10, 30, 30, 30, 40, 40, 40],
                Columns.Item: [13, 12, 14, 12, 11, 14, 12, 17, 11],
                Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
            }
        )
        assert_equal_reco_and_correct_scores_ranking(actual, expected)

    def test_raise_with_incorrect_loss(self, dataset: Dataset, trainer: Trainer) -> None:
        with pytest.raises(ValueError):
            model = SASRecModel(
                n_factors=32,
                n_blocks=2,
                session_max_len=3,
                lr=0.001,
                batch_size=4,
                epochs=2,
                loss="strange",  # type: ignore
                deterministic=True,
                trainer=trainer,
            )
            model.fit(dataset)

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
                        Columns.Item: [13, 17, 11, 11, 13, 17, 17, 11, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_with_whitelist(
        self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
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
        assert_equal_reco_and_correct_scores_ranking(actual, expected)

    @pytest.mark.parametrize(
        "filter_itself,whitelist,expected",
        (
            (
                False,
                None,
                pd.DataFrame(
                    {
                        Columns.TargetItem: [12, 12, 12, 14, 14, 14, 17, 17, 17],
                        Columns.Item: [12, 17, 11, 14, 11, 13, 17, 12, 14],
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
                        Columns.Item: [17, 11, 14, 11, 13, 17, 12, 14, 11],
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
                        Columns.Item: [14, 13, 15, 13, 15, 14, 15, 13],
                        Columns.Rank: [1, 2, 3, 1, 2, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_i2i(
        self,
        dataset: Dataset,
        trainer: Trainer,
        filter_itself: bool,
        whitelist: tp.Optional[np.ndarray],
        expected: pd.DataFrame,
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
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
        pd.testing.assert_frame_equal(actual.drop(columns=Columns.Score), expected)
        pd.testing.assert_frame_equal(
            actual.sort_values([Columns.TargetItem, Columns.Score], ascending=[True, False]).reset_index(drop=True),
            actual,
        )

    def test_second_fit_refits_model(self, dataset_hot_users_items: Dataset, trainer: Trainer) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
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
                        Columns.Item: [14, 12, 17],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [20, 20, 20],
                        Columns.Item: [13, 14, 12],
                        Columns.Rank: [1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_recommend_for_cold_user_with_hot_item(
        self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([20])
        actual = model.recommend(
            users=users,
            dataset=dataset,
            k=3,
            filter_viewed=filter_viewed,
        )
        assert_equal_reco_and_correct_scores_ranking(actual, expected)

    @pytest.mark.parametrize(
        "filter_viewed,expected",
        (
            (
                True,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 20, 20, 20],
                        Columns.Item: [17, 15, 14, 12, 17],
                        Columns.Rank: [1, 2, 1, 2, 3],
                    }
                ),
            ),
            (
                False,
                pd.DataFrame(
                    {
                        Columns.User: [10, 10, 10, 20, 20, 20],
                        Columns.Item: [13, 12, 14, 13, 14, 12],
                        Columns.Rank: [1, 2, 3, 1, 2, 3],
                    }
                ),
            ),
        ),
    )
    def test_warn_when_hot_user_has_cold_items_in_recommend(
        self, dataset: Dataset, trainer: Trainer, filter_viewed: bool, expected: pd.DataFrame
    ) -> None:
        model = SASRecModel(
            n_factors=32,
            n_blocks=2,
            session_max_len=3,
            lr=0.001,
            batch_size=4,
            epochs=2,
            deterministic=True,
            item_net_block_types=(IdEmbeddingsItemNet,),
            trainer=trainer,
        )
        model.fit(dataset=dataset)
        users = np.array([10, 20, 50])
        with pytest.warns() as record:
            actual = model.recommend(
                users=users,
                dataset=dataset,
                k=3,
                filter_viewed=filter_viewed,
                on_unsupported_targets="warn",
            )
            assert_equal_reco_and_correct_scores_ranking(actual, expected)
        assert str(record[0].message) == "1 target users were considered cold because of missing known items"
        assert (
            str(record[1].message)
            == """
                Model `<class 'rectools.models.sasrec.SASRecModel'>` doesn't support recommendations for cold users,
                but some of given users are cold: they are not in the `dataset.user_id_map`
            """
        )

    @pytest.mark.parametrize(
        "loss,expected",
        (
            ("softmax", False),
            ("BCE", True),
            ("gBCE", True),
            (DotProductSoftmaxHead(), False),
            (DotProductBCEHead(), True),
            (DotProductGBCEHead(0.1, 100), True),
        ),
    )
    def test_requires_negatives(self, loss: tp.Union[SessionEncoderHeadBase, LossName], expected: bool) -> None:
        model = SASRecModel(loss=loss)
        assert model.requires_negatives == expected


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
        users = [10, 20]
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
                {
                    "x": torch.tensor([[5, 2, 3], [0, 1, 3], [0, 0, 2]]),
                    "y": torch.tensor([[2, 3, 6], [0, 3, 2], [0, 0, 4]]),
                    "yw": torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
                }
            ),
        ),
    )
    def test_get_dataloader_train(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, train_batch: List
    ) -> None:
        dataset = data_preparator.process_dataset_train(dataset)
        dataloader = data_preparator.get_dataloader_train(dataset)
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, train_batch[key])

    @pytest.mark.parametrize(
        "recommend_batch",
        (({"x": torch.tensor([[2, 3, 6], [1, 3, 2], [0, 2, 4], [0, 0, 6]])}),),
    )
    def test_get_dataloader_recommend(
        self, dataset: Dataset, data_preparator: SASRecDataPreparator, recommend_batch: torch.Tensor
    ) -> None:
        data_preparator.process_dataset_train(dataset)
        dataset = data_preparator.transform_dataset_i2i(dataset)
        dataloader = data_preparator.get_dataloader_recommend(dataset)
        actual = next(iter(dataloader))
        for key, value in actual.items():
            assert torch.equal(value, recommend_batch[key])


class TestIdEmbeddingsItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, n_factors: int) -> None:
        item_id_embeddings = IdEmbeddingsItemNet.from_dataset(DATASET, n_factors=n_factors, dropout_rate=0.5)

        actual_n_items = item_id_embeddings.n_items
        actual_embedding_dim = item_id_embeddings.ids_emb.embedding_dim

        assert actual_n_items == DATASET.item_id_map.size
        assert actual_embedding_dim == n_factors

    @pytest.mark.parametrize("n_items,n_factors", ((2, 10), (4, 100)))
    def test_embedding_shape_after_model_pass(self, n_items: int, n_factors: int) -> None:
        items = torch.from_numpy(np.random.choice(DATASET.item_id_map.internal_ids, size=n_items, replace=False))
        item_id_embeddings = IdEmbeddingsItemNet.from_dataset(DATASET, n_factors=n_factors, dropout_rate=0.5)

        expected_item_ids = item_id_embeddings(items)
        assert expected_item_ids.shape == (n_items, n_factors)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestCatFeaturesItemNet:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset_item_features(self) -> Dataset:
        item_features = pd.DataFrame(
            [
                [11, "f1", "f1val1"],
                [11, "f2", "f2val1"],
                [12, "f1", "f1val1"],
                [12, "f2", "f2val2"],
                [13, "f1", "f1val1"],
                [13, "f2", "f2val3"],
                [14, "f1", "f1val2"],
                [14, "f2", "f2val1"],
                [15, "f1", "f1val2"],
                [15, "f2", "f2val2"],
                [17, "f1", "f1val2"],
                [17, "f2", "f2val3"],
                [16, "f1", "f1val2"],
                [16, "f2", "f2val3"],
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
                [17, "f3", 5],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    def test_feature_catalog(self, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)
        assert isinstance(cat_item_embeddings, CatFeaturesItemNet)
        expected_feature_catalog = torch.arange(0, cat_item_embeddings.n_cat_features)
        assert torch.equal(cat_item_embeddings.feature_catalog, expected_feature_catalog)

    def test_get_dense_item_features(self, dataset_item_features: Dataset) -> None:
        items = torch.from_numpy(
            dataset_item_features.item_id_map.convert_to_internal(INTERACTIONS[Columns.Item].unique())
        )
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(dataset_item_features, n_factors=5, dropout_rate=0.5)

        assert isinstance(cat_item_embeddings, CatFeaturesItemNet)

        actual_feature_dense = cat_item_embeddings.get_dense_item_features(items)
        expected_feature_dense = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        )

        assert torch.equal(actual_feature_dense, expected_feature_dense)

    @pytest.mark.parametrize("n_factors", (10, 100))
    def test_create_from_dataset(self, n_factors: int, dataset_item_features: Dataset) -> None:
        cat_item_embeddings = CatFeaturesItemNet.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5
        )

        assert isinstance(cat_item_embeddings, CatFeaturesItemNet)

        actual_item_features = cat_item_embeddings.item_features
        actual_n_items = cat_item_embeddings.n_items
        actual_n_cat_features = cat_item_embeddings.n_cat_features
        actual_embedding_dim = cat_item_embeddings.category_embeddings.embedding_dim

        expected_item_features = dataset_item_features.item_features

        assert isinstance(expected_item_features, SparseFeatures)
        expected_cat_item_features = expected_item_features.get_cat_features()

        assert_feature_set_equal(actual_item_features, expected_cat_item_features)
        assert actual_n_items == dataset_item_features.item_id_map.size
        assert actual_n_cat_features == len(expected_cat_item_features.names)
        assert actual_embedding_dim == n_factors

    @pytest.mark.parametrize(
        "n_items,n_factors",
        ((2, 10), (4, 100)),
    )
    def test_embedding_shape_after_model_pass(
        self, dataset_item_features: Dataset, n_items: int, n_factors: int
    ) -> None:
        items = torch.from_numpy(
            np.random.choice(dataset_item_features.item_id_map.internal_ids, size=n_items, replace=False)
        )
        cat_item_embeddings = IdEmbeddingsItemNet.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5
        )

        expected_item_ids = cat_item_embeddings(items)
        assert expected_item_ids.shape == (n_items, n_factors)

    @pytest.mark.parametrize(
        "item_features,cat_item_features,make_dense_item_features",
        (
            (None, (), False),
            (
                pd.DataFrame(
                    [
                        [11, "f3", 0],
                        [12, "f3", 1],
                        [13, "f3", 2],
                        [14, "f3", 3],
                        [15, "f3", 4],
                        [17, "f3", 5],
                        [16, "f3", 6],
                    ],
                    columns=["id", "feature", "value"],
                ),
                (),
                False,
            ),
            (
                pd.DataFrame(
                    [
                        [11, 1, 1],
                        [12, 1, 2],
                        [13, 1, 3],
                        [14, 2, 1],
                        [15, 2, 2],
                        [17, 2, 3],
                    ],
                    columns=[Columns.Item, "f1", "f2"],
                ),
                ["f1", "f2"],
                True,
            ),
        ),
    )
    def test_when_cat_item_features_is_none(
        self,
        item_features: tp.Optional[pd.DataFrame],
        cat_item_features: tp.Iterable[str],
        make_dense_item_features: bool,
    ) -> None:
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=cat_item_features,
            make_dense_item_features=make_dense_item_features,
        )
        cat_features_item_net = CatFeaturesItemNet.from_dataset(ds, n_factors=10, dropout_rate=0.5)
        assert cat_features_item_net is None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestItemNetConstructor:
    def setup_method(self) -> None:
        self._seed_everything()

    def _seed_everything(self) -> None:
        torch.use_deterministic_algorithms(True)
        seed_everything(32, workers=True)

    @pytest.fixture
    def dataset_item_features(self) -> Dataset:
        item_features = pd.DataFrame(
            [
                [11, "f1", "f1val1"],
                [11, "f2", "f2val1"],
                [12, "f1", "f1val1"],
                [12, "f2", "f2val2"],
                [13, "f1", "f1val1"],
                [13, "f2", "f2val3"],
                [14, "f1", "f1val2"],
                [14, "f2", "f2val1"],
                [15, "f1", "f1val2"],
                [15, "f2", "f2val2"],
                [16, "f1", "f1val2"],
                [16, "f2", "f2val3"],
                [11, "f3", 0],
                [12, "f3", 1],
                [13, "f3", 2],
                [14, "f3", 3],
                [15, "f3", 4],
                [16, "f3", 6],
            ],
            columns=["id", "feature", "value"],
        )
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            cat_item_features=["f1", "f2"],
        )
        return ds

    def test_catalog(self) -> None:
        item_net = ItemNetConstructor.from_dataset(
            DATASET, n_factors=10, dropout_rate=0.5, item_net_block_types=(IdEmbeddingsItemNet,)
        )
        expected_feature_catalog = torch.arange(0, DATASET.item_id_map.size)
        assert torch.equal(item_net.catalog, expected_feature_catalog)

    @pytest.mark.parametrize(
        "item_net_block_types,n_factors",
        (
            ((IdEmbeddingsItemNet,), 8),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), 16),
            ((CatFeaturesItemNet,), 16),
        ),
    )
    def test_get_all_embeddings(
        self, dataset_item_features: Dataset, item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]], n_factors: int
    ) -> None:
        item_net = ItemNetConstructor.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )
        assert item_net.get_all_embeddings().shape == (item_net.n_items, n_factors)

    @pytest.mark.parametrize(
        "item_net_block_types,make_dense_item_features,expected_n_item_net_blocks",
        (
            ((IdEmbeddingsItemNet,), False, 1),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), False, 2),
            ((IdEmbeddingsItemNet,), True, 1),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), True, 1),
        ),
    )
    def test_correct_number_of_item_net_blocks(
        self,
        dataset_item_features: Dataset,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        make_dense_item_features: bool,
        expected_n_item_net_blocks: int,
    ) -> None:
        if make_dense_item_features:
            item_features = pd.DataFrame(
                [
                    [11, "f3", 0],
                    [12, "f3", 1],
                    [13, "f3", 2],
                    [14, "f3", 3],
                    [15, "f3", 4],
                    [17, "f3", 5],
                    [16, "f3", 6],
                ],
                columns=["id", "feature", "value"],
            )
            ds = Dataset.construct(
                INTERACTIONS,
                item_features_df=item_features,
                make_dense_user_features=make_dense_item_features,
            )
        else:
            ds = dataset_item_features

        item_net: ItemNetConstructor = ItemNetConstructor.from_dataset(
            ds, n_factors=10, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )

        actual_n_items = item_net.n_items
        actual_n_item_net_blocks = len(item_net.item_net_blocks)

        assert actual_n_items == dataset_item_features.item_id_map.size
        assert actual_n_item_net_blocks == expected_n_item_net_blocks

    @pytest.mark.parametrize(
        "item_net_block_types,n_items,n_factors",
        (
            ((IdEmbeddingsItemNet,), 2, 16),
            ((IdEmbeddingsItemNet, CatFeaturesItemNet), 4, 8),
        ),
    )
    def test_embedding_shape_after_model_pass(
        self,
        dataset_item_features: Dataset,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        n_items: int,
        n_factors: int,
    ) -> None:
        items = torch.from_numpy(
            np.random.choice(dataset_item_features.item_id_map.internal_ids, size=n_items, replace=False)
        )
        item_net: ItemNetConstructor = ItemNetConstructor.from_dataset(
            dataset_item_features, n_factors=n_factors, dropout_rate=0.5, item_net_block_types=item_net_block_types
        )

        expected_embeddings = item_net(items)

        assert expected_embeddings.shape == (n_items, n_factors)

    @pytest.mark.parametrize(
        "item_net_block_types,item_features,make_dense_item_features",
        (
            ([], None, False),
            ((CatFeaturesItemNet,), None, False),
            (
                (CatFeaturesItemNet,),
                pd.DataFrame(
                    [
                        [11, 1, 1],
                        [12, 1, 2],
                        [13, 1, 3],
                        [14, 2, 1],
                        [15, 2, 2],
                        [17, 2, 3],
                    ],
                    columns=[Columns.Item, "f1", "f2"],
                ),
                True,
            ),
            (
                (CatFeaturesItemNet,),
                pd.DataFrame(
                    [
                        [11, "f3", 0],
                        [12, "f3", 1],
                        [13, "f3", 2],
                        [14, "f3", 3],
                        [15, "f3", 4],
                        [17, "f3", 5],
                        [16, "f3", 6],
                    ],
                    columns=[Columns.Item, "feature", "value"],
                ),
                False,
            ),
        ),
    )
    def test_raise_when_no_item_net_blocks(
        self,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        item_features: tp.Optional[pd.DataFrame],
        make_dense_item_features: bool,
    ) -> None:
        ds = Dataset.construct(
            INTERACTIONS,
            item_features_df=item_features,
            make_dense_item_features=make_dense_item_features,
        )
        with pytest.raises(ValueError):
            ItemNetConstructor.from_dataset(
                ds, n_factors=10, dropout_rate=0.5, item_net_block_types=item_net_block_types
            )
