#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

# pylint: disable=attribute-defined-outside-init,consider-using-enumerate
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.dataset.torch_datasets import DSSMItemDataset, DSSMTrainDataset, DSSMUserDataset


class WithFixtures:
    @pytest.fixture
    def interactions_df(self) -> pd.DataFrame:
        interactions = pd.DataFrame(
            [
                ["u1", "i1", 2, "2021-09-09"],
                ["u1", "i2", 2, "2021-09-05"],
                ["u1", "i1", 6, "2021-08-09"],
                ["u2", "i1", 7, "2020-09-09"],
                ["u2", "i5", 9, "2021-09-03"],
                ["u3", "i1", 2, "2021-09-09"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        )
        return interactions

    @pytest.fixture
    def wrong_interactions_df(self) -> pd.DataFrame:
        interactions = pd.DataFrame(
            [
                ["u1", "i1", 2, "2021-09-09"],
                ["u1", "i2", 2, "2021-09-05"],
                ["u1", "i1", 6, "2021-08-09"],
                ["u2", "i1", 7, "2020-09-09"],
                ["u2", "i5", 9, "2021-09-03"],
                ["u3", "i1", 0, "2021-09-09"],
            ],
            columns=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
        )
        return interactions

    @pytest.fixture
    def user_features_df(self) -> pd.DataFrame:
        user_features = pd.DataFrame(
            [
                ["u1", "f2", "f2val1"],
                ["u1", "f1", "f1val2"],
                ["u2", "f1", "f1val1"],
                ["u3", "f1", "f2val2"],
                ["u3", "f2", "f2val3"],
            ],
            columns=["id", "feature", "value"],
        )
        return user_features

    @pytest.fixture
    def item_features_df(self) -> pd.DataFrame:
        item_features = pd.DataFrame(
            [
                ["i2", "f1", "f1val2"],
                ["i2", "f2", "f2val1"],
                ["i5", "f2", "f2val2"],
                ["i5", "f2", "f2val1"],
            ],
            columns=["id", "feature", "value"],
        )
        return item_features

    @pytest.fixture
    def dataset(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> Dataset:
        ds = Dataset.construct(
            interactions_df,
            user_features_df=user_features_df,
            cat_user_features=["f1", "f2"],
            item_features_df=item_features_df,
            cat_item_features=["f1", "f2"],
        )
        return ds

    @pytest.fixture
    def wrong_dataset(
        self,
        wrong_interactions_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> Dataset:
        ds = Dataset.construct(
            wrong_interactions_df,
            user_features_df=user_features_df,
            cat_user_features=["f1", "f2"],
            item_features_df=item_features_df,
            cat_item_features=["f1", "f2"],
        )
        return ds

    @pytest.fixture
    def dataset_no_features(self, interactions_df: pd.DataFrame) -> Dataset:
        ds = Dataset.construct(
            interactions_df,
        )
        return ds

    @pytest.fixture
    def dataset_no_user_features(self, interactions_df: pd.DataFrame, item_features_df: pd.DataFrame) -> Dataset:
        ds = Dataset.construct(
            interactions_df,
            item_features_df=item_features_df,
            cat_item_features=["f1", "f2"],
        )
        return ds


class TestDSSMDataset(WithFixtures):
    def test_wrapper_len_equal_to_len(self, interactions_df: pd.DataFrame, dataset: Dataset) -> None:
        dssm_dataset = DSSMTrainDataset.from_dataset(dataset)
        assert len(dssm_dataset) == dataset.get_user_item_matrix().shape[0]
        assert len(dssm_dataset) == interactions_df[Columns.User].nunique()

    def test_representations_are_equal(self, dataset: Dataset) -> None:
        dssm_dataset = DSSMTrainDataset.from_dataset(dataset)
        assert np.allclose(
            dssm_dataset.interactions.toarray(),
            dataset.get_user_item_matrix().toarray(),
        )
        assert np.allclose(dssm_dataset.items.toarray(), dataset.item_features.get_sparse().toarray())  # type: ignore
        assert np.allclose(dssm_dataset.users.toarray(), dataset.user_features.get_sparse().toarray())  # type: ignore

    def test_getitem_reconstructs_users(self, dataset: Dataset) -> None:
        dssm_dataset = DSSMTrainDataset.from_dataset(dataset)
        all_user_features = []
        all_interactions = []
        for idx in range(len(dssm_dataset)):
            user_features, interactions, _, _ = dssm_dataset[idx]
            all_user_features.append(user_features.view(1, -1))
            all_interactions.append(interactions.view(1, -1))

        all_user_features = torch.cat(all_user_features, 0).numpy()
        all_interactions = torch.cat(all_interactions, 0).numpy()

        ui_matrix = dataset.get_user_item_matrix().toarray()
        assert np.allclose(all_user_features, dataset.user_features.get_sparse().toarray())  # type: ignore
        assert np.allclose(all_interactions, ui_matrix)

    def test_raises_item_features_attribute_error(self, dataset_no_features: Dataset) -> None:
        with pytest.raises(AttributeError):
            DSSMTrainDataset.from_dataset(dataset_no_features)

    def test_raises_user_features_attribute_error(self, dataset_no_user_features: Dataset) -> None:
        with pytest.raises(AttributeError):
            DSSMTrainDataset.from_dataset(dataset_no_user_features)

    def test_raises_value_error(self, wrong_dataset: Dataset) -> None:
        with pytest.raises(ValueError):
            DSSMTrainDataset.from_dataset(wrong_dataset)


class TestUsersDataset(WithFixtures):
    def test_wrapper_len_equal_to_len(self, interactions_df: pd.DataFrame, dataset: Dataset) -> None:
        users_dataset = DSSMUserDataset.from_dataset(dataset)
        assert len(users_dataset) == dataset.get_user_item_matrix().shape[0]
        assert len(users_dataset) == interactions_df[Columns.User].nunique()

    def test_representations_are_equal(self, dataset: Dataset) -> None:
        users_dataset = DSSMUserDataset.from_dataset(dataset)
        assert np.allclose(
            users_dataset.interactions.toarray(),
            dataset.get_user_item_matrix().toarray(),
        )
        assert np.allclose(users_dataset.users.toarray(), dataset.user_features.get_sparse().toarray())  # type: ignore

    def test_getitem_reconstructs_users(self, dataset: Dataset) -> None:
        users_dataset = DSSMUserDataset.from_dataset(dataset)
        all_user_features = []
        all_interactions = []
        for idx in range(len(users_dataset)):
            user_features, interactions = users_dataset[idx]
            all_user_features.append(user_features.view(1, -1))
            all_interactions.append(interactions.view(1, -1))

        all_user_features = torch.cat(all_user_features, 0).numpy()
        all_interactions = torch.cat(all_interactions, 0).numpy()

        ui_matrix = dataset.get_user_item_matrix().toarray()
        assert np.allclose(all_user_features, dataset.user_features.get_sparse().toarray())  # type: ignore
        assert np.allclose(all_interactions, ui_matrix)

    def test_raises_attribute_error(self, dataset_no_features: Dataset) -> None:
        with pytest.raises(AttributeError):
            DSSMUserDataset.from_dataset(dataset_no_features)

    def test_keep_users(self, dataset: Dataset) -> None:
        users_dataset = DSSMUserDataset.from_dataset(dataset, keep_users=[0, 1])
        assert (dataset.user_id_map.internal_ids.shape[0] - len(users_dataset)) == 1

    def test_raises_when_users_and_embeddings_have_different_lengths(self) -> None:
        users = sparse.csr_matrix(np.random.rand(4, 2))
        interactions = sparse.csr_matrix(np.random.rand(3, 5))
        with pytest.raises(ValueError):
            DSSMUserDataset(users, interactions)


class TestItemsDataset(WithFixtures):
    def test_wrapper_len_equal_to_len(self, interactions_df: pd.DataFrame, dataset: Dataset) -> None:
        items_dataset = DSSMItemDataset.from_dataset(dataset)
        assert len(items_dataset) == interactions_df[Columns.Item].nunique()

    def test_representations_are_equal(self, dataset: Dataset) -> None:
        items_dataset = DSSMItemDataset.from_dataset(dataset)
        assert np.allclose(items_dataset.items.toarray(), dataset.item_features.get_sparse().toarray())  # type: ignore

    def test_getitem_reconstructs_items(self, dataset: Dataset) -> None:
        items_dataset = DSSMItemDataset.from_dataset(dataset)
        all_item_features = []
        for idx in range(len(items_dataset)):
            item_features = items_dataset[idx]
            all_item_features.append(item_features.view(1, -1))

        all_item_features = torch.cat(all_item_features, 0).numpy()
        assert np.allclose(all_item_features, dataset.item_features.get_sparse().toarray())  # type: ignore

    def test_raises_attribute_error(self, dataset_no_features: Dataset) -> None:
        with pytest.raises(AttributeError):
            DSSMItemDataset.from_dataset(dataset_no_features)
