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

"""Special datasets used in neural models."""

from __future__ import annotations

import typing as tp

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset as TorchDataset

from .dataset import Dataset

DSSMTrainDatasetT = tp.TypeVar("DSSMTrainDatasetT", bound="DSSMTrainDatasetBase")
DSSMItemDatasetT = tp.TypeVar("DSSMItemDatasetT", bound="DSSMItemDatasetBase")
DSSMUserDatasetT = tp.TypeVar("DSSMUserDatasetT", bound="DSSMUserDatasetBase")


class DSSMTrainDatasetBase(TorchDataset[tp.Any]):
    """Base class for DSSM training datasets. Used only for type hinting."""

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError()

    @classmethod
    def from_dataset(cls: tp.Type[DSSMTrainDatasetT], dataset: Dataset) -> DSSMTrainDatasetT:
        raise NotImplementedError()


class DSSMTrainDataset(DSSMTrainDatasetBase):
    """
    Torch dataset wrapper for `rectools.dataset.dataset.Dataset`.
    Implements `torch.utils.data.Dataset` for subsequent usage with
    `torch.utils.data.DataLoader`. Does the following: for a given index
    takes a row of user interactions, a row of user features and samples
    one positive and one negative items and then returns them as tensors.

    This class is intended for internal usage or advanced users who want
    to implement more sophisticated sampling logic.

    Parameters
    ----------
    items : csr_matrix
        Item features.
    users : csr_matrix
        User features.
    interactions : csr_matrix
        Interactions matrix.
    """

    def __init__(
        self,
        items: sparse.csr_matrix,
        users: sparse.csr_matrix,
        interactions: sparse.csr_matrix,
    ) -> None:
        self.items = items
        self.users = users
        self.interactions = interactions
        if not self.interactions.sum(1).all() or (self.interactions < 0).sum(1).any():
            raise ValueError(
                "Impossible to sample from a row that either contains only negative items"
                " or contains any negatively signed integers."
                "Make sure that all rows from interactions have at least 1 positive item"
            )

    @classmethod
    def from_dataset(cls: tp.Type[DSSMTrainDatasetT], dataset: Dataset) -> DSSMTrainDatasetT:
        ui_matrix = dataset.get_user_item_matrix()

        # We take hot here since this dataset is used for fit only
        item_features = dataset.get_hot_item_features()
        user_features = dataset.get_hot_user_features()
        if item_features is None:
            raise AttributeError("Item features attribute of dataset could not be None")
        if user_features is None:
            raise AttributeError("User features attribute of dataset could not be None")
        return cls(items=item_features.get_sparse(), users=user_features.get_sparse(), interactions=ui_matrix)

    def __len__(self) -> int:
        return self.interactions.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tp.Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        interactions_vec = self.interactions[idx].toarray().flatten()
        probabilities = interactions_vec / interactions_vec.sum()
        pos_i = np.random.choice(np.arange(self.interactions.shape[1]), p=probabilities)
        neg_i = np.random.choice(np.arange(self.interactions.shape[1]))

        user_features = torch.FloatTensor(self.users[idx].toarray().flatten())
        interactions = torch.FloatTensor(interactions_vec)
        pos = torch.FloatTensor(self.items[pos_i].toarray().flatten())
        neg = torch.FloatTensor(self.items[neg_i].toarray().flatten())

        return user_features, interactions, pos, neg


class DSSMItemDatasetBase(TorchDataset[tp.Any]):
    """Base class for DSSM item datasets. Used only for type hinting."""

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError()

    @classmethod
    def from_dataset(cls: tp.Type[DSSMItemDatasetT], dataset: Dataset) -> DSSMItemDatasetT:
        raise NotImplementedError()


class DSSMItemDataset(DSSMItemDatasetBase):
    """
    Torch dataset wrapper for `rectools.dataset.dataset.Dataset`.
    Implements `torch.utils.data.Dataset` for subsequent usage with
    `torch.utils.data.DataLoader`. Does the following: for a given index
    takes a row of item features and then returns them as tensors.

    This class is intended for internal usage or advanced users.
    """

    def __init__(self, items: sparse.csr_matrix):
        self.items = items

    @classmethod
    def from_dataset(cls: tp.Type[DSSMItemDatasetT], dataset: Dataset) -> DSSMItemDatasetT:
        # We take all features here since this dataset is used for recommend only, not for fit
        if dataset.item_features is not None:
            return cls(dataset.item_features.get_sparse())
        raise AttributeError("Item features attribute of dataset could not be None")

    def __len__(self) -> int:
        return self.items.shape[0]

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        return torch.FloatTensor(self.items[idx].toarray().flatten())


class DSSMUserDatasetBase(TorchDataset[tp.Any]):
    """Base class for DSSM training datasets. Used only for type hinting."""

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise NotImplementedError()

    @classmethod
    def from_dataset(
        cls: tp.Type[DSSMUserDatasetT],
        dataset: Dataset,
        keep_users: tp.Optional[tp.Sequence[int]] = None,
    ) -> DSSMUserDatasetT:
        raise NotImplementedError()


class DSSMUserDataset(DSSMUserDatasetBase):
    """
    Torch dataset wrapper for `rectools.dataset.dataset.Dataset`.
    Implements `torch.utils.data.Dataset` for subsequent usage with
    `torch.utils.data.DataLoader`. Does the following: for a given index
    takes a row of user interactions, a row of user features and then
    returns them as tensors.

    This class is intended for internal usage or advanced users.
    """

    def __init__(
        self,
        users: sparse.csr_matrix,
        interactions: sparse.csr_matrix,
        keep_users: tp.Optional[tp.Sequence[int]] = None,
    ):
        if users.shape[0] != interactions.shape[0]:
            raise ValueError("Number of rows in user features matrix and in interactions matrix must be the same")
        if keep_users is not None:
            self.users = users[keep_users]
            self.interactions = interactions[keep_users]
        else:
            self.users = users
            self.interactions = interactions

    @classmethod
    def from_dataset(
        cls: tp.Type[DSSMUserDatasetT],
        dataset: Dataset,
        keep_users: tp.Optional[tp.Sequence[int]] = None,
    ) -> DSSMUserDatasetT:
        # We take all features here since this dataset is used for recommend only, not for fit
        if dataset.user_features is not None:
            return cls(
                dataset.user_features.get_sparse(),
                dataset.get_user_item_matrix(include_warm_users=True),
                keep_users,
            )
        raise AttributeError("User features attribute of dataset could not be None")

    def __len__(self) -> int:
        return self.users.shape[0]

    def __getitem__(self, idx: int) -> tp.Tuple[torch.FloatTensor, torch.FloatTensor]:
        user_features = self.users[idx].toarray().flatten()
        interactions = self.interactions[idx].toarray().flatten()
        return torch.FloatTensor(user_features), torch.FloatTensor(interactions)
