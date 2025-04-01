#  Copyright 2025 MTS (Mobile Telesystems)
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

import typing as tp
import warnings

import torch
import typing_extensions as tpe
from torch import nn

from rectools.dataset.dataset import Dataset, DatasetSchema, SparseFeaturesSchema
from rectools.dataset.features import SparseFeatures


class ItemNetBase(nn.Module):
    """Base class for item net."""

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError()

    @classmethod
    def from_dataset(cls, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> tp.Optional[tpe.Self]:
        """Construct ItemNet from Dataset."""
        raise NotImplementedError()

    @classmethod
    def from_dataset_schema(
        cls, dataset_schema: DatasetSchema, *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Optional[tpe.Self]:
        """Construct ItemNet from Dataset schema."""
        raise NotImplementedError()

    def get_all_embeddings(self) -> torch.Tensor:
        """Return item embeddings."""
        raise NotImplementedError()

    @property
    def out_dim(self) -> int:
        """Return item embedding output dimension."""
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        """Return ItemNet device."""
        return next(self.parameters()).device


class CatFeaturesItemNet(ItemNetBase):
    """
    Network for item embeddings based only on categorical item features.

    Parameters
    ----------
    emb_bag_inputs : torch.Tensor
        Inputs for `torch.nn.EmbeddingBag.forward` method for full items catalog.
    input_lengths : torch.Tensor
        Lengths of indexes in `emb_bag_inputs` for each item in full catalog.
    offsets : torch.Tensor
        Offsets for `torch.nn.EmbeddingBag.forward` method for full items catalog.
    n_cat_feature_values : torch.Tensor
        Number of stored unique category feature and value pairs.
    n_factors : int
        Latent embedding size of item embeddings.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(
        self,
        emb_bag_inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        offsets: torch.Tensor,
        n_cat_feature_values: int,
        n_factors: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ):
        super().__init__()

        self.n_cat_feature_values = n_cat_feature_values
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=n_cat_feature_values, embedding_dim=n_factors, mode="sum")
        self.dropout = nn.Dropout(dropout_rate)

        self.register_buffer("offsets", offsets)
        self.register_buffer("emb_bag_inputs", emb_bag_inputs)
        self.register_buffer("input_lengths", input_lengths)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get item embeddings from categorical item features.

        Parameters
        ----------
        items : torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        item_emb_bag_inputs, item_offsets = self._get_item_inputs_offsets(items)
        feature_embeddings_per_items = self.embedding_bag(input=item_emb_bag_inputs, offsets=item_offsets)
        feature_embeddings_per_items = self.dropout(feature_embeddings_per_items)
        return feature_embeddings_per_items

    def _get_item_inputs_offsets(self, items: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Get categorical item features and offsets for `items`."""
        length_range = torch.arange(self.get_buffer("input_lengths").max().item(), device=self.device)
        item_indexes = self.get_buffer("offsets")[items].unsqueeze(-1) + length_range
        length_mask = length_range < self.get_buffer("input_lengths")[items].unsqueeze(-1)
        item_emb_bag_inputs = self.get_buffer("emb_bag_inputs")[item_indexes[length_mask]]
        item_offsets = torch.cat(
            (torch.tensor([0], device=self.device), torch.cumsum(self.get_buffer("input_lengths")[items], dim=0)[:-1])
        )
        return item_emb_bag_inputs, item_offsets

    @staticmethod
    def _warn_for_unsupported_dataset_schema(dataset_schema: DatasetSchema) -> None:
        if dataset_schema.items.features is None:
            explanation = """Ignoring `CatFeaturesItemNet` block because dataset doesn't contain item features."""
            warnings.warn(explanation)

        elif dataset_schema.items.features.kind == "dense":
            explanation = """
            Ignoring `CatFeaturesItemNet` block because dataset item features are dense and
            one-hot-encoded categorical features were not created when constructing dataset.
            """
            warnings.warn(explanation)
            return

        elif len(dataset_schema.items.features.cat_feature_indices) == 0:
            explanation = """
            Ignoring `CatFeaturesItemNet` block because dataset item features do not contain categorical features.
            """
            warnings.warn(explanation)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        n_factors: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ) -> tp.Optional[tpe.Self]:
        """
        Create CatFeaturesItemNet from RecTools dataset.

        Parameters
        ----------
        dataset : Dataset
            RecTools dataset.
        n_factors : int
            Latent embedding size of item embeddings.
        dropout_rate : float
            Probability of a hidden unit of item embedding to be zeroed.
        """
        dataset_schema = DatasetSchema.model_validate(dataset.get_schema())
        cls._warn_for_unsupported_dataset_schema(dataset_schema)

        if isinstance(dataset.item_features, SparseFeatures):
            item_cat_features = dataset.item_features.get_cat_features()
            if item_cat_features.values.size == 0:
                return None

            emb_bag_inputs = torch.tensor(item_cat_features.values.indices, dtype=torch.long)
            offsets = torch.tensor(item_cat_features.values.indptr, dtype=torch.long)
            input_lengths = torch.diff(offsets, dim=0)
            n_cat_feature_values = len(item_cat_features.names)

            return cls(
                emb_bag_inputs=emb_bag_inputs,
                offsets=offsets[:-1],
                input_lengths=input_lengths,
                n_cat_feature_values=n_cat_feature_values,
                n_factors=n_factors,
                dropout_rate=dropout_rate,
            )
        return None

    @classmethod
    def from_dataset_schema(
        cls,
        dataset_schema: DatasetSchema,
        n_factors: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ) -> tp.Optional[tpe.Self]:
        """Construct CatFeaturesItemNet from Dataset schema.

        Parameters
        ----------
        dataset_schema : DatasetSchema
            RecTools schema for dataset.
        n_factors : int
            Latent embedding size of item embeddings.
        dropout_rate : float
            Probability of a hidden unit of item embedding to be zeroed.
        """
        cls._warn_for_unsupported_dataset_schema(dataset_schema)
        features_schema = dataset_schema.items.features

        if isinstance(features_schema, SparseFeaturesSchema) and len(features_schema.cat_feature_indices) > 0:
            emb_bag_inputs = torch.randint(high=dataset_schema.items.n_hot, size=(features_schema.cat_n_stored_values,))
            offsets = torch.randint(high=dataset_schema.items.n_hot, size=(dataset_schema.items.n_hot,))
            input_lengths = torch.randint(high=dataset_schema.items.n_hot, size=(dataset_schema.items.n_hot,))
            n_cat_feature_values = len(features_schema.cat_feature_indices)
            return cls(
                emb_bag_inputs=emb_bag_inputs,
                offsets=offsets,
                input_lengths=input_lengths,
                n_cat_feature_values=n_cat_feature_values,
                n_factors=n_factors,
                dropout_rate=dropout_rate,
            )
        return None

    @property
    def out_dim(self) -> int:
        """Return categorical item embedding output dimension."""
        return self.embedding_bag.embedding_dim


class IdEmbeddingsItemNet(ItemNetBase):
    """
    Network for item embeddings based only on item ids.

    Parameters
    ----------
    n_factors : int
        Latent embedding size of item embeddings.
    n_items : int
        Number of items in the dataset.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(
        self,
        n_factors: int,
        n_items: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ):
        super().__init__()

        self.n_items = n_items
        self.ids_emb = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=n_factors,
            padding_idx=0,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get item embeddings from item ids.

        Parameters
        ----------
        items : torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        item_embs = self.ids_emb(items.to(self.device))
        item_embs = self.dropout(item_embs)
        return item_embs

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        n_factors: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ) -> tpe.Self:
        """
        Create IdEmbeddingsItemNet from RecTools dataset.

        Parameters
        ----------
        dataset : Dataset
            RecTools dataset.
        n_factors : int
            Latent embedding size of item embeddings.
        dropout_rate : float
            Probability of a hidden unit of item embedding to be zeroed.
        """
        n_items = dataset.item_id_map.size
        return cls(n_factors, n_items, dropout_rate)

    @classmethod
    def from_dataset_schema(
        cls,
        dataset_schema: DatasetSchema,
        n_factors: int,
        dropout_rate: float,
        **kwargs: tp.Any,
    ) -> tpe.Self:
        """Construct ItemNet from Dataset schema.

        Parameters
        ----------
        dataset_schema : DatasetSchema
            RecTools schema for dataset.
        n_factors : int
            Latent embedding size of item embeddings.
        dropout_rate : float
            Probability of a hidden unit of item embedding to be zeroed.
        """
        n_items = dataset_schema.items.n_hot
        return cls(n_factors, n_items, dropout_rate)

    @property
    def out_dim(self) -> int:
        """Return item embedding output dimension."""
        return self.ids_emb.embedding_dim


class ItemNetConstructorBase(ItemNetBase):
    """
    Constructed network for item embeddings based on aggregation of embeddings from transferred item network types.

    Parameters
    ----------
    n_items : int
        Number of items in the dataset.
    item_net_blocks : Sequence(ItemNetBase)
        Latent embedding size of item embeddings.
    """

    def __init__(
        self,
        n_items: int,
        item_net_blocks: tp.Sequence[ItemNetBase],
        **kwargs: tp.Any,
    ) -> None:
        super().__init__()

        if len(item_net_blocks) == 0:
            raise ValueError("At least one type of net to calculate item embeddings should be provided.")

        self.n_items = n_items
        self.n_item_blocks = len(item_net_blocks)
        self.item_net_blocks = nn.ModuleList(item_net_blocks)

    @property
    def catalog(self) -> torch.Tensor:
        """Return tensor with elements in range [0, n_items)."""
        return torch.arange(0, self.n_items)

    def get_all_embeddings(self) -> torch.Tensor:
        """Return item embeddings."""
        return self.forward(self.catalog)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        n_factors: int,
        dropout_rate: float,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        **kwargs: tp.Any,
    ) -> tpe.Self:
        """
        Construct ItemNet from RecTools dataset and from various blocks of item networks.

        Parameters
        ----------
        dataset : Dataset
            RecTools dataset.
        n_factors : int
            Latent embedding size of item embeddings.
        dropout_rate : float
            Probability of a hidden unit of item embedding to be zeroed.
        item_net_block_types : sequence of `type(ItemNetBase)`
            Sequence item network block types.
        """
        n_items = dataset.item_id_map.size

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset(dataset, n_factors, dropout_rate, **kwargs)
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(n_items, item_net_blocks)

    @classmethod
    def from_dataset_schema(
        cls,
        dataset_schema: DatasetSchema,
        n_factors: int,
        dropout_rate: float,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        **kwargs: tp.Any,
    ) -> tpe.Self:
        """Construct ItemNet from Dataset schema.

        Parameters
        ----------
        dataset_schema : DatasetSchema
            RecTools schema for dataset.
        n_factors : int
            Latent embedding size of item embeddings.
        dropout_rate : float
            Probability of a hidden unit of item embedding to be zeroed.
        item_net_block_types : sequence of `type(ItemNetBase)`
            Sequence item network block types.
        """
        n_items = dataset_schema.items.n_hot

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset_schema(dataset_schema, n_factors, dropout_rate, **kwargs)
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(n_items, item_net_blocks)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """Forward pass through item net blocks and aggregation of the results.

        Parameters
        ----------
        items : torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        raise NotImplementedError()


class SumOfEmbeddingsConstructor(ItemNetConstructorBase):
    """
    Item net blocks constructor that simply sums all of the its net blocks embeddings.

    Parameters
    ----------
    n_items : int
        Number of items in the dataset.
    item_net_blocks : Sequence(ItemNetBase)
        Latent embedding size of item embeddings.
    """

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through item net blocks and aggregation of the results.
        Simple sum of embeddings.

        Parameters
        ----------
        items : torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        item_embs = []
        for idx_block in range(self.n_item_blocks):
            item_emb = self.item_net_blocks[idx_block](items)
            item_embs.append(item_emb)
        return torch.sum(torch.stack(item_embs, dim=0), dim=0)

    @property
    def out_dim(self) -> int:
        """Return item net constructor output dimension."""
        return self.item_net_blocks[0].out_dim
