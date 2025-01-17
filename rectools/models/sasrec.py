import typing as tp
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import typing_extensions as tpe
from implicit.gpu import HAS_CUDA
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset, Interactions
from rectools.dataset.features import SparseFeatures
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ErrorBehaviour, InternalRecoTriplet, ModelBase
from rectools.models.rank import Distance, ImplicitRanker
from rectools.types import InternalIdsArray

PADDING_VALUE = "PAD"

# pylint: disable=too-many-lines
# ####  --------------  Net blocks  --------------  #### #


class ItemNetBase(nn.Module):
    """TODO: use Protocol"""

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError()

    @classmethod
    def from_dataset(cls, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> tp.Optional[tpe.Self]:
        """Construct ItemNet."""
        raise NotImplementedError()

    def get_all_embeddings(self) -> torch.Tensor:
        """Return item embeddings."""
        raise NotImplementedError()


class TransformerLayersBase(nn.Module):
    """TODO: use Protocol"""

    def forward(
        self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError()


class PositionalEncodingBase(torch.nn.Module):
    """TODO: use Protocol"""

    def forward(self, sessions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError()


class CatFeaturesItemNet(ItemNetBase):
    """
    Network for item embeddings based only on categorical item features.

    Parameters
    ----------
    item_features: SparseFeatures
        Storage for sparse features.
    n_factors: int
        Latent embedding size of item embeddings.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(
        self,
        item_features: SparseFeatures,
        n_factors: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.item_features = item_features
        self.n_items = len(item_features)
        self.n_cat_features = len(item_features.names)

        self.category_embeddings = nn.Embedding(num_embeddings=self.n_cat_features, embedding_dim=n_factors)
        self.drop_layer = nn.Dropout(dropout_rate)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get item embeddings from categorical item features.

        Parameters
        ----------
        items: torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        device = self.category_embeddings.weight.device
        # TODO: Should we use torch.nn.EmbeddingBag?
        feature_dense = self.get_dense_item_features(items)

        feature_embs = self.category_embeddings(self.feature_catalog.to(device))
        feature_embs = self.drop_layer(feature_embs)

        feature_embeddings_per_items = feature_dense.to(device) @ feature_embs
        return feature_embeddings_per_items

    @property
    def feature_catalog(self) -> torch.Tensor:
        """Return tensor with elements in range [0, n_cat_features)."""
        return torch.arange(0, self.n_cat_features)

    def get_dense_item_features(self, items: torch.Tensor) -> torch.Tensor:
        """
        Get categorical item values by certain item ids in dense format.

        Parameters
        ----------
        items: torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            categorical item values in dense format.
        """
        # TODO: Add the whole `feature_dense` to the right gpu device at once?
        feature_dense = self.item_features.take(items.detach().cpu().numpy()).get_dense()
        return torch.from_numpy(feature_dense)

    @classmethod
    def from_dataset(cls, dataset: Dataset, n_factors: int, dropout_rate: float) -> tp.Optional[tpe.Self]:
        """
        Create CatFeaturesItemNet from RecTools dataset.

        Parameters
        ----------
        dataset: Dataset
            RecTools dataset.
        n_factors: int
            Latent embedding size of item embeddings.
        dropout_rate: float
            Probability of a hidden unit of item embedding to be zeroed.
        """
        item_features = dataset.item_features

        if item_features is None:
            explanation = """Ignoring `CatFeaturesItemNet` block because dataset doesn't contain item features."""
            warnings.warn(explanation)
            return None

        if not isinstance(item_features, SparseFeatures):
            explanation = """
            Ignoring `CatFeaturesItemNet` block because
            dataset item features are dense and unable to contain categorical features.
            """
            warnings.warn(explanation)
            return None

        item_cat_features = item_features.get_cat_features()

        if item_cat_features.values.size == 0:
            explanation = """
            Ignoring `CatFeaturesItemNet` block because dataset item features do not contain categorical features.
            """
            warnings.warn(explanation)
            return None

        return cls(item_cat_features, n_factors, dropout_rate)


class IdEmbeddingsItemNet(ItemNetBase):
    """
    Network for item embeddings based only on item ids.

    Parameters
    ----------
    n_factors: int
        Latent embedding size of item embeddings.
    n_items: int
        Number of items in the dataset.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(self, n_factors: int, n_items: int, dropout_rate: float):
        super().__init__()

        self.n_items = n_items
        self.ids_emb = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=n_factors,
            padding_idx=0,
        )
        self.drop_layer = nn.Dropout(dropout_rate)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get item embeddings from item ids.

        Parameters
        ----------
        items: torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        item_embs = self.ids_emb(items.to(self.ids_emb.weight.device))
        item_embs = self.drop_layer(item_embs)
        return item_embs

    @classmethod
    def from_dataset(cls, dataset: Dataset, n_factors: int, dropout_rate: float) -> tpe.Self:
        """TODO"""
        n_items = dataset.item_id_map.size
        return cls(n_factors, n_items, dropout_rate)


class ItemNetConstructor(ItemNetBase):
    """
    Constructed network for item embeddings based on aggregation of embeddings from transferred item network types.

    Parameters
    ----------
    n_items: int
        Number of items in the dataset.
    item_net_blocks: Sequence(ItemNetBase)
        Latent embedding size of item embeddings.
    """

    def __init__(
        self,
        n_items: int,
        item_net_blocks: tp.Sequence[ItemNetBase],
    ) -> None:
        """TODO"""
        super().__init__()

        if len(item_net_blocks) == 0:
            raise ValueError("At least one type of net to calculate item embeddings should be provided.")

        self.n_items = n_items
        self.n_item_blocks = len(item_net_blocks)
        self.item_net_blocks = nn.ModuleList(item_net_blocks)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get item embeddings from item network blocks.

        Parameters
        ----------
        items: torch.Tensor
            Internal item ids.

        Returns
        -------
        torch.Tensor
            Item embeddings.
        """
        item_embs = []
        # TODO: Add functionality for parallel computing.
        for idx_block in range(self.n_item_blocks):
            item_emb = self.item_net_blocks[idx_block](items)
            item_embs.append(item_emb)
        return torch.sum(torch.stack(item_embs, dim=0), dim=0)

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
    ) -> tpe.Self:
        """
        Construct ItemNet from RecTools dataset and from various blocks of item networks.

        Parameters
        ----------
        dataset: Dataset
            RecTools dataset.
        n_factors: int
            Latent embedding size of item embeddings.
        dropout_rate: float
            Probability of a hidden unit of item embedding to be zeroed.
        item_net_block_types: Sequence(Type(ItemNetBase))
            Sequence item network block types.
        """
        n_items = dataset.item_id_map.size

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset(dataset, n_factors, dropout_rate)
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(n_items, item_net_blocks)


class PointWiseFeedForward(nn.Module):
    """
    Feed-Forward network to introduce nonlinearity into the transformer model.
    This implementation is the one used by SASRec authors.

    Parameters
    ----------
    n_factors: int
        Latent embeddings size.
    n_factors_ff: int
        How many hidden units to use in the network.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(self, n_factors: int, n_factors_ff: int, dropout_rate: float, activation: torch.nn.Module) -> None:
        super().__init__()
        self.ff_linear1 = nn.Linear(n_factors, n_factors_ff)
        self.ff_dropout1 = torch.nn.Dropout(dropout_rate)
        self.ff_activation = activation
        self.ff_linear2 = nn.Linear(n_factors_ff, n_factors)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        seqs: torch.Tensor
            User sequences of item embeddings.

        Returns
        -------
        torch.Tensor
            User sequence that passed through all layers.
        """
        output = self.ff_activation(self.ff_linear1(seqs))
        fin = self.ff_linear2(self.ff_dropout1(output))
        return fin


class SASRecTransformerLayers(TransformerLayersBase):
    """
    Exactly SASRec author's transformer blocks architecture but with pytorch Multi-Head Attention realisation.

    Parameters
    ----------
    n_blocks: int
        Number of transformer blocks.
    n_factors: int
        Latent embeddings size.
    n_heads: int
        Number of attention heads.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.multi_head_attn = nn.ModuleList(
            [torch.nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True) for _ in range(n_blocks)]
        )  # TODO: original architecture had another version of MHA
        self.q_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.ff_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.feed_forward = nn.ModuleList(
            [PointWiseFeedForward(n_factors, n_factors, dropout_rate, torch.nn.ReLU()) for _ in range(n_blocks)]
        )
        self.dropout = nn.ModuleList([torch.nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        self.last_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)

    def forward(
        self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through transformer blocks.

        Parameters
        ----------
        seqs: torch.Tensor
            User sequences of item embeddings.
        timeline_mask: torch.Tensor
            Mask to zero out padding elements.
        attn_mask: torch.Tensor
            Mask to forbid model to use future interactions.

        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        # TODO: do we need to fill padding embeds in sessions to all zeros
        # or should we use the learnt padding embedding? Should we make it an option for user to decide?
        seqs *= timeline_mask  # [batch_size, session_max_len, n_factors]
        for i in range(self.n_blocks):
            q = self.q_layer_norm[i](seqs)
            mha_output, _ = self.multi_head_attn[i](
                q, seqs, seqs, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
            )
            seqs = q + mha_output
            ff_input = self.ff_layer_norm[i](seqs)
            seqs = self.feed_forward[i](ff_input)
            seqs = self.dropout[i](seqs)
            seqs += ff_input
            seqs *= timeline_mask

        seqs = self.last_layernorm(seqs)

        return seqs


class LearnableInversePositionalEncoding(PositionalEncodingBase):
    """
    Class to introduce learnable positional embeddings.

    Parameters
    ----------
    use_pos_emb: bool
        If ``True``, adds learnable positional encoding to session item embeddings.
    session_max_len: int
        Maximum length of user sequence.
    n_factors: int
       Latent embeddings size.
    """

    def __init__(self, use_pos_emb: bool, session_max_len: int, n_factors: int):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(session_max_len, n_factors) if use_pos_emb else None

    def forward(self, sessions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add learnable positional encoding to sessions and mask padding elements.

        Parameters
        ----------
        sessions: torch.Tensor
            User sessions in the form of sequences of items ids.
        timeline_mask: torch.Tensor
            Mask to zero out padding elements.

        Returns
        -------
        torch.Tensor
            Encoded user sessions with added positional encoding if `use_pos_emb` is ``True``.
        """
        batch_size, session_max_len, _ = sessions.shape

        if self.pos_emb is not None:
            # Inverse positions are appropriate for variable length sequences across different batches
            # They are equal to absolute positions for fixed sequence length across different batches
            positions = torch.tile(
                torch.arange(session_max_len - 1, -1, -1), (batch_size, 1)
            )  # [batch_size, session_max_len]
            sessions += self.pos_emb(positions.to(sessions.device))

        return sessions


# ####  --------------  Session Encoder  --------------  #### #


class TransformerBasedSessionEncoder(torch.nn.Module):
    """
    Torch model for recommendations.

    Parameters
    ----------
    n_blocks: int
        Number of transformer blocks.
    n_factors: int
        Latent embeddings size.
    n_heads: int
        Number of attention heads.
    session_max_len: int
        Maximum length of user sequence.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    use_pos_emb: bool, default True
        If ``True``, adds learnable positional encoding to session item embeddings.
    use_causal_attn: bool, default True
        If ``True``, causal mask is used in multi-head self-attention.
    transformer_layers_type: Type(TransformerLayersBase), default `SasRecTransformerLayers`
        Type of transformer layers architecture.
    item_net_type: Type(ItemNetBase), default IdEmbeddingsItemNet
        Type of network returning item embeddings.
    pos_encoding_type: Type(PositionalEncodingBase), default LearnableInversePositionalEncoding
        Type of positional encoding.
    """

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        session_max_len: int,
        dropout_rate: float,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        transformer_layers_type: tp.Type[TransformerLayersBase] = SASRecTransformerLayers,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
    ) -> None:
        super().__init__()

        self.item_model: ItemNetConstructor
        self.pos_encoding = pos_encoding_type(use_pos_emb, session_max_len, n_factors)
        self.emb_dropout = torch.nn.Dropout(dropout_rate)
        self.transformer_layers = transformer_layers_type(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
        )
        self.use_causal_attn = use_causal_attn
        self.use_key_padding_mask = use_key_padding_mask
        self.n_factors = n_factors
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads

        self.item_net_block_types = item_net_block_types

    def construct_item_net(self, dataset: Dataset) -> None:
        """
        Construct network for item embeddings from dataset.

        Parameters
        ----------
        dataset: Dataset
            RecTools dataset with user-item interactions.
        """
        self.item_model = ItemNetConstructor.from_dataset(
            dataset, self.n_factors, self.dropout_rate, self.item_net_block_types
        )

    def encode_sessions(self, sessions: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings.
        Add positional encoding.
        Pass history through transformer blocks.

        Parameters
        ----------
        sessions:  torch.Tensor
            User sessions in the form of sequences of items ids.
        item_embs: torch.Tensor
            Item embeddings.

        Returns
        -------
        torch.Tensor. [batch_size, session_max_len, n_factors]
            Encoded session embeddings.
        """
        session_max_len = sessions.shape[1]
        attn_mask = None
        key_padding_mask = None
        # TODO: att_mask and key_padding_mask together result into NaN scores
        if self.use_causal_attn:
            attn_mask = ~torch.tril(
                torch.ones((session_max_len, session_max_len), dtype=torch.bool, device=sessions.device)
            )
        if self.use_key_padding_mask:
            key_padding_mask = sessions == 0
        timeline_mask = (sessions != 0).unsqueeze(-1)  # [batch_size, session_max_len, 1]
        seqs = item_embs[sessions]  # [batch_size, session_max_len, n_factors]
        seqs = self.pos_encoding(seqs)
        seqs = self.emb_dropout(seqs)
        # TODO: stop passing timeline_mask together with key_padding_mask because they have same information
        seqs = self.transformer_layers(seqs, timeline_mask, attn_mask, key_padding_mask)
        return seqs

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_max_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get item and session embeddings.
        Get item embeddings.
        Pass user sessions through transformer blocks.

        Parameters
        ----------
        sessions: torch.Tensor
            User sessions in the form of sequences of items ids.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
        """
        item_embs = self.item_model.get_all_embeddings()  # [n_items + n_item_extra_tokens, n_factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_max_len, n_factors]
        return item_embs, session_embs


# ####  --------------  Data Processor  --------------  #### #


class SequenceDataset(TorchDataset):
    """
    Dataset for sequential data.

    Parameters
    ----------
    sessions: List[List[int]]
        User sessions in the form of sequences of items ids.
    weights: List[List[float]]
        Weight of each interaction from the session.
    """

    def __init__(self, sessions: List[List[int]], weights: List[List[float]]):
        self.sessions = sessions
        self.weights = weights

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int) -> Tuple[List[int], List[float]]:
        session = self.sessions[index]  # [session_len]
        weights = self.weights[index]  # [session_len]
        return session, weights

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
    ) -> "SequenceDataset":
        """
        Group interactions by user.
        Construct SequenceDataset from grouped interactions.

        Parameters
        ----------
        interactions: pd.DataFrame
            User-item interactions.
        """
        sessions = (
            interactions.sort_values(Columns.Datetime)
            .groupby(Columns.User, sort=True)[[Columns.Item, Columns.Weight]]
            .agg(list)
        )
        sessions, weights = (
            sessions[Columns.Item].to_list(),
            sessions[Columns.Weight].to_list(),
        )

        return cls(sessions=sessions, weights=weights)


class SessionEncoderDataPreparatorBase:
    """
    Base class for data preparator. To change train/recommend dataset processing, train/recommend dataloaders inherit
    from this class and pass your custom data preparator to your model parameters.

    Parameters
    ----------
    session_max_len: int
        Maximum length of user sequence.
    batch_size: int
        How many samples per batch to load.
    dataloader_num_workers: int
        Number of loader worker processes.
    item_extra_tokens: Sequence(Hashable), default (PADDING_VALUE,) = ("PAD",)
        Which element to use for sequence padding.
    shuffle_train: bool, default True
        If ``True``, reshuffles data at each epoch.
    train_min_user_interactions: int, default 2
        Minimum length of user sequence. Cannot be less than 2.
    """

    def __init__(
        self,
        session_max_len: int,
        batch_size: int,
        dataloader_num_workers: int,
        shuffle_train: bool = True,
        item_extra_tokens: tp.Sequence[tp.Hashable] = (PADDING_VALUE,),
        train_min_user_interactions: int = 2,
        n_negatives: tp.Optional[int] = None,
        get_val_mask_func: tp.Optional[tp.Callable] = None,
    ) -> None:
        """TODO"""
        self.item_id_map: IdMap
        self.extra_token_ids: tp.Dict
        self.train_dataset: Dataset
        self.val_interactions: tp.Optional[pd.DataFrame] = None
        self.session_max_len = session_max_len
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions
        self.item_extra_tokens = item_extra_tokens
        self.shuffle_train = shuffle_train
        self.get_val_mask_func = get_val_mask_func

    def get_known_items_sorted_internal_ids(self) -> np.ndarray:
        """Return internal item ids from processed dataset in sorted order."""
        return self.item_id_map.get_sorted_internal()[self.n_item_extra_tokens :]

    def get_known_item_ids(self) -> np.ndarray:
        """Return external item ids from processed dataset in sorted order."""
        return self.item_id_map.get_external_sorted_by_internal()[self.n_item_extra_tokens :]

    @property
    def n_item_extra_tokens(self) -> int:
        """Return number of padding elements"""
        return len(self.item_extra_tokens)

    def process_dataset_train(
        self,
        dataset: Dataset,
        split_interactions_train_val: tp.Optional[tp.Callable] = None,
    ) -> None:
        """TODO"""
        interactions = dataset.get_raw_interactions()

        # Exclude val interaction targets from train if needed
        interactions = raw_interactions
        if self.get_val_mask_func is not None:
            val_mask = self.get_val_mask_func(raw_interactions)
            interactions = raw_interactions[~val_mask]

        # Filter train interactions
        user_stats = interactions[Columns.User].value_counts()
        users = user_stats[user_stats >= self.train_min_user_interactions].index
        interactions = interactions[(interactions[Columns.User].isin(users))]
        interactions = interactions.sort_values(Columns.Datetime).groupby(Columns.User).tail(self.session_max_len + 1)

        # Construct dataset
        # TODO: user features are dropped for now
        user_id_map = IdMap.from_values(interactions[Columns.User].values)
        item_id_map = IdMap.from_values(self.item_extra_tokens)
        item_id_map = item_id_map.add_ids(interactions[Columns.Item])

        # get item features
        item_features = None
        if dataset.item_features is not None:
            item_features = dataset.item_features
            # TODO: remove assumption on SparseFeatures and add Dense Features support
            if not isinstance(item_features, SparseFeatures):
                raise ValueError("`item_features` in `dataset` must be `SparseFeatures` instance.")

            internal_ids = dataset.item_id_map.convert_to_internal(
                item_id_map.get_external_sorted_by_internal()[self.n_item_extra_tokens :]
            )
            sorted_item_features = item_features.take(internal_ids)

            dtype = sorted_item_features.values.dtype
            n_features = sorted_item_features.values.shape[1]
            extra_token_feature_values = sparse.csr_matrix((self.n_item_extra_tokens, n_features), dtype=dtype)

            full_feature_values: sparse.scr_matrix = sparse.vstack(
                [extra_token_feature_values, sorted_item_features.values], format="csr"
            )

            item_features = SparseFeatures.from_iterables(values=full_feature_values, names=item_features.names)

        interactions = Interactions.from_raw(interactions, user_id_map, item_id_map, keep_extra_cols=True)

        self.train_dataset = Dataset(user_id_map, item_id_map, dataset_interactions, item_features=item_features)

        self.item_id_map = self.train_dataset.item_id_map
        extra_token_ids = self.item_id_map.convert_to_internal(self.item_extra_tokens)
        self.extra_token_ids = dict(zip(self.item_extra_tokens, extra_token_ids))

        # Define val interactions
        if self.get_val_mask_func is not None:
            val_targets = raw_interactions[val_mask]
            val_targets = val_targets[
                (val_targets[Columns.User].isin(user_id_map.external_ids))
                & (val_targets[Columns.Item].isin(item_id_map.external_ids))
            ]
            val_interactions = interactions[interactions[Columns.User].isin(val_targets[Columns.User].unique())].copy()
            val_interactions[Columns.Weight] = 0
            val_interactions = pd.concat([val_interactions, val_targets], axis=0)
            self.val_interactions = Interactions.from_raw(val_interactions, user_id_map, item_id_map).df

    def get_dataloader_train(self) -> DataLoader:
        """
        Construct train dataloader from processed dataset.

        Parameters
        ----------
        processed_dataset: Dataset
            RecTools dataset prepared for training.

        Returns
        -------
        DataLoader
            Train dataloader.
        """
        sequence_dataset = SequenceDataset.from_interactions(self.train_dataset.interactions.df)
        train_dataloader = DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_train,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=self.shuffle_train,
        )
        return train_dataloader

    def get_dataloader_val(self) -> tp.Optional[DataLoader]:
        """
        Construct validation dataloader from processed dataset.

        Parameters
        ----------
        processed_dataset: Optional(Dataset), default None
            RecTools dataset prepared for training.

        Returns
        -------
        Optional(Dataset)
            Validation dataloader.
        """
        if self.val_interactions is None:
            return None

        sequence_dataset = SequenceDataset.from_interactions(self.val_interactions)
        val_dataloader = DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_val,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )
        return val_dataloader

    def get_dataloader_recommend(self, dataset: Dataset) -> DataLoader:
        """TODO"""
        sequence_dataset = SequenceDataset.from_interactions(dataset.interactions.df)
        recommend_dataloader = DataLoader(
            sequence_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn_recommend,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )
        return recommend_dataloader

    def transform_dataset_u2i(self, dataset: Dataset, users: ExternalIds) -> Dataset:
        """
        Process dataset for u2i recommendations.
        Filter out interactions and adapt id maps.
        All users beyond target users for recommendations are dropped.
        All target users that do not have at least one known item in interactions are dropped.

        Parameters
        ----------
        dataset: Dataset
            RecTools dataset.
        users: ExternalIds
            Array of external user ids to recommend for.

        Returns
        -------
        Dataset
            Processed RecTools dataset.
            Final dataset will consist only of model known items during fit and only of required
            (and supported) target users for recommendations.
            Final user_id_map is an enumerated list of supported (filtered) target users.
            Final item_id_map is model item_id_map constructed during training.
        """
        # Filter interactions in dataset internal ids
        interactions = dataset.interactions.df
        users_internal = dataset.user_id_map.convert_to_internal(users, strict=False)
        items_internal = dataset.item_id_map.convert_to_internal(self.get_known_item_ids(), strict=False)
        interactions = interactions[interactions[Columns.User].isin(users_internal)]  # todo: fast_isin
        interactions = interactions[interactions[Columns.Item].isin(items_internal)]

        # Convert to external ids
        interactions[Columns.Item] = dataset.item_id_map.convert_to_external(interactions[Columns.Item])
        interactions[Columns.User] = dataset.user_id_map.convert_to_external(interactions[Columns.User])

        # Prepare new user id mapping
        rec_user_id_map = IdMap.from_values(interactions[Columns.User])

        # Construct dataset
        # TODO: For now features are dropped because model doesn't support them
        n_filtered = len(users) - rec_user_id_map.size
        if n_filtered > 0:
            explanation = f"""{n_filtered} target users were considered cold because of missing known items"""
            warnings.warn(explanation)
        filtered_interactions = Interactions.from_raw(interactions, rec_user_id_map, self.item_id_map)
        filtered_dataset = Dataset(rec_user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def transform_dataset_i2i(self, dataset: Dataset) -> Dataset:
        """
        Process dataset for i2i recommendations.
        Filter out interactions and adapt id maps.

        Parameters
        ----------
        dataset: Dataset
            RecTools dataset.

        Returns
        -------
        Dataset
            Processed RecTools dataset.
            Final dataset will consist only of model known items during fit.
            Final user_id_map is the same as dataset original.
            Final item_id_map is model item_id_map constructed during training.
        """
        # TODO: optimize by filtering in internal ids
        interactions = dataset.get_raw_interactions()
        interactions = interactions[interactions[Columns.Item].isin(self.get_known_item_ids())]
        filtered_interactions = Interactions.from_raw(interactions, dataset.user_id_map, self.item_id_map)
        filtered_dataset = Dataset(dataset.user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """TODO"""
        raise NotImplementedError()

    def _collate_fn_val(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """TODO"""
        raise NotImplementedError()

    def _collate_fn_recommend(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """TODO"""
        raise NotImplementedError()


class SASRecDataPreparator(SessionEncoderDataPreparatorBase):
    """TODO"""

    def __init__(
        self,
        session_max_len: int,
        batch_size: int,
        dataloader_num_workers: int,
        shuffle_train: bool = True,
        item_extra_tokens: tp.Sequence[tp.Hashable] = (PADDING_VALUE,),
        train_min_user_interactions: int = 2,
        n_negatives: tp.Optional[int] = None,
        get_val_mask_func: tp.Optional[tp.Callable] = None,
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            shuffle_train=shuffle_train,
            item_extra_tokens=item_extra_tokens,
            train_min_user_interactions=train_min_user_interactions,
            n_negatives=n_negatives,
            get_val_mask_func=get_val_mask_func,
        )

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Truncate each session from right to keep (session_max_len+1) last items.
        Do left padding until  (session_max_len+1) is reached.
        Split to `x`, `y`, and `yw`.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        for i, (ses, ses_weights) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_max_len]
            y[i, -len(ses) + 1 :] = ses[1:]  # ses: [session_len] -> y[i]: [session_max_len]
            yw[i, -len(ses) + 1 :] = ses_weights[1:]  # ses_weights: [session_len] -> yw[i]: [session_max_len]

        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        # TODO: we are sampling negatives for paddings
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, self.session_max_len, self.n_negatives),
            )  # [batch_size, session_max_len, n_negatives]
            batch_dict["negatives"] = negatives
        return batch_dict

    def _collate_fn_val(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_target_size = max(sum(1 for weight in ses_weights if weight != 0) for _, ses_weights in batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, max_target_size))
        yw = np.zeros((batch_size, max_target_size))
        for i, (ses, ses_weights) in enumerate(batch):
            input_session = [ses[idx] for idx, weight in enumerate(ses_weights) if weight == 0]
            target_idx = [idx for idx, weight in enumerate(ses_weights) if weight != 0]

            targets = list(map(ses.__getitem__, target_idx))
            targets_weights = list(map(ses_weights.__getitem__, target_idx))

            # ses: [session_len] -> x[i]: [session_max_len]
            x[i, -len(input_session) :] = input_session[-self.session_max_len :]
            y[i, -len(targets) :] = targets  # y[i]: [val_k_out]
            yw[i, -len(targets_weights) :] = targets_weights  # yw[i]: [val_k_out]

        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        # TODO: we are sampling negatives for paddings
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, max_target_size, self.n_negatives),
            )  # [batch_size, session_max_len, n_negatives]
            batch_dict["negatives"] = negatives
        return batch_dict

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len))
        for i, (ses, _) in enumerate(batch):
            x[i, -len(ses) :] = ses[-self.session_max_len :]
        return {"x": torch.LongTensor(x)}


# ####  --------------  Lightning Model  --------------  #### #


class SessionEncoderLightningModuleBase(LightningModule):
    """
    Base class for lightning module. To change train procedure inherit
    from this class and pass your custom LightningModule to your model parameters.

    Parameters
    ----------
    torch_model: TransformerBasedSessionEncoder
        Torch model to make recommendations.
    lr: float
        Learning rate.
    loss: str, default "softmax"
        Loss function.
    adam_betas: Tuple[float, float], default (0.9, 0.98)
        Coefficients for running averages of gradient and its square.
    """

    def __init__(
        self,
        torch_model: TransformerBasedSessionEncoder,
        lr: float,
        gbce_t: float,
        data_preparator: SessionEncoderDataPreparatorBase,
        loss: str = "softmax",
        adam_betas: Tuple[float, float] = (0.9, 0.98),
        verbose: int = 0,
        top_k_saved_val_reco: tp.Optional[int] = None,
    ):
        super().__init__()
        self.lr = lr
        self.loss = loss
        self.torch_model = torch_model
        self.adam_betas = adam_betas
        self.gbce_t = gbce_t
        self.data_preparator = data_preparator
        self.verbose = verbose
        self.top_k_saved_val_reco = top_k_saved_val_reco
        self.item_embs: torch.Tensor

    def configure_optimizers(self) -> torch.optim.Adam:
        """Choose what optimizers and learning-rate schedulers to use in optimization"""
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        raise NotImplementedError()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validate step."""
        raise NotImplementedError()

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        raise NotImplementedError()


class SessionEncoderLightningModule(SessionEncoderLightningModuleBase):
    """Lightning module to train SASRec model."""

    def __init__(
        self,
        torch_model: TransformerBasedSessionEncoder,
        lr: float,
        gbce_t: float,
        data_preparator: SessionEncoderDataPreparatorBase,
        loss: str = "softmax",
        adam_betas: Tuple[float, float] = (0.9, 0.98),
        verbose: int = 1,
        top_k_saved_val_reco: tp.Optional[int] = None,
    ):
        super().__init__(
            torch_model=torch_model,
            lr=lr,
            gbce_t=gbce_t,
            data_preparator=data_preparator,
            loss=loss,
            adam_betas=adam_betas,
            verbose=verbose,
            top_k_saved_val_reco=top_k_saved_val_reco,
        )

        if self.top_k_saved_val_reco is not None:
            self.epoch_val_recos: tp.List[tp.List[int]] = []
            self.epoch_targets: tp.List[tp.List[int]] = []

    def on_train_start(self) -> None:
        """Initialize parameters with values from Xavier normal distribution."""
        # TODO: init padding embedding with zeros
        self._xavier_normal_init()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """TODO"""
        x, y, w = batch["x"], batch["y"], batch["yw"]
        if self.loss == "softmax":
            logits = self._get_full_catalog_logits(x)
            train_loss = self._calc_softmax_loss(logits, y, w)
        elif self.loss == "BCE":
            negatives = batch["negatives"]
            logits = self._get_pos_neg_logits(x, y, negatives)
            train_loss = self._calc_bce_loss(logits, y, w)
        elif self.loss == "gBCE":
            negatives = batch["negatives"]
            logits = self._get_pos_neg_logits(x, y, negatives)
            train_loss = self._calc_gbce_loss(logits, y, w, negatives)
        else:
            raise ValueError(f"loss {self.loss} is not supported")
        self.log("train/loss", train_loss, on_step=False, on_epoch=True, prog_bar=self.verbose > 0)
        return train_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validate step."""
        x, y, w = batch["x"], batch["y"], batch["yw"]
        logits = None
        if self.loss == "softmax":
            logits = self._get_full_catalog_logits(x)[:, -1:, :]
            val_loss = self._calc_softmax_loss(logits, y, w)
            logits = logits.squeeze()
        elif self.loss == "BCE":
            negatives = batch["negatives"]
            pos_neg_logits = self._get_pos_neg_logits(x, y, negatives)[:, -1:, :]
            val_loss = self._calc_bce_loss(pos_neg_logits, y, w)
        elif self.loss == "gBCE":
            negatives = batch["negatives"]
            pos_neg_logits = self._get_pos_neg_logits(x, y, negatives)[:, -1:, :]
            val_loss = self._calc_gbce_loss(pos_neg_logits, y, w, negatives)
        else:
            raise ValueError(f"loss {self.loss} is not supported")

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=self.verbose > 0)

        if self.top_k_saved_val_reco is not None:
            if logits is None:
                logits = self._get_full_catalog_logits(x)[:, -1, :]
            _, sorted_batch_recos = logits.topk(k=self.top_k_saved_val_reco)
            self.epoch_val_recos.extend(sorted_batch_recos.tolist())
            self.epoch_targets.extend(y.tolist())

        return val_loss

    def _get_full_catalog_logits(self, x: torch.Tensor) -> torch.Tensor:
        item_embs, session_embs = self.torch_model(x)
        logits = session_embs @ item_embs.T
        return logits

    def _get_pos_neg_logits(self, x: torch.Tensor, y: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        # [n_items + n_item_extra_tokens, n_factors], [batch_size, session_max_len, n_factors]
        item_embs, session_embs = self.torch_model(x)
        pos_neg = torch.cat([y.unsqueeze(-1), negatives], dim=-1)  # [batch_size, session_max_len, n_negatives + 1]
        pos_neg_embs = item_embs[pos_neg]  # [batch_size, session_max_len, n_negatives + 1, n_factors]
        # [batch_size, session_max_len, n_negatives + 1]
        logits = (pos_neg_embs @ session_embs.unsqueeze(-1)).squeeze(-1)
        return logits

    def _get_reduced_overconfidence_logits(self, logits: torch.Tensor, n_items: int, n_negatives: int) -> torch.Tensor:
        # https://arxiv.org/pdf/2308.07192.pdf
        alpha = n_negatives / (n_items - 1)  # sampling rate
        beta = alpha * (self.gbce_t * (1 - 1 / alpha) + 1 / alpha)

        pos_logits = logits[:, :, 0:1].to(torch.float64)
        neg_logits = logits[:, :, 1:].to(torch.float64)

        epsilon = 1e-10
        pos_probs = torch.clamp(torch.sigmoid(pos_logits), epsilon, 1 - epsilon)
        pos_probs_adjusted = torch.clamp(pos_probs.pow(-beta), 1 + epsilon, torch.finfo(torch.float64).max)
        pos_probs_adjusted = torch.clamp(
            torch.div(1, (pos_probs_adjusted - 1)), epsilon, torch.finfo(torch.float64).max
        )
        pos_logits_transformed = torch.log(pos_probs_adjusted)
        logits = torch.cat([pos_logits_transformed, neg_logits], dim=-1)
        return logits

    @classmethod
    def _calc_softmax_loss(cls, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # We are using CrossEntropyLoss with a multi-dimensional case

        # Logits must be passed in form of [batch_size, n_items + n_item_extra_tokens, session_max_len],
        #  where n_items + n_item_extra_tokens is number of classes

        # Target label indexes must be passed in a form of [batch_size, session_max_len]
        # (`0` index for "PAD" ix excluded from loss)

        # Loss output will have a shape of [batch_size, session_max_len]
        # and will have zeros for every `0` target label
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y, ignore_index=0, reduction="none"
        )  # [batch_size, session_max_len]
        loss = loss * w
        n = (loss > 0).to(loss.dtype)
        loss = torch.sum(loss) / torch.sum(n)
        return loss

    @classmethod
    def _calc_bce_loss(cls, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mask = y != 0
        target = torch.zeros_like(logits)
        target[:, :, 0] = 1

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )  # [batch_size, session_max_len, n_negatives + 1]
        loss = loss.mean(-1) * mask * w  # [batch_size, session_max_len]
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

    def _calc_gbce_loss(
        self, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        n_actual_items = self.torch_model.item_model.n_items - self.data_preparator.n_item_extra_tokens
        n_negatives = negatives.shape[2]
        logits = self._get_reduced_overconfidence_logits(logits, n_actual_items, n_negatives)
        loss = self._calc_bce_loss(logits, y, w)
        return loss

    def on_train_end(self) -> None:
        """Save item embeddings"""
        self.eval()
        self.item_embs = self.torch_model.item_model.get_all_embeddings()

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Prediction step.
        Encode user sessions.
        """
        encoded_sessions = self.torch_model.encode_sessions(batch["x"], self.item_embs.to(self.device))[:, -1, :]
        return encoded_sessions

    def _xavier_normal_init(self) -> None:
        for _, param in self.torch_model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError:
                pass


class TransformerModelBase(ModelBase):  # pylint: disable=too-many-instance-attributes
    """
    Base model for all recommender algorithms that work on transformer architecture (e.g. SASRec, Bert4Rec).
    To create a custom transformer model it is necessary to inherit from this class
    and write self.data_preparator initialization logic.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        transformer_layers_type: tp.Type[TransformerLayersBase],
        data_preparator_type: tp.Type[SessionEncoderDataPreparatorBase],
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        dropout_rate: float = 0.2,
        session_max_len: int = 32,
        loss: str = "softmax",
        gbce_t: float = 0.5,
        lr: float = 0.01,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_device: Union[str, Accelerator] = "auto",
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
        top_k_saved_val_reco: tp.Optional[int] = None,
    ) -> None:
        super().__init__(verbose=verbose)
        self.recommend_n_threads = recommend_n_threads
        self.recommend_device = recommend_device
        self.recommend_use_gpu_ranking = recommend_use_gpu_ranking
        self._torch_model = TransformerBasedSessionEncoder(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            session_max_len=session_max_len,
            dropout_rate=dropout_rate,
            use_pos_emb=use_pos_emb,
            use_causal_attn=use_causal_attn,
            use_key_padding_mask=use_key_padding_mask,
            transformer_layers_type=transformer_layers_type,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
        )
        self.lightning_model: SessionEncoderLightningModuleBase
        self.lightning_module_type = lightning_module_type
        self.fit_trainer: Trainer
        if trainer is None:
            self._trainer = Trainer(
                max_epochs=epochs,
                min_epochs=epochs,
                deterministic=deterministic,
                enable_progress_bar=verbose > 0,
                enable_model_summary=verbose > 0,
                logger=verbose > 0,
            )
        else:
            self._trainer = trainer
        self.data_preparator: SessionEncoderDataPreparatorBase
        self.u2i_dist = Distance.DOT
        self.i2i_dist = Distance.COSINE
        self.lr = lr
        self.loss = loss
        self.gbce_t = gbce_t
        self.top_k_saved_val_reco = top_k_saved_val_reco

    def _fit(self, dataset: Dataset) -> None:
        self.data_preparator.process_dataset_train(dataset)
        train_dataloader = self.data_preparator.get_dataloader_train()
        val_dataloader = self.data_preparator.get_dataloader_val()

        torch_model = deepcopy(self._torch_model)  # TODO: check that it works
        torch_model.construct_item_net(self.data_preparator.train_dataset)

        self.lightning_model = self.lightning_module_type(  # TODO: data_preparator arg lightning_model
            torch_model=torch_model,
            lr=self.lr,
            loss=self.loss,
            gbce_t=self.gbce_t,
            data_preparator=self.data_preparator,
            verbose=self.verbose,
            top_k_saved_val_reco=self.top_k_saved_val_reco,
        )

        self.fit_trainer = deepcopy(self._trainer)
        self.fit_trainer.fit(self.lightning_model, train_dataloader, val_dataloader)

    def _custom_transform_dataset_u2i(
        self, dataset: Dataset, users: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        return self.data_preparator.transform_dataset_u2i(dataset, users)

    def _custom_transform_dataset_i2i(
        self, dataset: Dataset, target_items: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        return self.data_preparator.transform_dataset_i2i(dataset)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,  # [n_rec_users x n_items + n_item_extra_tokens]
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],  # model_internal
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:  # TODO: move to _get_sorted_item_ids_to_recommend
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()  # model internal

        recommend_trainer = Trainer(devices=1, accelerator=self.recommend_device)
        recommend_dataloader = self.data_preparator.get_dataloader_recommend(dataset)
        session_embs = recommend_trainer.predict(model=self.lightning_model, dataloaders=recommend_dataloader)
        if session_embs is not None:
            user_embs = np.concatenate(session_embs, axis=0)
            user_embs = user_embs[user_ids]
            item_embs = self.lightning_model.item_embs
            item_embs_np = item_embs.detach().cpu().numpy()

            ranker = ImplicitRanker(
                self.u2i_dist,
                user_embs,  # [n_rec_users, n_factors]
                item_embs_np,  # [n_items + n_item_extra_tokens, n_factors]
            )
            if filter_viewed:
                user_items = dataset.get_user_item_matrix(include_weights=False)
                ui_csr_for_filter = user_items[user_ids]
            else:
                ui_csr_for_filter = None

            # TODO: When filter_viewed is not needed and user has GPU, torch DOT and topk should be faster
            user_ids_indices, all_reco_ids, all_scores = ranker.rank(
                subject_ids=np.arange(user_embs.shape[0]),  # n_rec_users
                k=k,
                filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + n_item_extra_tokens]
                sorted_object_whitelist=sorted_item_ids_to_recommend,  # model_internal
                num_threads=self.recommend_n_threads,
                use_gpu=self.recommend_use_gpu_ranking and HAS_CUDA,
            )
            all_target_ids = user_ids[user_ids_indices]
        else:
            explanation = """Received empty recommendations. Used for type-annotation"""
            raise ValueError(explanation)
        return all_target_ids, all_reco_ids, all_scores  # n_rec_users, model_internal, scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,  # model internal
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()

        item_embs = self.lightning_model.item_embs.detach().cpu().numpy()
        # TODO: i2i reco do not need filtering viewed. And user most of the times has GPU
        # Should we use torch dot and topk? Should be faster

        ranker = ImplicitRanker(
            self.i2i_dist,
            item_embs,  # [n_items + n_item_extra_tokens, n_factors]
            item_embs,  # [n_items + n_item_extra_tokens, n_factors]
        )
        return ranker.rank(
            subject_ids=target_ids,  # model internal
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model internal
            num_threads=self.recommend_n_threads,
            use_gpu=self.recommend_use_gpu_ranking and HAS_CUDA,
        )

    @property
    def torch_model(self) -> TransformerBasedSessionEncoder:
        """TODO"""
        return self.lightning_model.torch_model


# ####  --------------  SASRec Model  --------------  #### #


class SASRecModel(TransformerModelBase):
    """
    SASRec model for i2i and u2i recommendations.

    n_blocks: int, default 1
        Number of transformer blocks.
    n_heads: int, default 1
        Number of attention heads.
    n_factors: int, default 128
        Latent embeddings size.
    use_pos_emb: bool, default ``True``
        If ``True``, adds learnable positional encoding to session item embeddings.
    use_causal_attn: bool, default ``True``
        If ``True``, uses causal mask as attn_mask in Multi-head Attention.
    use_key_padding_mask: bool, default ``False``
        If ``True``, uses key_padding_mask in Multi-head Attention.
    dropout_rate: float, default 0.2
        Probability of a hidden unit to be zeroed.
    session_max_len: int, default 32
        Maximum length of user sequence.
    train_min_user_interaction: int, default 2
        Minimum number of interactions user should have to be used for training. Should be greater than 1.
    dataloader_num_workers: int, default 0
        Number of loader worker processes.
    batch_size: int, default 128
        How many samples per batch to load.
    loss: str, default "softmax"
        Loss function.
    n_negatives: int, default 1
        Number of negatives for BCE and gBCE losses.
    gbce_t: float, default 0.2
        Calibration parameter for gBCE loss.
    lr: float, default 0.01
        Learning rate.
    epochs: int, default 3
        Number of training epochs.
    verbose: int, default 0
        Verbosity level.
    deterministic: bool, default ``False``
        If ``True``, sets deterministic algorithms for PyTorch operations.
        Use `pytorch_lightning.seed_everything` together with this parameter to fix the random state.
    recommend_device: Union[str, Accelerator], default "auto"
        Device for recommend. Used at predict_step of lightning module.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_device` attribute.
    recommend_n_threads: int, default 0
        Number of threads to use in ranker.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_n_threads` attribute.
    recommend_use_gpu_ranking: bool, default ``True``
        If ``True`` and HAS_CUDA ``True``, sets use_gpu=True in ImplicitRanker.rank.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_use_gpu_ranking` attribute.
    trainer: Optional(Trainer), default None
        Which trainer to use for training.
        If trainer is None, default pytorch_lightning Trainer is created.
    item_net_block_types: Type(ItemNetBase), default (IdEmbeddingsItemNet, CatFeaturesItemNet)
        Type of network returning item enbeddings.
        (IdEmbeddingsItemNet,) - item embeddings based on ids.
        (, CatFeaturesItemNet) - item embeddings based on categorical features.
        (IdEmbeddingsItemNet, CatFeaturesItemNet) - item embeddings based on ids and categorical features.
    pos_encoding_type: Type(PositionalEncodingBase), default `LearnableInversePositionalEncoding`
        Type of positional encoding.
    transformer_layers_type: Type(TransformerLayersBase), default `SasRecTransformerLayers`
        Type of transformer layers architecture.
    data_preparator_type: Type(SessionEncoderDataPreparatorBase), default `SasRecDataPreparator`
        Type of data preparator used for dataset processing and dataloader creation.
    lightning_module_type: Type(SessionEncoderLightningModuleBase), default `SessionEncoderLightningModule`
        Type of lightning module defining training procedure.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        dropout_rate: float = 0.2,
        session_max_len: int = 32,
        dataloader_num_workers: int = 0,
        batch_size: int = 128,
        loss: str = "softmax",
        n_negatives: int = 1,
        gbce_t: float = 0.2,
        lr: float = 0.01,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_device: Union[str, Accelerator] = "auto",
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        train_min_user_interaction: int = 2,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = SASRecTransformerLayers,  # SASRec authors net
        data_preparator_type: tp.Type[SessionEncoderDataPreparatorBase] = SASRecDataPreparator,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
        top_k_saved_val_reco: tp.Optional[int] = None,
        get_val_mask_func: tp.Optional[tp.Callable] = None,
    ):
        super().__init__(
            transformer_layers_type=transformer_layers_type,
            data_preparator_type=data_preparator_type,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_factors=n_factors,
            use_pos_emb=use_pos_emb,
            use_causal_attn=use_causal_attn,
            use_key_padding_mask=use_key_padding_mask,
            dropout_rate=dropout_rate,
            session_max_len=session_max_len,
            loss=loss,
            gbce_t=gbce_t,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
            deterministic=deterministic,
            recommend_device=recommend_device,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
            trainer=trainer,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
            top_k_saved_val_reco=top_k_saved_val_reco,
        )
        self.data_preparator = data_preparator_type(
            session_max_len=session_max_len,
            n_negatives=n_negatives if loss != "softmax" else None,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interaction,
            get_val_mask_func=get_val_mask_func,
        )
