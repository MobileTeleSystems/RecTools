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

import torch

from rectools.dataset.dataset import Dataset, DatasetSchema

from .item_net import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructorBase,
    SumOfEmbeddingsConstructor,
)
from .transformer_net_blocks import (
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    PreLNTransformerLayers,
    TransformerLayersBase,
)


class TransformerTorchBackbone(torch.nn.Module):
    """
    Torch model for encoding user sessions based on transformer architecture.

    Parameters
    ----------
    n_blocks : int
        Number of transformer blocks.
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    session_max_len : int
        Maximum length of user sequence.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    use_pos_emb : bool, default True
        If ``True``, learnable positional encoding will be added to session item embeddings.
    use_causal_attn : bool, default True
        If ``True``, causal mask is used in multi-head self-attention.
    transformer_layers_type : type(TransformerLayersBase), default `PreLNTransformerLayers`
        Type of transformer layers architecture.
    item_net_type : type(ItemNetBase), default `IdEmbeddingsItemNet`
        Type of network returning item embeddings.
    pos_encoding_type : type(PositionalEncodingBase), default `LearnableInversePositionalEncoding`
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
        transformer_layers_type: tp.Type[TransformerLayersBase] = PreLNTransformerLayers,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        item_net_constructor_type: tp.Type[ItemNetConstructorBase] = SumOfEmbeddingsConstructor,
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
    ) -> None:
        super().__init__()

        self.item_model: ItemNetConstructorBase
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
        self.item_net_constructor_type = item_net_constructor_type

    def construct_item_net(self, dataset: Dataset) -> None:
        """
        Construct network for item embeddings from dataset.

        Parameters
        ----------
        dataset : Dataset
            RecTools dataset with user-item interactions.
        """
        self.item_model = self.item_net_constructor_type.from_dataset(
            dataset, self.n_factors, self.dropout_rate, self.item_net_block_types
        )

    def construct_item_net_from_dataset_schema(self, dataset_schema: DatasetSchema) -> None:
        """
        Construct network for item embeddings from dataset schema.

        Parameters
        ----------
        dataset_schema : DatasetSchema
            RecTools schema with dataset statistics.
        """
        self.item_model = self.item_net_constructor_type.from_dataset_schema(
            dataset_schema, self.n_factors, self.dropout_rate, self.item_net_block_types
        )

    @staticmethod
    def _convert_mask_to_float(mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(mask, dtype=query.dtype).masked_fill_(mask, float("-inf"))

    def _merge_masks(
        self, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge `attn_mask` and `key_padding_mask` as a new `attn_mask`.
        Both masks are expanded to shape ``(batch_size * n_heads, session_max_len, session_max_len)``
        and combined with logical ``or``.
        Diagonal elements in last two dimensions are set equal to ``0``.
        This prevents nan values in gradients for pytorch < 2.5.0 when both masks are present in forward pass of
        `torch.nn.MultiheadAttention` (https://github.com/pytorch/pytorch/issues/41508).

        Parameters
        ----------
        attn_mask:  torch.Tensor. [session_max_len, session_max_len]
            Boolean causal attention mask.
        key_padding_mask: torch.Tensor. [batch_size, session_max_len]
            Boolean padding mask.
        query: torch.Tensor
            Query tensor used to acquire correct shapes and dtype for new `attn_mask`.

        Returns
        -------
        torch.Tensor. [batch_size * n_heads, session_max_len, session_max_len]
            Merged mask to use as new `attn_mask` with zeroed diagonal elements in last 2 dimensions.
        """
        batch_size, seq_len, _ = query.shape

        key_padding_mask_expanded = self._convert_mask_to_float(  # [batch_size, session_max_len]
            key_padding_mask, query
        ).view(
            batch_size, 1, seq_len
        )  # [batch_size, 1, session_max_len]

        attn_mask_expanded = (
            self._convert_mask_to_float(attn_mask, query)  # [session_max_len, session_max_len]
            .view(1, seq_len, seq_len)
            .expand(batch_size, -1, -1)
        )  # [batch_size, session_max_len, session_max_len]

        merged_mask = attn_mask_expanded + key_padding_mask_expanded
        res = (
            merged_mask.view(batch_size, 1, seq_len, seq_len)
            .expand(-1, self.n_heads, -1, -1)
            .view(-1, seq_len, seq_len)
        )  # [batch_size * n_heads, session_max_len, session_max_len]
        torch.diagonal(res, dim1=1, dim2=2).zero_()
        return res

    def encode_sessions(self, sessions: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings.
        Add positional encoding.
        Pass history through transformer blocks.

        Parameters
        ----------
        sessions :  torch.Tensor
            User sessions in the form of sequences of items ids.
        item_embs : torch.Tensor
            Item embeddings.

        Returns
        -------
        torch.Tensor. [batch_size, session_max_len, n_factors]
            Encoded session embeddings.
        """
        session_max_len = sessions.shape[1]
        attn_mask = None
        key_padding_mask = None

        timeline_mask = (sessions != 0).unsqueeze(-1)  # [batch_size, session_max_len, 1]

        seqs = item_embs[sessions]  # [batch_size, session_max_len, n_factors]
        seqs = self.pos_encoding(seqs)
        seqs = self.emb_dropout(seqs)

        if self.use_causal_attn:
            attn_mask = ~torch.tril(
                torch.ones((session_max_len, session_max_len), dtype=torch.bool, device=sessions.device)
            )
        if self.use_key_padding_mask:
            key_padding_mask = sessions == 0
            if attn_mask is not None:  # merge masks to prevent nan gradients for torch < 2.5.0
                attn_mask = self._merge_masks(attn_mask, key_padding_mask, seqs)
                key_padding_mask = None

        seqs = self.transformer_layers(seqs, timeline_mask, attn_mask, key_padding_mask)
        return seqs

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_max_len]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get item and session embeddings.
        Get item embeddings.
        Pass user sessions through transformer blocks.

        Parameters
        ----------
        sessions : torch.Tensor
            User sessions in the form of sequences of items ids.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
        """
        item_embs = self.item_model.get_all_embeddings()  # [n_items + n_item_extra_tokens, n_factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_max_len, n_factors]
        return item_embs, session_embs
