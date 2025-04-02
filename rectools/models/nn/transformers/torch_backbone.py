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

from ..item_net import ItemNetBase
from .net_blocks import PositionalEncodingBase, TransformerLayersBase
from .similarity import SimilarityModuleBase


class TransformerBackboneBase(torch.nn.Module):
    """Base class for transformer torch backbone."""

    def __init__(
        self,
        n_heads: int,
        dropout_rate: float,
        item_model: ItemNetBase,
        pos_encoding_layer: PositionalEncodingBase,
        transformer_layers: TransformerLayersBase,
        similarity_module: SimilarityModuleBase,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        **kwargs: tp.Any,
    ) -> None:
        """
        Initialize transformer torch backbone.

        Parameters
        ----------
        n_heads : int
            Number of attention heads.
        dropout_rate : float
            Probability of a hidden unit to be zeroed.
        item_model : ItemNetBase
            Network for item embeddings.
        pos_encoding_layer : PositionalEncodingBase
            Positional encoding layer.
        transformer_layers : TransformerLayersBase
            Transformer layers.
        similarity_module : SimilarityModuleBase
            Similarity module.
        use_causal_attn : bool, default True
            If ``True``, causal mask is used in multi-head self-attention.
        use_key_padding_mask : bool, default False
            If ``True``, key padding mask is used in multi-head self-attention.
        **kwargs : Any
            Additional keyword arguments for future extensions.
        """
        super().__init__()

        self.item_model = item_model
        self.pos_encoding_layer = pos_encoding_layer
        self.emb_dropout = torch.nn.Dropout(dropout_rate)
        self.transformer_layers = transformer_layers
        self.similarity_module = similarity_module
        self.use_causal_attn = use_causal_attn
        self.use_key_padding_mask = use_key_padding_mask
        self.n_heads = n_heads

    def encode_sessions(self, batch: tp.Dict[str, torch.Tensor], item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings.
        Add positional encoding.
        Pass history through transformer blocks.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing user sessions data.
        item_embs : torch.Tensor
            Item embeddings.

        Returns
        -------
        torch.Tensor. [batch_size, session_max_len, n_factors]
            Encoded session embeddings.
        """
        raise NotImplementedError()

    def forward(
        self,
        batch: tp.Dict[str, torch.Tensor],  # batch["x"]: [batch_size, session_max_len]
        candidate_item_ids: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to get item and session embeddings.
        Get item embeddings.
        Pass user sessions through transformer blocks.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing user sessions data, with "x" key containing session tensor.
        candidate_item_ids : optional(torch.Tensor), default ``None``
            Defined item ids for similarity calculation.

        Returns
        -------
        torch.Tensor
        """
        raise NotImplementedError()


class TransformerTorchBackbone(TransformerBackboneBase):
    """
    Torch model for encoding user sessions based on transformer architecture.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    item_model : ItemNetBase
        Network for item embeddings.
    pos_encoding_layer : PositionalEncodingBase
        Positional encoding layer.
    transformer_layers : TransformerLayersBase
        Transformer layers.
    similarity_module : SimilarityModuleBase
        Similarity module.
    use_causal_attn : bool, default True
        If ``True``, causal mask is used in multi-head self-attention.
    use_key_padding_mask : bool, default False
        If ``True``, key padding mask is used in multi-head self-attention.
    **kwargs : Any
        Additional keyword arguments for future extensions.
    """

    def __init__(
        self,
        n_heads: int,
        dropout_rate: float,
        item_model: ItemNetBase,
        pos_encoding_layer: PositionalEncodingBase,
        transformer_layers: TransformerLayersBase,
        similarity_module: SimilarityModuleBase,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            item_model=item_model,
            pos_encoding_layer=pos_encoding_layer,
            transformer_layers=transformer_layers,
            similarity_module=similarity_module,
            use_causal_attn=use_causal_attn,
            use_key_padding_mask=use_key_padding_mask,
            **kwargs,
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
            .reshape(-1, seq_len, seq_len)
        )  # [batch_size * n_heads, session_max_len, session_max_len]
        torch.diagonal(res, dim1=1, dim2=2).zero_()
        return res

    def encode_sessions(self, batch: tp.Dict[str, torch.Tensor], item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings.
        Add positional encoding.
        Pass history through transformer blocks.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing user sessions data.
        item_embs : torch.Tensor
            Item embeddings.

        Returns
        -------
        torch.Tensor. [batch_size, session_max_len, n_factors]
            Encoded session embeddings.
        """
        sessions = batch["x"]  # [batch_size, session_max_len]
        session_max_len = sessions.shape[1]
        attn_mask = None
        key_padding_mask = None

        timeline_mask = (sessions != 0).unsqueeze(-1)  # [batch_size, session_max_len, 1]

        seqs = item_embs[sessions]  # [batch_size, session_max_len, n_factors]
        seqs = self.pos_encoding_layer(seqs)
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
        batch: tp.Dict[str, torch.Tensor],  # batch["x"]: [batch_size, session_max_len]
        candidate_item_ids: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to get item and session embeddings.
        Get item embeddings.
        Pass user sessions through transformer blocks.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing user sessions data, with "x" key containing session tensor.
        candidate_item_ids : optional(torch.Tensor), default ``None``
            Defined item ids for similarity calculation.

        Returns
        -------
        torch.Tensor
        """
        item_embs = self.item_model.get_all_embeddings()  # [n_items + n_item_extra_tokens, n_factors]
        session_embs = self.encode_sessions(batch, item_embs)  # [batch_size, session_max_len, n_factors]
        logits = self.similarity_module(session_embs, item_embs, candidate_item_ids)
        return logits
