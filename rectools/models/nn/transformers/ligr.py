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
from torch import nn

from rectools.models.nn.transformers.net_blocks import TransformerLayersBase

from .net_blocks import init_feed_forward


class LiGRLayer(nn.Module):
    """
    Transformer Layer as described in "From Features to Transformers:
    Redefining Ranking for Scalable Impact" https://arxiv.org/pdf/2502.03417

    Parameters
    ----------
    n_factors: int
        Latent embeddings size.
    n_heads: int
        Number of attention heads.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    ff_factors_multiplier: int, default 4
        Feed-forward layers latent embedding size multiplier.
    bias_in_ff: bool, default ``False``
        Add bias in Linear layers of Feed Forward
    ff_activation: {"swiglu", "relu", "gelu"}, default "swiglu"
        Activation function to use.
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
        bias_in_ff: bool = False,
        ff_activation: str = "swiglu",
    ):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(n_factors)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(n_factors)
        self.feed_forward = init_feed_forward(n_factors, ff_factors_multiplier, dropout_rate, ff_activation, bias_in_ff)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.gating_linear_1 = nn.Linear(n_factors, n_factors)
        self.gating_linear_2 = nn.Linear(n_factors, n_factors)

    def forward(
        self,
        seqs: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor],
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Parameters
        ----------
        seqs: torch.Tensor
            User sequences of item embeddings.
        attn_mask: torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `attn_mask`.
        key_padding_mask: torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `key_padding_mask`.


        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        mha_input = self.layer_norm_1(seqs)
        mha_output, _ = self.multi_head_attn(
            mha_input,
            mha_input,
            mha_input,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gated_skip = torch.nn.functional.sigmoid(self.gating_linear_1(seqs))
        seqs = seqs + torch.mul(gated_skip, self.dropout_1(mha_output))

        ff_input = self.layer_norm_2(seqs)
        ff_output = self.feed_forward(ff_input)
        gated_skip = torch.nn.functional.sigmoid(self.gating_linear_2(seqs))
        seqs = seqs + torch.mul(gated_skip, self.dropout_2(ff_output))
        return seqs


class LiGRLayers(TransformerLayersBase):
    """
    LiGR Transformer blocks.

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
    ff_factors_multiplier: int, default 4
        Feed-forward layers latent embedding size multiplier. Pass in ``transformer_layers_kwargs`` to override.
    ff_activation: {"swiglu", "relu", "gelu"}, default "swiglu"
        Activation function to use. Pass in ``transformer_layers_kwargs`` to override.
    bias_in_ff: bool, default ``False``
        Add bias in Linear layers of Feed Forward. Pass in ``transformer_layers_kwargs`` to override.
    """

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
        ff_activation: str = "swiglu",
        bias_in_ff: bool = False,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_factors = n_factors
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.ff_factors_multiplier = ff_factors_multiplier
        self.ff_activation = ff_activation
        self.bias_in_ff = bias_in_ff
        self.transformer_blocks = nn.ModuleList([self._init_transformer_block() for _ in range(self.n_blocks)])

    def _init_transformer_block(self) -> nn.Module:
        return LiGRLayer(
            self.n_factors,
            self.n_heads,
            self.dropout_rate,
            self.ff_factors_multiplier,
            bias_in_ff=self.bias_in_ff,
            ff_activation=self.ff_activation,
        )

    def forward(
        self,
        seqs: torch.Tensor,
        timeline_mask: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor],
        key_padding_mask: tp.Optional[torch.Tensor],
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """
        Forward pass through transformer blocks.

        Parameters
        ----------
        seqs: torch.Tensor
            User sequences of item embeddings.
        timeline_mask: torch.Tensor
            Mask indicating padding elements.
        attn_mask: torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `attn_mask`.
        key_padding_mask: torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `key_padding_mask`.


        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        for block_idx in range(self.n_blocks):
            seqs = self.transformer_blocks[block_idx](seqs, attn_mask, key_padding_mask)
        return seqs
