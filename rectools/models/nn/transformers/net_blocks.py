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


class PointWiseFeedForward(nn.Module):
    """
    Feed-Forward network to introduce nonlinearity into the transformer model.
    This implementation is the one used by SASRec authors.

    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    n_factors_ff : int
        How many hidden units to use in the network.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    activation: torch.nn.Module
        Activation function module.
    bias: bool, default ``True``
        If ``True``, add bias to linear layers.
    """

    def __init__(
        self, n_factors: int, n_factors_ff: int, dropout_rate: float, activation: torch.nn.Module, bias: bool = True
    ) -> None:
        super().__init__()
        self.ff_linear_1 = nn.Linear(n_factors, n_factors_ff, bias)
        self.ff_dropout_1 = torch.nn.Dropout(dropout_rate)
        self.ff_activation = activation
        self.ff_linear_2 = nn.Linear(n_factors_ff, n_factors, bias)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.

        Returns
        -------
        torch.Tensor
            User sequence that passed through all layers.
        """
        output = self.ff_activation(self.ff_linear_1(seqs))
        fin = self.ff_linear_2(self.ff_dropout_1(output))
        return fin


class SwigluFeedForward(nn.Module):
    """
    Feed-Forward network to introduce nonlinearity into the transformer model.
    This implementation is based on FuXi and LLama SwigLU https://arxiv.org/pdf/2502.03036,
    LiGR https://arxiv.org/pdf/2502.03417

    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    n_factors_ff : int
        How many hidden units to use in the network.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    bias: bool, default ``True``
        If ``True``, add bias to linear layers.
    """

    def __init__(self, n_factors: int, n_factors_ff: int, dropout_rate: float, bias: bool = True) -> None:
        super().__init__()
        self.ff_linear_1 = nn.Linear(n_factors, n_factors_ff, bias=bias)
        self.ff_dropout_1 = torch.nn.Dropout(dropout_rate)
        self.ff_activation = torch.nn.SiLU()
        self.ff_linear_2 = nn.Linear(n_factors_ff, n_factors, bias=bias)
        self.ff_linear_3 = nn.Linear(n_factors, n_factors_ff, bias=bias)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.

        Returns
        -------
        torch.Tensor
            User sequence that passed through all layers.
        """
        output = self.ff_activation(self.ff_linear_1(seqs)) * self.ff_linear_3(seqs)
        fin = self.ff_linear_2(self.ff_dropout_1(output))
        return fin


def init_feed_forward(
    n_factors: int, ff_factors_multiplier: int, dropout_rate: float, ff_activation: str, bias: bool = True
) -> nn.Module:
    """
    Initialise Feed-Forward network with one of activation functions: "swiglu", "relu", "gelu".

    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    ff_factors_multiplier : int
        How many hidden units to use in the network.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    ff_activation : {"swiglu", "relu", "gelu"}
        Activation function to use.
    bias: bool, default ``True``
        If ``True``, add bias to linear layers.

    Returns
    -------
    nn.Module
        Feed-Forward network.
    """
    if ff_activation == "swiglu":
        return SwigluFeedForward(n_factors, n_factors * ff_factors_multiplier, dropout_rate, bias=bias)
    if ff_activation == "gelu":
        return PointWiseFeedForward(
            n_factors, n_factors * ff_factors_multiplier, dropout_rate, activation=torch.nn.GELU(), bias=bias
        )
    if ff_activation == "relu":
        return PointWiseFeedForward(
            n_factors,
            n_factors * ff_factors_multiplier,
            dropout_rate,
            activation=torch.nn.ReLU(),
            bias=bias,
        )
    raise ValueError(f"Unsupported ff_activation: {ff_activation}")


class TransformerLayersBase(nn.Module):
    """Base class for transformer layers."""

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
        raise NotImplementedError()


class PreLNTransformerLayer(nn.Module):
    """
    Pre-LN Transformer Layer as described in "On Layer Normalization in the Transformer
    Architecture" https://arxiv.org/pdf/2002.04745

    Parameters
    ----------
    n_factors: int
        Latent embeddings size.
    n_heads: int
        Number of attention heads.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    ff_factors_multiplier: int
        Feed-forward layers latent embedding size multiplier.
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
    ):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(n_factors)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(n_factors)
        self.feed_forward = PointWiseFeedForward(
            n_factors, n_factors * ff_factors_multiplier, dropout_rate, torch.nn.GELU()
        )
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)

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
        seqs = seqs + self.dropout_1(mha_output)
        ff_input = self.layer_norm_2(seqs)
        ff_output = self.feed_forward(ff_input)
        seqs = seqs + self.dropout_2(ff_output)
        seqs = self.dropout_3(seqs)
        return seqs


class PreLNTransformerLayers(TransformerLayersBase):
    """
    Pre-LN Transformer blocks.

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
    ff_factors_multiplier: int
        Feed-forward layers latent embedding size multiplier.
    """

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
        **kwargs: tp.Any,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.transformer_blocks = nn.ModuleList(
            [
                PreLNTransformerLayer(
                    n_factors,
                    n_heads,
                    dropout_rate,
                    ff_factors_multiplier,
                )
                for _ in range(self.n_blocks)
            ]
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


class PositionalEncodingBase(torch.nn.Module):
    """Base class for positional encoding."""

    def forward(self, sessions: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError()


class LearnableInversePositionalEncoding(PositionalEncodingBase):
    """
    Class to introduce learnable positional embeddings.

    Parameters
    ----------
    use_pos_emb : bool
        If ``True``, learnable positional encoding will be added to session item embeddings.
    session_max_len : int
        Maximum length of user sequence.
    n_factors : int
        Latent embeddings size.
    use_scale_factor : int
        Use multiplication embedding on the root of the dimension embedding
    """

    def __init__(
        self,
        use_pos_emb: bool,
        session_max_len: int,
        n_factors: int,
        use_scale_factor: bool = False,
        **kwargs: tp.Any,
    ):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(session_max_len, n_factors) if use_pos_emb else None
        self.use_scale_factor = use_scale_factor

    def forward(self, sessions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add learnable positional encoding to sessions and mask padding elements.

        Parameters
        ----------
        sessions : torch.Tensor
            User sessions in the form of sequences of items ids.

        Returns
        -------
        torch.Tensor
            Encoded user sessions with added positional encoding if `use_pos_emb` is ``True``.
        """
        batch_size, session_max_len, n_factors = sessions.shape

        if self.use_scale_factor:
            sessions = sessions * (n_factors**0.5)
        if self.pos_emb is not None:
            # Inverse positions are appropriate for variable length sequences across different batches
            # They are equal to absolute positions for fixed sequence length across different batches
            positions = torch.tile(
                torch.arange(session_max_len - 1, -1, -1), (batch_size, 1)
            )  # [batch_size, session_max_len]
            sessions += self.pos_emb(positions.to(sessions.device))

        return sessions
