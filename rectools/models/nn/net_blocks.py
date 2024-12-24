import torch
from torch import nn


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


class TransformerLayersBase(nn.Module):
    """Base class for transformer layers."""

    def forward(
        self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError()


class PreLNTransformerLayers(TransformerLayersBase):
    """
    Pre-LN Transformer Layers as described in "On Layer Normalization in the Transformer
    Architecture" https://arxiv.org/pdf/2002.04745
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
            [nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True) for _ in range(n_blocks)]
        )
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.dropout1 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.feed_forward = nn.ModuleList(
            [PointWiseFeedForward(n_factors, n_factors * 4, dropout_rate, torch.nn.GELU()) for _ in range(n_blocks)]
        )
        self.dropout2 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        self.dropout3 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])

    def forward(
        self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        for i in range(self.n_blocks):
            mha_input = self.layer_norm1[i](seqs)
            mha_output, _ = self.multi_head_attn[i](
                mha_input,
                mha_input,
                mha_input,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            seqs = seqs + self.dropout1[i](mha_output)
            ff_input = self.layer_norm2[i](seqs)
            ff_output = self.feed_forward[i](ff_input)
            seqs = seqs + self.dropout2[i](ff_output)
            seqs = self.dropout3[i](seqs)
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
