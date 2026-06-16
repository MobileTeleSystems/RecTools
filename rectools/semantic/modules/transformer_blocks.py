import math
import typing as tp

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


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
        self,
        n_factors: int,
        n_factors_ff: int,
        dropout_rate: float,
        activation: torch.nn.Module,
        bias: bool = True,
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
    n_factors: int,
    ff_factors_multiplier: int,
    dropout_rate: float,
    ff_activation: str,
    bias: bool = True,
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
            n_factors,
            n_factors * ff_factors_multiplier,
            dropout_rate,
            activation=torch.nn.GELU(),
            bias=bias,
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


class T5RMSNorm(nn.Module):
    """
    T5-style RMSNorm: normalise by root-mean-square, then scale.

    Unlike ``nn.LayerNorm`` this has **no bias** and does **not**
    subtract the mean -- it only divides by the RMS and applies a
    learned scale (weight) vector.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    eps : float
        Epsilon for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class T5RelativePositionBias(nn.Module):
    """
    Learned relative-position bias table with log-spaced bucketing, as
    described in the T5 paper (Raffel et al., 2019).

    The bias is added directly to the attention logits and is shared
    across all layers of the encoder (or decoder).

    Parameters
    ----------
    num_heads : int
        Number of attention heads (one scalar bias per head).
    bidirectional : bool
        ``True`` for the encoder (positions attend in both directions),
        ``False`` for the decoder (causal -- only attends to earlier
        positions).
    num_buckets : int
        Number of distance buckets.
    max_distance : int
        Distances beyond this value are clamped to the last bucket.
    """

    def __init__(
        self,
        num_heads: int,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool,
        num_buckets: int,
        max_distance: int,
    ) -> torch.Tensor:
        """Map signed relative positions to bucket indices."""
        ret = torch.zeros_like(relative_position)

        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = relative_position.abs()
        else:
            # Clamp future positions to 0 (causal -- no peeking ahead).
            n = (-relative_position).clamp(min=0)

        # Half the buckets are for exact small distances.
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half are for log-spaced larger distances.
        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = val_if_large.clamp(max=num_buckets - 1)

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute position bias.

        Returns
        -------
        Tensor [1, num_heads, query_len, key_len]
            Additive bias to be added to the attention logits.
        """
        q_pos = torch.arange(query_len, dtype=torch.long, device=device)
        k_pos = torch.arange(key_len, dtype=torch.long, device=device)
        # relative_position[i, j] = j - i  (key pos minus query pos)
        relative_position = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)

        buckets = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        # [query_len, key_len, num_heads]
        values = self.relative_attention_bias(buckets)
        # -> [1, num_heads, query_len, key_len]
        return values.permute(2, 0, 1).unsqueeze(0)


class T5Attention(nn.Module):
    """
    Multi-head attention with separate key/value dimension, as in T5.

    Unlike ``nn.MultiheadAttention`` where the per-head dimension is
    always ``embed_dim // num_heads``, this module lets you choose
    ``d_kv`` freely.  Projections have no bias (following T5).

    Uses ``scaled_dot_product_attention`` for the core computation,
    which dispatches to FlashAttention / memory-efficient backends
    automatically.

    Parameters
    ----------
    d_model : int
        Model (query input) dimension.
    num_heads : int
        Number of attention heads.
    d_kv : int
        Per-head dimension for keys and values (and queries).
    dropout : float
        Attention dropout probability (applied only during training).
    kv_input_dim : int, optional
        Input dimension for key/value tensors.  Defaults to ``d_model``.
        Set to a different value for cross-attention when the encoder
        output dimension differs from the decoder's ``d_model``.
    """

    # nn.TransformerEncoder/Decoder access self_attn.batch_first
    batch_first: bool = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        dropout: float = 0.0,
        kv_input_dim: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        if kv_input_dim is None:
            kv_input_dim = d_model

        self.num_heads = num_heads
        self.d_kv = d_kv
        self.inner_dim = num_heads * d_kv

        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(kv_input_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(kv_input_dim, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor] = None,
        key_padding_mask: tp.Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
        position_bias: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[torch.Tensor, None]:
        """
        Parameters
        ----------
        query : Tensor [B, T_q, d_model]
        key : Tensor [B, T_kv, kv_input_dim]
        value : Tensor [B, T_kv, kv_input_dim]
        attn_mask : Tensor, optional
            Additive float mask broadcastable to ``(B, num_heads, T_q, T_kv)``.
            ``-inf`` blocks attention.
        key_padding_mask : BoolTensor [B, T_kv], optional
            ``True`` at positions that should be masked (padding).
        need_weights : bool
            Ignored (always returns ``None`` for weights).
        is_causal : bool
            If ``True``, applies a causal mask inside SDPA.
        position_bias : Tensor, optional
            Additive relative-position bias broadcastable to
            ``(B, num_heads, T_q, T_kv)``.  Produced by
            ``T5RelativePositionBias``.

        Returns
        -------
        (output, None)
            output : Tensor [B, T_q, d_model]
        """
        B, T_q, _ = query.shape
        T_kv = key.shape[1]

        # Project and reshape to (B, num_heads, T, d_kv)
        q = self.q_proj(query).view(B, T_q, self.num_heads, self.d_kv).transpose(1, 2)
        k = self.k_proj(key).view(B, T_kv, self.num_heads, self.d_kv).transpose(1, 2)
        v = self.v_proj(value).view(B, T_kv, self.num_heads, self.d_kv).transpose(1, 2)

        # Convert key_padding_mask [B, T_kv] to additive attention mask
        # broadcastable to (B, num_heads, T_q, T_kv).
        # The mask may arrive as bool (True=pad) or as a float tensor
        # (0/-inf) -- nn.TransformerEncoder runs _canonical_mask which
        # converts bool to float before calling the layer.
        effective_mask = attn_mask
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                pad_mask = torch.zeros(key_padding_mask.shape, dtype=q.dtype, device=q.device).masked_fill_(
                    key_padding_mask, float("-inf")
                )
            else:
                pad_mask = key_padding_mask.to(dtype=q.dtype)
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T_kv]
            if effective_mask is not None:
                effective_mask = effective_mask + pad_mask
            else:
                effective_mask = pad_mask

        # Add relative position bias to attention logits.
        if position_bias is not None:
            if effective_mask is not None:
                effective_mask = effective_mask + position_bias
            else:
                effective_mask = position_bias

        # When an explicit mask is provided, don't also pass is_causal=True
        # -- SDPA does not allow both at the same time.
        if effective_mask is not None:
            is_causal = False

        out = scaled_dot_product_attention(  # pylint: disable=not-callable
            q,
            k,
            v,
            attn_mask=effective_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )  # [B, num_heads, T_q, d_kv]

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.inner_dim)
        out = self.o_proj(out)
        return out, None


class T5EncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer with T5-style attention.

    Architecture per step::

        x_norm = RMSNorm(x)
        x = x + Dropout(SelfAttention(x_norm, x_norm, x_norm))
        x_norm = RMSNorm(x)
        x = x + Dropout(FFN(x_norm))

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    d_kv : int
        Per-head key/value (and query) dimension.
    dim_feedforward : int
        Inner dimension of the feed-forward network.
    dropout : float
        Dropout probability.
    activation : str
        FFN activation (``"relu"`` or ``"gelu"``).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = T5Attention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
        )
        self.norm1 = T5RMSNorm(d_model)
        self.norm2 = T5RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation: nn.Module
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: tp.Optional[torch.Tensor] = None,
        position_bias: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass."""
        x = src
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=src_key_padding_mask,
            position_bias=position_bias,
        )
        x = x + self.dropout1(attn_out)

        x_norm = self.norm2(x)
        ff_out = self.linear2(self.activation(self.linear1(x_norm)))
        x = x + self.dropout2(ff_out)
        return x


class T5DecoderLayer(nn.Module):
    """
    Pre-norm Transformer decoder layer with T5-style attention.

    Architecture per step::

        x_norm = RMSNorm(x)
        x = x + Dropout(CausalSelfAttention(x_norm, x_norm, x_norm))
        x_norm = RMSNorm(x)
        x = x + Dropout(CrossAttention(x_norm, memory, memory))
        x_norm = RMSNorm(x)
        x = x + Dropout(FFN(x_norm))

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    d_kv : int
        Per-head key/value (and query) dimension.
    dim_feedforward : int
        Inner dimension of the feed-forward network.
    dropout : float
        Dropout probability.
    activation : str
        FFN activation (``"relu"`` or ``"gelu"``).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = T5Attention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
        )
        self.cross_attn = T5Attention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout=dropout,
            kv_input_dim=d_model,
        )
        self.norm1 = T5RMSNorm(d_model)
        self.norm2 = T5RMSNorm(d_model)
        self.norm3 = T5RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation: nn.Module
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: tp.Optional[torch.Tensor] = None,
        memory_key_padding_mask: tp.Optional[torch.Tensor] = None,
        position_bias: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass."""
        x = tgt

        # Self-attention (causal)
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=tgt_mask,
            position_bias=position_bias,
        )
        x = x + self.dropout1(sa_out)

        # Cross-attention (no position bias per T5 design)
        x_norm = self.norm2(x)
        ca_out, _ = self.cross_attn(
            x_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout2(ca_out)

        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.linear2(self.activation(self.linear1(x_norm)))
        x = x + self.dropout3(ff_out)
        return x
