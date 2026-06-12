#  Copyright 2026 MTS (Mobile Telesystems)
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

"""UniSRec network: SASRec encoder with pretrained text embeddings and learnable adaptor."""

import typing as tp

import torch
from torch import nn


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class FeedForwardConv1d(nn.Module):
    """Point-wise FFN via Conv1d (kernel_size=1), matching the reference UniSRec."""

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply two Conv1d layers with ReLU and dropout."""
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.relu(self.dropout1(outputs))
        outputs = self.conv2(outputs)
        outputs = self.dropout2(outputs)
        return outputs.transpose(-1, -2)


# keep old name as alias
FeedForward = FeedForwardConv1d


def make_ffn(n_factors: int, ffn_type: str, expansion: int, dropout: float) -> nn.Module:
    """Create a feed-forward block.

    Parameters
    ----------
    n_factors : int
        Input and output dimension.
    ffn_type : ``"conv1d"`` | ``"linear_gelu"`` | ``"linear_relu"``
        Type of feed-forward block.
    expansion : int
        Hidden-dimension multiplier (e.g. 1 or 4).
    dropout : float
        Dropout rate applied inside the block.

    Returns
    -------
    nn.Module
        A feed-forward network module.
    """
    if ffn_type == "conv1d":
        return FeedForwardConv1d(n_factors, dropout)
    hidden = n_factors * expansion
    if ffn_type == "linear_gelu":
        return nn.Sequential(
            nn.Linear(n_factors, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_factors),
            nn.Dropout(dropout),
        )
    if ffn_type == "linear_relu":
        return nn.Sequential(
            nn.Linear(n_factors, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_factors),
        )
    raise ValueError(f"Unknown ffn_type: {ffn_type}. Choose from: conv1d, linear_gelu, linear_relu")


class UniSRecNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    UniSRec network: sequential recommender with pretrained text embeddings + adaptor.

    Architecture::

        frozen_emb  -->  adaptor (PCA/BN + optional MLP)  -->  SASRec encoder

    The adaptor projects frozen pretrained embeddings (e.g. from a
    sentence-transformer) into the transformer hidden space.  All training
    is joint — adaptor and transformer are trained together in a single phase.

    Parameters
    ----------
    n_items : int
        Number of real items (excluding padding token at index 0).
    pretrained_embeddings : Tensor
        Shape ``(n_items + 1, D_text)`` or ``(n_items + 1, n_variants, D_text)``.
        Index 0 = padding (zeros), indices 1..n_items = item text embeddings.
    n_factors : int
        Hidden / output dimension of the transformer.
    projection_hidden : int
        Intermediate dimension for the PCA adaptor head.
    n_blocks : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads.
    session_max_len : int
        Maximum sequence length (positional embedding size).
    dropout : float
        Dropout in transformer blocks.
    adaptor_dropout : float
        Dropout inside the adaptor MLP.
    adaptor_type : ``"pca"`` | ``"bn"``
        Type of adaptor for projecting pretrained embeddings.
    use_adaptor_ffn : bool
        Whether to use a 2-layer MLP head after the linear projection.
    initializer_range : float
        Std for normal weight initialisation.
    """

    PADDING_IDX = 0

    def __init__(
        self,
        n_items: int,
        pretrained_embeddings: torch.Tensor,
        n_factors: int = 256,
        projection_hidden: int = 512,
        n_blocks: int = 2,
        n_heads: int = 1,
        session_max_len: int = 200,
        dropout: float = 0.1,
        adaptor_dropout: float = 0.2,
        adaptor_type: str = "pca",
        use_adaptor_ffn: bool = True,
        initializer_range: float = 0.02,
        ffn_type: str = "conv1d",
        ffn_expansion: int = 1,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = n_factors
        self.session_max_len = session_max_len
        self.n_blocks = n_blocks
        self.adaptor_type = adaptor_type
        self.use_adaptor_ffn = use_adaptor_ffn
        self.initializer_range = initializer_range

        if not use_adaptor_ffn and adaptor_type != "pca":
            raise ValueError("use_adaptor_ffn=False is only supported with adaptor_type='pca'")

        # ── Frozen pretrained embeddings ──
        if pretrained_embeddings.ndim == 2:
            pretrained_embeddings = pretrained_embeddings.unsqueeze(1)
        self.frozen_emb: torch.Tensor
        self.register_buffer("frozen_emb", pretrained_embeddings)
        self.n_variants = pretrained_embeddings.shape[1]

        qwen_dim = pretrained_embeddings.shape[2]
        emb_for_init = pretrained_embeddings[1:, 0, :]  # skip padding row

        # ── Adaptor ──
        self.head: tp.Optional[nn.Sequential] = None
        if adaptor_type == "pca":
            self.whitening_bias = nn.Parameter(emb_for_init.mean(dim=0))
            if use_adaptor_ffn:
                self.whitening_proj = nn.Parameter(self._pca_init(emb_for_init, projection_hidden))
                proj_dim = self.whitening_proj.shape[1]
                self.head = _make_mlp(proj_dim, proj_dim, n_factors, adaptor_dropout)
            else:
                self.whitening_proj = nn.Parameter(self._pca_init(emb_for_init, n_factors))
        elif adaptor_type == "bn":
            self.bn_input = nn.BatchNorm1d(qwen_dim)
            self.bn_score = nn.BatchNorm1d(qwen_dim)
            self.head = _make_mlp(qwen_dim, n_factors, n_factors, adaptor_dropout)
        else:
            raise ValueError(f"Unknown adaptor_type: {adaptor_type}")

        # ── Positional embedding + dropout ──
        self.pos_emb = nn.Embedding(session_max_len, n_factors)
        self.emb_dropout = nn.Dropout(dropout)

        # ── Transformer blocks (pre-norm) ──
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(n_factors, eps=1e-12)

        for _ in range(n_blocks):
            self.attention_layernorms.append(nn.LayerNorm(n_factors, eps=1e-12))
            self.attention_layers.append(nn.MultiheadAttention(n_factors, n_heads, dropout, batch_first=True))
            self.forward_layernorms.append(nn.LayerNorm(n_factors, eps=1e-12))
            self.forward_layers.append(make_ffn(n_factors, ffn_type, ffn_expansion, dropout))

        self.apply(self._init_weights)

    # ── Init helpers ──

    @staticmethod
    def _pca_init(embeddings: torch.Tensor, out_dim: int) -> torch.Tensor:
        centered = embeddings - embeddings.mean(dim=0)
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)  # pylint: disable=not-callable
        out_dim = min(out_dim, Vh.shape[0])
        return Vh[:out_dim].T.contiguous()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ── Adaptor ──

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptor_type == "pca":
            projected = (x - self.whitening_bias) @ self.whitening_proj
            return self.head(projected) if self.head is not None else projected
        assert self.head is not None
        shape = x.shape
        flat = x.view(-1, shape[-1])
        return self.head(self.bn_input(flat)).view(*shape[:-1], self.n_factors)

    def _adapt_score(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptor_type == "pca":
            projected = (x - self.whitening_bias) @ self.whitening_proj
            return self.head(projected) if self.head is not None else projected
        assert self.head is not None
        shape = x.shape
        flat = x.view(-1, shape[-1])
        return self.head(self.bn_score(flat)).view(*shape[:-1], self.n_factors)

    def _sample_frozen(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Look up pretrained embeddings, sampling a random variant during training."""
        if self.n_variants == 1 or not self.training:
            return self.frozen_emb[item_ids, 0]
        vi = torch.randint(self.n_variants, item_ids.shape, device=item_ids.device)
        vi = vi * (item_ids != 0).long()  # padding always uses variant 0
        return self.frozen_emb[item_ids, vi]

    def project_all(self) -> torch.Tensor:
        """Project all frozen embeddings (variant 0) through the score adaptor.

        Returns
        -------
        Tensor (n_items + 1, n_factors)
            Adapted embeddings for every item including padding at index 0.
        """
        return self._adapt_score(self.frozen_emb[:, 0])

    # ── Encoder ──

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _encode(self, seqs: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        _B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        pad_mask = input_ids == self.PADDING_IDX  # (B, L)
        pad_mask_3d = pad_mask.unsqueeze(-1)  # (B, L, 1)
        seqs = seqs.masked_fill(pad_mask_3d, 0.0)  # zero out padding

        attn_mask = self._causal_mask(L, seqs.device)
        key_padding_mask = pad_mask

        for i in range(self.n_blocks):
            normed = self.attention_layernorms[i](seqs)
            # Zero padding in Q/K/V so NaN can never appear in dot-products
            normed = normed.masked_fill(pad_mask_3d, 0.0)
            mha_out, _ = self.attention_layers[i](
                normed,
                normed,
                normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            # masked_fill handles NaN*0 correctly (unlike multiplication)
            seqs = (seqs + mha_out).masked_fill(pad_mask_3d, 0.0)
            seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            seqs = seqs.masked_fill(pad_mask_3d, 0.0)

        return self.last_layernorm(seqs)

    # ── Public forward / encode ──

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of item IDs through the adaptor + transformer.

        Parameters
        ----------
        input_ids : LongTensor (B, L)
            Left-padded item ID sequences (0 = padding).

        Returns
        -------
        Tensor (B, L, n_factors)
        """
        seqs = self._adapt_input(self._sample_frozen(input_ids))
        return self._encode(seqs, input_ids)

    def encode_last(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode and return the last-position representation.

        Parameters
        ----------
        input_ids : LongTensor (B, L)
            Left-padded item ID sequences (0 = padding).

        Returns
        -------
        Tensor (B, n_factors)
            Representation of the last (rightmost) position for each
            sequence in the batch.
        """
        h = self.forward(input_ids)  # (B, L, D)
        return h[:, -1, :]  # left-padded → last position is always the rightmost
