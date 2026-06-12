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

"""Flat SASRec network: pre-norm transformer encoder with plain id embeddings."""

import typing as tp

import torch
from torch import nn


class SASRecBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual."""

    def __init__(self, n_factors: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_factors)
        self.mha = nn.MultiheadAttention(n_factors, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_factors)
        self.ffn = nn.Sequential(
            nn.Linear(n_factors, n_factors * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_factors * 4, n_factors),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor] = None,
        key_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply pre-norm attention and FFN with residual connections."""
        h = self.ln1(x)
        h, _ = self.mha(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


class FlatSASRecNet(nn.Module):
    """
    Flat SASRec network: sequential recommender with plain id-embedding table
    (no ItemNet hierarchy).

    Parameters
    ----------
    n_items : int
        Total number of items (excluding padding token 0).
    n_factors : int
        Embedding / hidden dimension.
    n_blocks : int
        Number of transformer blocks.
    n_heads : int
        Number of attention heads.
    session_max_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    """

    PADDING_IDX = 0

    def __init__(
        self,
        n_items: int,
        n_factors: int,
        n_blocks: int,
        n_heads: int,
        session_max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = n_factors
        self.session_max_len = session_max_len

        # +1 for padding at index 0
        self.item_emb = nn.Embedding(n_items + 1, n_factors, padding_idx=self.PADDING_IDX)
        self.pos_emb = nn.Embedding(session_max_len, n_factors)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([SASRecBlock(n_factors, n_heads, dropout) for _ in range(n_blocks)])
        self.final_ln = nn.LayerNorm(n_factors)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode full sequence.

        Parameters
        ----------
        x : LongTensor (B, L)
            Item id sequences (0 = padding).

        Returns
        -------
        Tensor (B, L, D)
        """
        _B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.item_emb(x) + self.pos_emb(positions)
        h = self.emb_dropout(h)

        # timeline_mask: zero out padding positions to prevent NaN from attention
        timeline_mask = (x != self.PADDING_IDX).unsqueeze(-1).float()  # (B, L, 1)
        attn_mask = self._causal_mask(L, x.device)
        key_padding_mask = x == self.PADDING_IDX

        for block in self.blocks:
            h = h * timeline_mask
            h = block(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        h = h * timeline_mask
        h = self.final_ln(h)
        return h

    def encode_last(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and return only the last non-padding position representation.

        Parameters
        ----------
        x : LongTensor (B, L)

        Returns
        -------
        Tensor (B, D)
        """
        h = self.encode(x)  # (B, L, D)
        return h[:, -1, :]  # left-padded: last position is always rightmost

    def all_item_embeddings(self) -> torch.Tensor:
        """
        Return embeddings for all items (1..n_items), excluding padding.

        Returns
        -------
        Tensor (n_items, D)
        """
        ids = torch.arange(1, self.n_items + 1, device=self.item_emb.weight.device)
        return self.item_emb(ids)

    def forward(self, batch: tp.Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training forward pass.

        Parameters
        ----------
        batch : dict
            Must contain 'x' (B, L) and 'y' (B, L).
            Optionally 'negatives' (B, L, N) for candidate-logits branch.

        Returns
        -------
        logits : Tensor
            If negatives present: (B, L, 1 + N) — positive + negative logits.
            Otherwise: (B, L, n_items) — full catalog logits.
        """
        x = batch["x"]  # (B, L)
        y = batch["y"]  # (B, L)

        h = self.encode(x)  # (B, L, D)

        if "negatives" in batch:
            negatives = batch["negatives"]  # (B, L, N)
            pos_emb = self.item_emb(y).unsqueeze(3)  # (B, L, D, 1)
            neg_emb = self.item_emb(negatives)  # (B, L, N, D)
            neg_emb = neg_emb.transpose(2, 3)  # (B, L, D, N)
            all_emb = torch.cat([pos_emb, neg_emb], dim=3)  # (B, L, D, 1+N)
            logits = (h.unsqueeze(2) @ all_emb).squeeze(2)  # (B, L, 1+N)
            # -> shape is (B, L, 1+N) where first column is positive logit
        else:
            item_embs = self.all_item_embeddings()  # (n_items, D)
            logits = h @ item_embs.T  # (B, L, n_items)
        return logits
