import math
import typing as tp

import torch
from torch import nn
from torch.nn.functional import log_softmax

from rectools.semantic.modules import T5DecoderLayer, T5EncoderLayer, T5RMSNorm, T5RelativePositionBias


class T5Encoder(nn.Module):
    """
    T5-style encoder stack with shared relative position bias.

    Replaces ``nn.TransformerEncoder`` for the T5 code path so that a
    single ``T5RelativePositionBias`` is computed once and shared
    across all layers.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    d_kv : int
        Per-head key/value (and query) dimension.
    num_layers : int
        Number of encoder layers.
    dim_feedforward : int
        FFN inner dimension.
    dropout : float
        Dropout probability.
    activation : str
        FFN activation.
    relative_position_num_buckets : int
        Number of buckets for relative position bias.
    relative_position_max_distance : int
        Max distance for relative position bucketing.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        relative_position_num_buckets: int = 32,
        relative_position_max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                T5EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_kv=d_kv,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = T5RMSNorm(d_model)
        self.position_bias = T5RelativePositionBias(
            num_heads=num_heads,
            bidirectional=True,
            num_buckets=relative_position_num_buckets,
            max_distance=relative_position_max_distance,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_bias = self.position_bias(query_len=src.size(1), key_len=src.size(1), device=src.device)
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, position_bias=pos_bias)
        return self.final_norm(x)


class T5Decoder(nn.Module):
    """
    T5-style decoder stack with shared relative position bias.

    Replaces ``nn.TransformerDecoder`` for the T5 code path so that a
    single ``T5RelativePositionBias`` (causal / unidirectional) is
    computed once and shared across all layers.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int
        Number of attention heads.
    d_kv : int
        Per-head key/value (and query) dimension.
    num_layers : int
        Number of decoder layers.
    dim_feedforward : int
        FFN inner dimension.
    dropout : float
        Dropout probability.
    activation : str
        FFN activation.
    relative_position_num_buckets : int
        Number of buckets for relative position bias.
    relative_position_max_distance : int
        Max distance for relative position bucketing.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        relative_position_num_buckets: int = 32,
        relative_position_max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                T5DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_kv=d_kv,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = T5RMSNorm(d_model)
        self.position_bias = T5RelativePositionBias(
            num_heads=num_heads,
            bidirectional=False,
            num_buckets=relative_position_num_buckets,
            max_distance=relative_position_max_distance,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: tp.Optional[torch.Tensor] = None,
        memory_key_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_bias = self.position_bias(query_len=tgt.size(1), key_len=tgt.size(1), device=tgt.device)
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                position_bias=pos_bias,
            )
        return self.final_norm(x)


class TIGERNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    Generative retrieval model for sequential recommendation.

    Items are represented as Semantic IDs -- tuples of discrete codewords
    produced by RQ-VAE.  The model is a Transformer encoder-decoder:
      * Encoder input : flattened Semantic ID tokens of the user history.
      * Decoder output: Semantic ID tokens of the next item, predicted
        autoregressively one codeword at a time.

    Special tokens
    --------------
    0 -- padding token  (in both encoder and decoder)
    1 -- BOS (beginning-of-sequence) token fed to the decoder at step 0

    All codeword tokens are offset by +2 so that the vocabulary is:
        {0: PAD, 1: BOS, 2 .. 2+sum(codebook_sizes)-1 : codewords}
    Each codebook level gets its own contiguous slice of the vocabulary.

    Parameters
    ----------
    codebook_sizes : list[int]
        Number of codes per RQ-VAE level (e.g. [256, 256, 256]).
    hidden_units : int
        Dimensionality of the Transformer hidden states.
    num_blocks : int
        Number of Transformer layers in **each** of the encoder and decoder.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout probability.
    max_length : int
        Maximum number of items in the encoder history.
    initializer_range : float
        Std for weight initialisation.
    ff_dim : int, optional
        Feed-forward inner dimension.  Defaults to ``4 * hidden_units``.
    d_kv : int, optional
        Per-head key/value (and query) projected dimension for T5-style
        attention.  When set, the encoder and decoder use T5-style layers
        with RMSNorm and relative position bias (no absolute positional
        embeddings).  When ``None`` (default), standard
        ``nn.TransformerEncoderLayer`` / ``nn.TransformerDecoderLayer``
        are used with learned absolute positional embeddings.
    relative_position_num_buckets : int
        Number of buckets for T5 relative position bias (only used when
        ``d_kv`` is set).
    relative_position_max_distance : int
        Max distance for T5 relative position bucketing (only used when
        ``d_kv`` is set).
    """

    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    CODEWORD_OFFSET = 2  # first codeword token id

    def __init__(
        self,
        codebook_sizes: tp.List[int],
        hidden_units: int = 256,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        max_length: int = 256,
        initializer_range: float = 0.02,
        ff_dim: tp.Optional[int] = None,
        d_kv: tp.Optional[int] = None,
        relative_position_num_buckets: int = 32,
        relative_position_max_distance: int = 128,
    ) -> None:
        super().__init__()

        self.codebook_sizes = codebook_sizes
        self.sid_len = len(codebook_sizes)  # number of codewords per item
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.initializer_range = initializer_range
        self.d_kv = d_kv

        # Total vocabulary: PAD + BOS + all codeword tokens
        self.vocab_size = self.CODEWORD_OFFSET + sum(codebook_sizes)

        self.enc_max_len = max_length * self.sid_len
        self.dec_max_len = self.sid_len

        if ff_dim is None:
            ff_dim = hidden_units * 4

        self.token_emb = nn.Embedding(self.vocab_size, hidden_units, padding_idx=self.PAD_TOKEN_ID)

        self.enc_dropout = nn.Dropout(dropout_rate)
        self.dec_dropout = nn.Dropout(dropout_rate)

        self.encoder: nn.Module
        self.decoder: nn.Module
        if d_kv is not None:
            # T5 path: relative position bias instead of absolute embeddings.
            self.enc_pos_emb = None
            self.dec_pos_emb = None

            self.encoder = T5Encoder(
                d_model=hidden_units,
                num_heads=num_heads,
                d_kv=d_kv,
                num_layers=num_blocks,
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu",
                relative_position_num_buckets=relative_position_num_buckets,
                relative_position_max_distance=relative_position_max_distance,
            )
            self.decoder = T5Decoder(
                d_model=hidden_units,
                num_heads=num_heads,
                d_kv=d_kv,
                num_layers=num_blocks,
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu",
                relative_position_num_buckets=relative_position_num_buckets,
                relative_position_max_distance=relative_position_max_distance,
            )
        else:
            # Standard path: learned absolute positional embeddings.
            self.enc_pos_emb = nn.Embedding(self.enc_max_len, hidden_units)
            self.dec_pos_emb = nn.Embedding(self.dec_max_len + 1, hidden_units)  # +1 for BOS step

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
                norm_first=True,
            )

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, norm=nn.LayerNorm(hidden_units))
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_blocks, norm=nn.LayerNorm(hidden_units))

        self.output_heads = nn.ModuleList()
        for cb_size in codebook_sizes:
            self.output_heads.append(nn.Linear(hidden_units, cb_size))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
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
        elif isinstance(module, T5RMSNorm):
            module.weight.data.fill_(1.0)

    def _codebook_offsets(self, device: torch.device) -> torch.Tensor:
        """Return tensor [sid_len] with the per-level offset into the vocab."""
        offsets = [self.CODEWORD_OFFSET]
        for s in self.codebook_sizes[:-1]:
            offsets.append(offsets[-1] + s)
        return torch.tensor(offsets, device=device)

    def sid_to_tokens(self, sid: torch.Tensor) -> torch.Tensor:
        offsets = self._codebook_offsets(sid.device)
        return sid + offsets

    def tokens_to_sid(self, tokens: torch.Tensor) -> torch.Tensor:
        offsets = self._codebook_offsets(tokens.device)
        return tokens - offsets

    def encode(
        self,
        enc_input: torch.Tensor,
        enc_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode the user history.

        Parameters
        ----------
        enc_input : LongTensor [B, T_enc]
            Vocabulary token ids for the encoder (already offset).
        enc_padding_mask : BoolTensor [B, T_enc], optional
            True where the token is padding.

        Returns
        -------
        memory : Tensor [B, T_enc, H]
        """
        x = self.token_emb(enc_input)
        if self.enc_pos_emb is not None:
            positions = torch.arange(enc_input.size(1), device=enc_input.device).unsqueeze(0)
            x = x + self.enc_pos_emb(positions)
        x = self.enc_dropout(x)

        memory = self.encoder(x, src_key_padding_mask=enc_padding_mask)
        return memory

    def decode(
        self,
        dec_input: torch.Tensor,
        memory: torch.Tensor,
        enc_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode with teacher forcing.

        Parameters
        ----------
        dec_input : LongTensor [B, T_dec]
            Decoder input token ids (BOS followed by target tokens shifted right).
        memory : Tensor [B, T_enc, H]
            Encoder output.
        enc_padding_mask : BoolTensor [B, T_enc], optional
            Padding mask for cross-attention.

        Returns
        -------
        Tensor [B, T_dec, H]
        """
        x = self.token_emb(dec_input)
        if self.dec_pos_emb is not None:
            x = x * math.sqrt(self.hidden_units)
            positions = torch.arange(dec_input.size(1), device=dec_input.device).unsqueeze(0)
            x = x + self.dec_pos_emb(positions)
        x = self.dec_dropout(x)

        # causal mask so position i can only attend to positions <= i
        tgt_len = dec_input.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=dec_input.device)

        out = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=enc_padding_mask,
        )
        return out

    def forward(
        self,
        enc_input: torch.Tensor,
        dec_input: torch.Tensor,
        enc_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.List[torch.Tensor]:
        memory = self.encode(enc_input, enc_padding_mask)
        hidden = self.decode(dec_input, memory, enc_padding_mask)  # [B, sid_len, H]

        logits = []
        for d in range(self.sid_len):
            logits.append(self.output_heads[d](hidden[:, d, :]))  # [B, cb_size_d]

        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        enc_input: torch.Tensor,
        enc_padding_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(enc_input, enc_padding_mask)
        B = enc_input.size(0)
        device = enc_input.device

        # Start with BOS token
        generated_tokens = [torch.full((B,), self.BOS_TOKEN_ID, dtype=torch.long, device=device)]
        offsets = self._codebook_offsets(device)
        predicted_codes = []

        for d in range(self.sid_len):
            dec_input = torch.stack(generated_tokens, dim=1)  # [B, d+1]
            hidden = self.decode(dec_input, memory, enc_padding_mask)  # [B, d+1, H]
            logits_d = self.output_heads[d](hidden[:, -1, :])  # [B, cb_size_d]
            code_d = logits_d.argmax(dim=-1)  # [B]
            predicted_codes.append(code_d)

            # Convert raw code to vocab token for the next decoder step
            token_d = code_d + offsets[d]
            generated_tokens.append(token_d)

        return torch.stack(predicted_codes, dim=1)  # [B, sid_len]

    @torch.no_grad()
    def generate(  # pylint: disable=too-many-locals
        self,
        enc_input: torch.Tensor,
        enc_padding_mask: tp.Optional[torch.Tensor] = None,
        beam_size: int = 10,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding (fully batched).

        Returns
        -------
        all_sids : LongTensor [B, beam_size, sid_len]
            Top-``beam_size`` predicted Semantic IDs (raw codewords, 0-based).
        all_scores : Tensor [B, beam_size]
            Log-probability scores for each beam.
        """
        memory = self.encode(enc_input, enc_padding_mask)
        B = enc_input.size(0)
        device = enc_input.device
        offsets = self._codebook_offsets(device)

        K = beam_size
        BK = B * K

        memory = memory.unsqueeze(1).expand(-1, K, -1, -1).reshape(BK, memory.size(1), memory.size(2))
        if enc_padding_mask is not None:
            enc_padding_mask = enc_padding_mask.unsqueeze(1).expand(-1, K, -1).reshape(BK, -1)

        scores = torch.zeros(B, K, device=device)
        scores[:, 1:] = -1e9

        all_codes = torch.zeros(B, K, 0, dtype=torch.long, device=device)

        dec_tokens = torch.full((BK, 1), self.BOS_TOKEN_ID, dtype=torch.long, device=device)

        for d in range(self.sid_len):
            hidden = self.decode(dec_tokens, memory, enc_padding_mask)
            logits_d = self.output_heads[d](hidden[:, -1, :])
            log_probs = log_softmax(logits_d, dim=-1)

            cb_size = self.codebook_sizes[d]
            topk_k = min(K, cb_size)
            topk_vals, topk_idx = log_probs.topk(topk_k, dim=-1)

            topk_vals = topk_vals.view(B, K, topk_k)
            topk_idx = topk_idx.view(B, K, topk_k)

            candidate_scores = scores.unsqueeze(-1) + topk_vals
            candidate_scores = candidate_scores.view(B, K * topk_k)
            candidate_idx = topk_idx.view(B, K * topk_k)
            candidate_beam = (
                torch.arange(K, device=device).unsqueeze(-1).expand(-1, topk_k).reshape(1, K * topk_k).expand(B, -1)
            )

            best_scores, best_flat = candidate_scores.topk(K, dim=-1)
            best_beam = candidate_beam.gather(1, best_flat)
            best_code = candidate_idx.gather(1, best_flat)

            scores = best_scores

            prev_codes = all_codes.gather(1, best_beam.unsqueeze(-1).expand(-1, -1, d))
            all_codes = torch.cat([prev_codes, best_code.unsqueeze(-1)], dim=-1)

            beam_offset = torch.arange(B, device=device).unsqueeze(1) * K
            flat_beam_idx = (beam_offset + best_beam).view(BK)

            dec_tokens = dec_tokens[flat_beam_idx]
            new_token = (best_code + offsets[d]).view(BK, 1)
            dec_tokens = torch.cat([dec_tokens, new_token], dim=1)

        return all_codes, scores
