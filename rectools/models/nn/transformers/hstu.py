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
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import math
from ..item_net import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructorBase,
    SumOfEmbeddingsConstructor,
)
from .base import (
    TrainerCallable,
    TransformerDataPreparatorType,
    TransformerLayersType,
    TransformerLightningModule,
    TransformerLightningModuleBase,
    TransformerModelBase,
    TransformerModelConfig,
    ValMaskCallable,
)
from .data_preparator import InitKwargs,  BatchElement, TransformerDataPreparatorBase
from .negative_sampler import CatalogUniformSampler, TransformerNegativeSamplerBase
from .net_blocks import (
    PositionalEncodingBase,
    TransformerLayersBase, LearnableInversePositionalEncoding,
)
from .similarity import DistanceSimilarityModule, SimilarityModuleBase
from .torch_backbone import HSTUTorchBackbone, TransformerBackboneBase
from rectools import Columns
from rectools.dataset import Dataset
check_split_train_dataset = True
check_split_val_dataset = True
class HSTUDataPreparator(TransformerDataPreparatorBase):
    """Data preparator for HSTUModel.

    Parameters
    ----------
    session_max_len : int
        Maximum length of user sequence.
    batch_size : int
        How many samples per batch to load.
    dataloader_num_workers : int
        Number of loader worker processes.
    item_extra_tokens : Sequence(Hashable)
        Which element to use for sequence padding.
    shuffle_train : bool, default True
        If ``True``, reshuffles data at each epoch.
        Minimum length of user sequence. Cannot be less than 2.
    get_val_mask_func : Callable, default None
        Function to get validation mask.
    n_negatives : optional(int), default ``None``
        Number of negatives for BCE, gBCE and sampled_softmax losses.
    negative_sampler: optional(TransformerNegativeSamplerBase), default ``None``
        Negative sampler.
    get_val_mask_func_kwargs: optional(InitKwargs), default ``None``
        Additional arguments for the get_val_mask_func.
        Make sure all dict values have JSON serializable types.
    """
    #TODO

    train_session_max_len_addition: int = 1


    def _collate_fn_train(
        self,
        batch: List[BatchElement],
    ) -> Dict[str, torch.Tensor]:
        """
        Truncate each session from right to keep `session_max_len` items.
        Do left padding until `session_max_len` is reached.
        Split to `x`, `y`, and `yw`.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        t = np.zeros((batch_size, self.session_max_len+1))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        extras_train = {}
        for i, (ses, ses_weights, extras) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_max_len]
            if extras:
                t[i, -len(ses):] = extras["unix_ts"]
                len_to_pad = self.session_max_len+1 - len(ses)
                if len_to_pad > 0:
                    t[i, :len_to_pad] = t[i, len_to_pad]
            y[i, -len(ses) + 1 :] = ses[1:]  # ses: [session_len] -> y[i]: [session_max_len]
            yw[i, -len(ses) + 1 :] = ses_weights[1:]  # ses_weights: [session_len] -> yw[i]: [session_max_len]
        if self.extra_cols is not None:
            extras_train.update({"unix_ts":torch.LongTensor(t)})
        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        batch_dict.update(extras_train)
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size
            )
        return batch_dict

    def _collate_fn_val(self, batch: List[BatchElement]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        t = np.zeros((batch_size, self.session_max_len+1))
        y = np.zeros((batch_size, 1))  # Only leave-one-strategy is supported for losses
        yw = np.zeros((batch_size, 1))  # Only leave-one-strategy is supported for losses
        extras_val = {}
        for i, (ses, ses_weights, extras) in enumerate(batch):
            ses = ses[1:]
            ses_weights = ses_weights[1:]


            input_session = [ses[idx] for idx, weight in enumerate(ses_weights) if weight == 0]
            target_idx = [idx for idx, weight in enumerate(ses_weights) if weight != 0][0]
            if extras:
                extras["unix_ts"] = extras["unix_ts"][1:]
                t[i, -len(ses):] = extras["unix_ts"]
                len_to_pad = self.session_max_len+1 -len(ses)
                if len_to_pad > 0:
                    t[i, :len_to_pad] = t[i, len_to_pad]
            # ses: [session_len] -> x[i]: [session_max_len]
            x[i, -len(input_session) :] = input_session[-self.session_max_len :]
            y[i, -1:] = ses[target_idx]  # y[i]: [1]
            yw[i, -1:] = ses_weights[target_idx]  # yw[i]: [1]
        if self.extra_cols is not None:
            extras_val.update({"unix_ts":torch.LongTensor(t)})
        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        batch_dict.update(extras_val)
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size, session_len_limit=1
            )
        return batch_dict

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len))
        t = np.zeros((len(batch), self.session_max_len + 1))
        extras_recommend = {}
        for i, (ses, _, extras) in enumerate(batch):
            ses = ses [:-1] # drop dummy item
            x[i, -len(ses):] = ses[-self.session_max_len:]
            #x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_max_len]
            if extras:
                t[i, -len(ses)-1:] = extras["unix_ts"][-self.session_max_len-1:]
                len_to_pad = self.session_max_len- len(ses)
                if len_to_pad > 0:
                    t[i, :len_to_pad] = t[i, len_to_pad]
            #print(payloads)
        if self.extra_cols is not None:
            extras_recommend.update({"unix_ts":torch.LongTensor(t)})
        batch_dict = {"x": torch.LongTensor(x)}
        batch_dict.update(extras_recommend)
        return batch_dict

class RelativeAttention(torch.nn.Module):
    """
    Module calculate relative time and positional attention

    Parameters
    ----------
    max_seq_len : int.
        Maximum length of user sequence padded or truncated to
    attention_mode : List[str]
        Policy for calculating attention.
        Coulde be one of these "rel_pos_bias", "rel_ts_bias", "rel_pos_ts_bias"
    num_buckets : float
        Maximum  number of buckets model work with
    quantization_func: Callable
        Function that quantizes the space of timestamps differences into buckets
    """

    def __init__(
        self,
        max_seq_len: int,
        relative_time_attention:bool,
        relative_pos_attention: bool,
        num_buckets: int = 128,
        #TODO suggest DO not make private func of RelativeAttention
        quantization_func: tp.Optional[Callable[[torch.Tensor], torch.Tensor]] = lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.quantization_func  = quantization_func
        self.relative_time_attention = relative_time_attention
        self.relative_pos_attention = relative_pos_attention
        if relative_time_attention:
            self.time_weights = torch.nn.Parameter(
                torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
            )
        if relative_pos_attention:
            self.pos_weights = torch.nn.Parameter(
                torch.empty(2 * (max_seq_len+1) - 1).normal_(mean=0, std=0.02),
            )
    def forward_time_attention(self,all_timestamps: torch.Tensor)->torch.Tensor:
        N = self.max_seq_len +1  # N+1, 1 for target item time
        B = all_timestamps.size(0)
        extended_timestamps = torch.cat([all_timestamps, all_timestamps[:, N - 1: N]], dim=1)
        early_time_binding = extended_timestamps[:, 1:].unsqueeze(2) - extended_timestamps[:, :-1].unsqueeze(1)
        bucketed_timestamps = torch.clamp(
            self.quantization_func(early_time_binding),
            min=0,
            max=self.num_buckets,
        ).detach()
        rel_time_attention = torch.index_select(
            self.time_weights, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        rel_time_attention = rel_time_attention[:, :-1, :-1]
        # reducted target time (N -> self.max_seq_len)
        return rel_time_attention # (B,N, N)
    def forward_pos_attention(self) ->torch.Tensor:
        N = self.max_seq_len  # N+1, 1 for target item time
        t = F.pad(self.pos_weights[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2
        rel_pos_attention = t[:, :, r:-r]
        return rel_pos_attention # (1,N,N)
    def forward(
        self,
        batch: tp.Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Parametrs
        ---------
        all_timestamps: torch.Tensor (B, N+1)
            User sequence of timestamps + 1 target item timestamp
        Returns
        ---------
        torch.Tensor (B, N, N)
            Variate of sum relative pos/time attention
        """
        B = batch["x"].size(0)
        rel_attn = torch.zeros((B, self.max_seq_len, self.max_seq_len)).to(batch["x"].device)
        if self.relative_time_attention:
            rel_attn += self.forward_time_attention(batch["unix_ts"])
        if self.relative_pos_attention:
            rel_attn+= self.forward_pos_attention()
        return rel_attn

class STU(nn.Module):
    """
    HSTU author's encoder block architecture rewritten from jagged tensor to dense

    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    linear_hidden_dim : int
        U, V size.
    attention_dim : int
        Q, K size.
    attn_dropout_rate : float
        Probability of a attention unit to be zeroed.
    session_max_len : int
        Maximum length of user sequence padded or truncated to
    attention_mode : str
        Policy
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        linear_hidden_dim:int,
        attention_dim: int,
        session_max_len: int,
        relative_time_attention: bool,
        relative_pos_attention: bool,
        attn_dropout_rate: float = 0.1,
        dropout_rate: float = 0.2,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.rel_attn = RelativeAttention(
                            max_seq_len=session_max_len,
                            relative_time_attention = relative_time_attention,
                            relative_pos_attention = relative_pos_attention,
        )
        self.n_factors =  n_factors
        self.n_heads = n_heads
        self.linear_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.dropout_rate  = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.max_seq_len = session_max_len
        self.eps = epsilon
        self.uvqk_proj: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    n_factors,
                    self.linear_dim * 2 * n_heads
                    + attention_dim * n_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        self.output_mlp = torch.nn.Linear(
            in_features=linear_hidden_dim * n_heads,
            out_features=self.n_factors,
        )
        # TODO оно надо вообще запоминать
        self.relative_time_attention = relative_time_attention
        #torch.nn.init.xavier_uniform_(self.output_mlp.weight)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self.n_factors], eps=self.eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self.linear_dim * self.n_heads], eps=self.eps
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: tp.Optional[Dict[str, torch.Tensor]],
        attn_mask: torch.Tensor,
        timeline_mask: torch.Tensor,
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.
        attn_mask : torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `attn_mask`.
        key_padding_mask : torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `key_padding_mask`.


        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        B, N, _ = x.shape
        normed_x = self._norm_input(x) * timeline_mask  # prevent null emb convert to (b,b, ,,, b)
        general_trasform = torch.matmul(normed_x, self.uvqk_proj)
        batched_mm_output = F.silu(general_trasform) * timeline_mask
        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self.linear_dim * self.n_heads,  #
                self.linear_dim * self.n_heads,
                self.attention_dim * self.n_heads,
                self.attention_dim * self.n_heads,
            ],
            dim=-1,
        )
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q.view(B, N, self.n_heads, self.attention_dim),
            k.view(B, N, self.n_heads, self.attention_dim),
        ) #(B, n_head, N, N), attention on Q, K
        qk_attn = qk_attn + self.rel_attn(batch).unsqueeze(1) #(B, N, N).unsqueeze(1) -> (B,1,N,N) broadcast to n_heaad
        qk_attn = F.silu(qk_attn) / N

        time_line_mask_reducted = timeline_mask.squeeze(-1)
        time_line_mask_fix = time_line_mask_reducted.unsqueeze(1) * timeline_mask

        qk_attn = qk_attn * attn_mask.unsqueeze(0).unsqueeze(0) * time_line_mask_fix.unsqueeze(1)

        attn_output = torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                v.reshape(B, N, self.n_heads, self.linear_dim),
        ).reshape(B, N, self.n_heads * self.linear_dim)

        o_input = u * self._norm_attn_output(attn_output) * timeline_mask

        new_outputs = (
            self.output_mlp(
                F.dropout(
                    o_input,
                    p=self.dropout_rate,
                    training=self.training,
                )
            )
            + x
        )

        return new_outputs


class STULayers(TransformerLayersBase):
    """
    SASRec transformer blocks.

    Parameters
    ----------
    n_blocks : int
        Number of transformer blocks.
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        linear_hidden_dim: int,
        attention_dim: int,
        session_max_len: int,
        relative_time_attention: bool,
        relative_pos_attention: bool,
        attn_dropout_rate: float = 0.1,
        dropout_rate: float = 0.2,
        epsilon: float = 1e-6,
        **kwargs: tp.Any,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.stu_blocks = nn.ModuleList(
            [
                STU(
                    n_factors=n_factors,
                    n_heads=n_heads,
                    dropout_rate=dropout_rate,
                    linear_hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    relative_time_attention = relative_time_attention,
                    relative_pos_attention = relative_pos_attention,
                    attn_dropout_rate=attn_dropout_rate,
                    session_max_len=session_max_len,
                    epsilon=epsilon,
                )
                for _ in range(self.n_blocks)
            ]
        )

    def forward(
        self,
        seqs: torch.Tensor,
        batch: tp.Optional[Dict[str, torch.Tensor]],
        timeline_mask: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor],
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through transformer blocks.

        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.
        timeline_mask : torch.Tensor
            Mask indicating padding elements.
        attn_mask : torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `attn_mask`.
        key_padding_mask : torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `key_padding_mask`.


        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        # scaling + encoders
        # TODO attn_mask convert
        for i in range(self.n_blocks):
            seqs *= timeline_mask  # [batch_size, session_max_len, n_factors]
            seqs = self.stu_blocks[i](seqs, batch, attn_mask, timeline_mask, key_padding_mask)
        seqs *= timeline_mask
        return seqs / torch.clamp(
            torch.linalg.norm(seqs, ord=None, dim=-1, keepdim=True),
            min=1e-6,
        )


class HSTUModelConfig(TransformerModelConfig):
    """HSTU model config."""

    data_preparator_type: TransformerDataPreparatorType = HSTUDataPreparator
    transformer_layers_type: TransformerLayersType = STULayers
    use_causal_attn: bool = True
    dqk: int = 0,
    dvu: int = 0,
    relative_time_attention: bool = True,
    relative_pos_attention: bool = True,




class HSTUModel(TransformerModelBase[HSTUModelConfig]):
    """
    SASRec model: transformer-based sequential model with unidirectional attention mechanism and
    "Shifted Sequence" training objective.
    Our implementation covers multiple loss functions and a variable number of negatives for them.

    References
    ----------
    Transformers tutorial: https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_tutorial.html
    Advanced training guide:
    https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_advanced_training_guide.html
    Public benchmark: https://github.com/blondered/bert4rec_repro
    Original SASRec paper: https://arxiv.org/abs/1808.09781
    gBCE loss and gSASRec paper: https://arxiv.org/pdf/2308.07192

    Parameters
    ----------
    n_blocks : int, default 2
        Number of transformer blocks.
    n_heads : int, default 4
        Number of attention heads.
    n_factors : int, default 256
        Latent embeddings size.
    dropout_rate : float, default 0.2
        Probability of a hidden unit to be zeroed.
    session_max_len : int, default 100
        Maximum length of user sequence.
    train_min_user_interactions : int, default 2
        Minimum number of interactions user should have to be used for training. Should be greater
        than 1.
    loss : {"softmax", "BCE", "gBCE", "sampled_softmax"}, default "softmax"
        Loss function.
    n_negatives : int, default 1
        Number of negatives for BCE, gBCE and sampled_softmax losses.
    gbce_t : float, default 0.2
        Calibration parameter for gBCE loss.
    lr : float, default 0.001
        Learning rate.
    batch_size : int, default 128
        How many samples per batch to load.
    epochs : int, default 3
        Exact number of training epochs.
        Will be omitted if `get_trainer_func` is specified.
    deterministic : bool, default ``False``
        `deterministic` flag passed to lightning trainer during initialization.
        Use `pytorch_lightning.seed_everything` together with this parameter to fix the random seed.
        Will be omitted if `get_trainer_func` is specified.
    verbose : int, default 0
        Verbosity level.
        Enables progress bar, model summary and logging in default lightning trainer when set to a
        positive integer.
        Will be omitted if `get_trainer_func` is specified.
    dataloader_num_workers : int, default 0
        Number of loader worker processes.
    use_pos_emb : bool, default ``True``
        If ``True``, learnable positional encoding will be added to session item embeddings.
    use_key_padding_mask : bool, default ``False``
        If ``True``, key_padding_mask will be added in Multi-head Attention.
    use_causal_attn : bool, default ``True``
        If ``True``, causal mask will be added as attn_mask in Multi-head Attention. Please note that default
        SASRec training task ("Shifted Sequence") does not work without causal masking. Set this
        parameter to ``False`` only when you change the training task with custom
        `data_preparator_type` or if you are absolutely sure of what you are doing.
    item_net_block_types : sequence of `type(ItemNetBase)`, default `(IdEmbeddingsItemNet, CatFeaturesItemNet)`
        Type of network returning item embeddings.
        (IdEmbeddingsItemNet,) - item embeddings based on ids.
        (CatFeaturesItemNet,) - item embeddings based on categorical features.
        (IdEmbeddingsItemNet, CatFeaturesItemNet) - item embeddings based on ids and categorical features.
    item_net_constructor_type : type(ItemNetConstructorBase), default `SumOfEmbeddingsConstructor`
        Type of item net blocks aggregation constructor.
    pos_encoding_type : type(PositionalEncodingBase), default `LearnableInversePositionalEncoding`
        Type of positional encoding.
    transformer_layers_type : type(TransformerLayersBase), default `SasRecTransformerLayers`
        Type of transformer layers architecture.
    data_preparator_type : type(TransformerDataPreparatorBase), default `SasRecDataPreparator`
        Type of data preparator used for dataset processing and dataloader creation.
    lightning_module_type : type(TransformerLightningModuleBase), default `TransformerLightningModule`
        Type of lightning module defining training procedure.
    negative_sampler_type: type(TransformerNegativeSamplerBase), default `CatalogUniformSampler`
        Type of negative sampler.
    similarity_module_type : type(SimilarityModuleBase), default `DistanceSimilarityModule`
        Type of similarity module.
    backbone_type : type(TransformerBackboneBase), default `TransformerTorchBackbone`
        Type of torch backbone.
    get_val_mask_func : Callable, default ``None``
        Function to get validation mask.
    get_trainer_func : Callable, default ``None``
        Function for get custom lightning trainer.
        If `get_trainer_func` is None, default trainer will be created based on `epochs`,
        `deterministic` and `verbose` argument values. Model will be trained for the exact number of
        epochs. Checkpointing will be disabled.
        If you want to assign custom trainer after model is initialized, you can manually assign new
        value to model `_trainer` attribute.
    recommend_batch_size : int, default 256
        How many samples per batch to load during `recommend`.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_batch_size` attribute.
    recommend_torch_device : {"cpu", "cuda", "cuda:0", ...}, default ``None``
        String representation for `torch.device` used for model inference.
        When set to ``None``, "cuda" will be used if it is available, "cpu" otherwise.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_torch_device` attribute.
    get_val_mask_func_kwargs: optional(InitKwargs), default ``None``
        Additional keyword arguments for the get_val_mask_func.
        Make sure all dict values have JSON serializable types.
    get_trainer_func_kwargs: optional(InitKwargs), default ``None``
        Additional keyword arguments for the get_trainer_func.
        Make sure all dict values have JSON serializable types.
    data_preparator_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `data_preparator_type` initialization.
        Make sure all dict values have JSON serializable types.
    transformer_layers_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `transformer_layers_type` initialization.
        Make sure all dict values have JSON serializable types.
    item_net_constructor_kwargs optional(dict), default ``None``
        Additional keyword arguments to pass during `item_net_constructor_type` initialization.
        Make sure all dict values have JSON serializable types.
    pos_encoding_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `pos_encoding_type` initialization.
        Make sure all dict values have JSON serializable types.
    lightning_module_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `lightning_module_type` initialization.
        Make sure all dict values have JSON serializable types.
    negative_sampler_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `negative_sampler_type` initialization.
        Make sure all dict values have JSON serializable types.
    similarity_module_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `similarity_module_type` initialization.
        Make sure all dict values have JSON serializable types.
    backbone_kwargs: optional(dict), default ``None``
        Additional keyword arguments to pass during `backbone_type` initialization.
        Make sure all dict values have JSON serializable types.
    """

    config_class = HSTUModelConfig

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 2,
        n_heads: int = 4,
        n_factors: int = 256,
        dropout_rate: float = 0.2,
        session_max_len: int = 100,
        train_min_user_interactions: int = 2,
        loss: str = "softmax",
        n_negatives: int = 1,
        gbce_t: float = 0.2,
        lr: float = 0.001,
        batch_size: int = 128,
        epochs: int = 3,
        deterministic: bool = False,
        verbose: int = 0,
        dataloader_num_workers: int = 0,
        use_pos_emb: bool = True,
        use_key_padding_mask: bool = False,
        use_causal_attn: bool = True,
        dqk: int =  0,
        dvu: int = 0,
        relative_time_attention: bool = True,
        relative_pos_attention: bool = True,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        item_net_constructor_type: tp.Type[ItemNetConstructorBase] = SumOfEmbeddingsConstructor,
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = STULayers,
        data_preparator_type: tp.Type[TransformerDataPreparatorBase] = HSTUDataPreparator,
        lightning_module_type: tp.Type[TransformerLightningModuleBase] = TransformerLightningModule,
        negative_sampler_type: tp.Type[TransformerNegativeSamplerBase] = CatalogUniformSampler,
        similarity_module_type: tp.Type[SimilarityModuleBase] = DistanceSimilarityModule,
        backbone_type: tp.Type[TransformerBackboneBase] = HSTUTorchBackbone,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        get_trainer_func: tp.Optional[TrainerCallable] = None,
        get_val_mask_func_kwargs: tp.Optional[InitKwargs] = None,
        get_trainer_func_kwargs: tp.Optional[InitKwargs] = None,
        recommend_batch_size: int = 256,
        recommend_torch_device: tp.Optional[str] = None,
        recommend_use_torch_ranking: bool = True,
        recommend_n_threads: int = 0,
        data_preparator_kwargs: tp.Optional[InitKwargs] = None,
        transformer_layers_kwargs: tp.Optional[InitKwargs] = None,
        item_net_constructor_kwargs: tp.Optional[InitKwargs] = None,
        pos_encoding_kwargs: tp.Optional[InitKwargs] = None,
        lightning_module_kwargs: tp.Optional[InitKwargs] = None,
        negative_sampler_kwargs: tp.Optional[InitKwargs] = None,
        similarity_module_kwargs: tp.Optional[InitKwargs] = None,
        backbone_kwargs: tp.Optional[InitKwargs] = None,
    ):
        self.relative_time_attention = relative_time_attention
        self.relative_pos_attention = relative_pos_attention
        head_dim = int(n_factors / n_heads)
        self.dqk = dqk or head_dim
        self.dvu = dvu or head_dim

        if relative_time_attention:
            self._require_recommend_context = True
            data_preparator_kwargs = {
                "extra_cols": ["unix_ts"],
                "add_unix_ts": True
            }

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
            dataloader_num_workers=dataloader_num_workers,
            batch_size=batch_size,
            loss=loss,
            n_negatives=n_negatives,
            gbce_t=gbce_t,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
            deterministic=deterministic,
            recommend_batch_size=recommend_batch_size,
            recommend_torch_device=recommend_torch_device,
            recommend_n_threads=recommend_n_threads,
            recommend_use_torch_ranking=recommend_use_torch_ranking,
            train_min_user_interactions=train_min_user_interactions,
            similarity_module_type=similarity_module_type,
            item_net_block_types=item_net_block_types,
            item_net_constructor_type=item_net_constructor_type,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
            negative_sampler_type=negative_sampler_type,
            backbone_type=backbone_type,
            get_val_mask_func=get_val_mask_func,
            get_trainer_func=get_trainer_func,
            get_val_mask_func_kwargs=get_val_mask_func_kwargs,
            get_trainer_func_kwargs=get_trainer_func_kwargs,
            data_preparator_kwargs=data_preparator_kwargs,
            transformer_layers_kwargs=transformer_layers_kwargs,
            item_net_constructor_kwargs=item_net_constructor_kwargs,
            pos_encoding_kwargs=pos_encoding_kwargs,
            lightning_module_kwargs=lightning_module_kwargs,
            negative_sampler_kwargs=negative_sampler_kwargs,
            similarity_module_kwargs=similarity_module_kwargs,
            backbone_kwargs=backbone_kwargs,
        )

    def _init_transformer_layers(self) -> STULayers:
        return self.transformer_layers_type(
            n_blocks=self.n_blocks,
            n_factors=self.n_factors,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            session_max_len = self.session_max_len,
            attention_dim=self.dqk,
            linear_hidden_dim = self.dvu,
            relative_time_attention = self.relative_time_attention,
            relative_pos_attention = self.relative_pos_attention,
            **self._get_kwargs(self.transformer_layers_kwargs),
        )
    def preproc_recommend_context(
        self,
        recommend_dataset: Dataset,
        context: pd.DataFrame
    ) -> tp.Dict[str, torch.Tensor]:
        return self._preproc_recommend_context(recommend_dataset, context)

