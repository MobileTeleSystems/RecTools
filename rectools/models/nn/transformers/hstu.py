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

check_split_train_dataset = True
check_split_val_dataset = True
class HSTUDataPreparator(TransformerDataPreparatorBase):
    """Data preparator for SASRecModel.
    TODO
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
    train_min_user_interactions : int, default 2
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

    train_session_max_len_addition: int = 1

    def append_list_to_file(self,filename, data, separator=", "):
        """
        Дописывает элементы списка `data` в файл `filename`, объединяя их через `separator` в одной строке.

        Параметры:
            filename (str): Имя файла.
            data (list): Список строк или объектов, которые можно преобразовать в строки.
            separator (str): Разделитель между элементами (например, пробел, запятая и т.д.).
        """
        with open(filename, "a", encoding="utf-8") as file:
            # Преобразуем элементы в строки и объединяем через разделитель
            file.write(separator.join(map(str, data)) + "\n")
    def datetime64_to_unixtime(self,dt_list: list) -> list:
        epoch = np.datetime64("1970-01-01T00:00:00")
        res = [int((dt - epoch) / np.timedelta64(1, "s")) for dt in dt_list]
        return res

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
        payloads_train = {}
        for i, (ses, ses_weights, payloads) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_max_len]
            t[i, -len(ses):] = self.datetime64_to_unixtime(payloads[Columns.Datetime])
            len_to_pad = self.session_max_len+1 - len(ses)
            if len_to_pad > 0:
                t[i, :len_to_pad] = t[i, len_to_pad]
            y[i, -len(ses) + 1 :] = ses[1:]  # ses: [session_len] -> y[i]: [session_max_len]
            yw[i, -len(ses) + 1 :] = ses_weights[1:]  # ses_weights: [session_len] -> yw[i]: [session_max_len]
        payloads_train.update({Columns.Datetime:torch.LongTensor(t)})
        batch_dict = {"x": torch.LongTensor(x),"payloads": payloads_train , "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size
            )
        return batch_dict
    """
    [324, 849, 515, 775, 2137], 376 #train collate fn
          [_______________________]
    324, 849, 515, 775, 2137 - X,
    849, 515, 775, 2137, 376 - Y

    324, [849, 515, 775, 2137, 376], 890  #val collate fn
                                    [___]
    лоо некст должно быть
    849, 515, 775, 2137, 376 - X
    515, 775, 2137, 376, [890] - один таргет на валидации
    """
    def _collate_fn_val(self, batch: List[BatchElement]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        t = np.zeros((batch_size, self.session_max_len+1))
        y = np.zeros((batch_size, 1))  # Only leave-one-strategy is supported for losses
        yw = np.zeros((batch_size, 1))  # Only leave-one-strategy is supported for losses
        payloads_val = {}
        for i, (ses, ses_weights, payloads) in enumerate(batch):
            ses = ses[1:]
            ses_weights = ses_weights[1:]
            payloads[Columns.Datetime] = payloads[Columns.Datetime][1:]

            input_session = [ses[idx] for idx, weight in enumerate(ses_weights) if weight == 0]
            target_idx = [idx for idx, weight in enumerate(ses_weights) if weight != 0][0]

            t[i, -len(ses):] = self.datetime64_to_unixtime(payloads[Columns.Datetime])
            len_to_pad = self.session_max_len+1 -len(ses)
            if len_to_pad > 0:
                t[i, :len_to_pad] = t[i, len_to_pad]
            # ses: [session_len] -> x[i]: [session_max_len]
            x[i, -len(input_session) :] = input_session[-self.session_max_len :]
            y[i, -1:] = ses[target_idx]  # y[i]: [1]
            yw[i, -1:] = ses_weights[target_idx]  # yw[i]: [1]
        payloads_val.update({Columns.Datetime: torch.LongTensor(t)})
        batch_dict = {"x": torch.LongTensor(x),"payloads": payloads_val, "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size, session_len_limit=1
            )
        return batch_dict

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len))
        payloads_recommend = {}
        for i, (ses, _) in enumerate(batch):
            x[i, -len(ses) :] = ses[-self.session_max_len :]
        payloads_recommend.update({Columns.Datetime: None})
        return {"x": torch.LongTensor(x), "payloads": payloads_recommend}

class RelativeBucketedTimeAndPositionBasedBias(torch.nn.Module):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        attention_mode:str,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )
        self._attention_mode = attention_mode

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2


        #здесь кажется ошибка у них, дата лик
        #первый токен уже знает о времени второго

            # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()

        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        rel_pos_bias=  rel_pos_bias[:,:-1,:-1] # (1,N-1, N-1) # last one is supervision time
        rel_ts_bias = rel_ts_bias[:,:-1,:-1] # (B, N-1, N-1)
        if self._attention_mode == "rel_pos_bias":
            return rel_pos_bias
        elif self._attention_mode == "rel_ts_bias":
            return rel_ts_bias
        elif self._attention_mode == "rel_pos_ts_bias":
            return rel_pos_bias + rel_ts_bias


class STU(nn.Module):
    """
    TODO

    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        linear_hidden_dim:int,
        attention_dim: int,
        attn_dropout_ratio: float,
        session_max_len: int,
        attention_mode :str,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self._rel_attn_bias = RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=session_max_len+1, # add supervision time
                            attention_mode = attention_mode,
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
        )
        self._embedding_dim: int = n_factors
        self._num_heads = n_heads
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_rate
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._eps: float = epsilon
        self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    n_factors,
                    self._linear_dim * 2 * n_heads
                    + attention_dim * n_heads * 2,
                )
            ).normal_(mean=0, std=0.02),
        )
        in_dim = self._uvqk.shape[0]
        out_dim = self._uvqk.shape[1]
        #self.linear_layer = nn.Linear(in_dim, out_dim, bias=False)
        #self.linear_layer.weight.data = self._uvqk.t()
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * n_heads,
            out_features=self._embedding_dim,
        )
        self._attention_mode = attention_mode
        torch.nn.init.xavier_uniform_(self._o.weight)

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(
        self,
        x: torch.Tensor,
        payloads: tp.Optional[Dict[str, torch.Tensor]],
        attn_mask: torch.Tensor,
        timeline_mask: torch.Tensor,
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        TODO
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
        normed_x = self._norm_input(x)*timeline_mask
        general_trasform = torch.matmul(normed_x, self._uvqk)
        batched_mm_output = F.silu(general_trasform) * timeline_mask
        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads,
            ],
            dim=-1,
        )
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q.view(B, N, self._num_heads, self._attention_dim),
            k.view(B, N, self._num_heads, self._attention_dim),
        )
        time_line_mask_reducted = timeline_mask.squeeze(-1)
        time_line_mask_fix = time_line_mask_reducted.unsqueeze(1) * timeline_mask

        if payloads[Columns.Datetime] is not None:
            qk_attn = qk_attn + self._rel_attn_bias(payloads[Columns.Datetime]).unsqueeze(1)
        qk_attn = F.silu(qk_attn) / N
        qk_attn = qk_attn * attn_mask.unsqueeze(0).unsqueeze(0) *time_line_mask_fix.unsqueeze(1)
        attn_output = torch.einsum(
                "bhnm,bmhd->bnhd",
                qk_attn,
                v.reshape(B, N, self._num_heads, self._linear_dim),
        ).reshape(B, N, self._num_heads * self._linear_dim)

        o_input = u * self._norm_attn_output(attn_output) * timeline_mask

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
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
        dropout_rate: float,
        linear_hidden_dim: int,
        attention_dim: int,
        attn_dropout_ratio: float,
        session_max_len: int,
        attention_mode: str,
        epsilon: float = 1e-6,
        **kwargs: tp.Any,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.stu_blocks = nn.ModuleList(
            [
                STU(
                    n_factors,
                    n_heads,
                    dropout_rate,
                    linear_hidden_dim,
                    attention_dim,
                    attn_dropout_ratio,
                    session_max_len,
                    attention_mode,
                    epsilon
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.last_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)

    def forward(
        self,
        seqs: torch.Tensor,
        payloads: tp.Optional[Dict[str, torch.Tensor]],
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
        for i in range(self.n_blocks):
            seqs *= timeline_mask  # [batch_size, session_max_len, n_factors]
            seqs = self.stu_blocks[i](seqs, payloads, attn_mask, timeline_mask, key_padding_mask)
        seqs *= timeline_mask
        return seqs / torch.clamp(
            torch.linalg.norm(seqs, ord=None, dim=-1, keepdim=True),
            min=1e-6,
        )
        return seqs


class HSTUModelConfig(TransformerModelConfig):
    """SASRecModel config."""

    data_preparator_type: TransformerDataPreparatorType = HSTUDataPreparator
    transformer_layers_type: TransformerLayersType = STULayers
    use_causal_attn: bool = True


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x

class LearnablePositionalEncoding(PositionalEncodingBase):
    def __init__(
        self,
        use_pos_emb: bool,
        max_sequence_len: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self._embedding_dim: int = embedding_dim
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len,
            self._embedding_dim,
        )
        #TODO
        plug_drop_rate = 0.2
        self._dropout_rate: float = plug_drop_rate
        self._emb_dropout = torch.nn.Dropout(p=plug_drop_rate)
        self.reset_state()


    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(self, sessions_embeddings: torch.Tensor) -> torch.Tensor:
        B, N, D = sessions_embeddings.size()

        # Найдём маску реальных токенов (не padding): True для ненулевых
        is_real = (sessions_embeddings.abs().sum(dim=-1) != 0)  # shape: [B, N]

        # Найдём длину padding'а слева (сколько нулей перед первым ненулевым)
        first_nonzero_idx = is_real.int().argmax(dim=1)  # shape: [B], индекс первого ненулевого

        # Создадим матрицу позиций, где позиции начинаются с 0 после padding'а
        positions = torch.arange(N, device=sessions_embeddings.device).expand(B, N)
        shifted_positions = (positions - first_nonzero_idx.view(-1, 1))  # вычитаем сдвиг
        shifted_positions = shifted_positions.masked_fill(~is_real, 0)  # заменяем отрицательные/ненастоящие на 0

        # Применяем positional embeddings
        pos_embeddings = self._pos_emb(shifted_positions)

        # Теперь собираем финальное представление
        user_embeddings = sessions_embeddings * (D ** 0.5) + pos_embeddings
        user_embeddings = self._emb_dropout(user_embeddings)

        return user_embeddings

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

