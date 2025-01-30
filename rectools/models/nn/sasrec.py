#  Copyright 2024 MTS (Mobile Telesystems)
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
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .item_net import CatFeaturesItemNet, IdEmbeddingsItemNet, ItemNetBase
from .transformer_base import (
    TrainerCallable,
    TransformerDataPreparatorType,
    TransformerLayersType,
    TransformerLightningModule,
    TransformerLightningModuleBase,
    TransformerModelBase,
    TransformerModelConfig,
    ValMaskCallable,
)
from .transformer_data_preparator import TransformerDataPreparatorBase
from .transformer_net_blocks import (
    LearnableInversePositionalEncoding,
    PointWiseFeedForward,
    PositionalEncodingBase,
    TransformerLayersBase,
)


class SASRecDataPreparator(TransformerDataPreparatorBase):
    """Data preparator for SASRecModel."""

    train_session_max_len_addition: int = 1

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Truncate each session from right to keep `session_max_len` items.
        Do left padding until `session_max_len` is reached.
        Split to `x`, `y`, and `yw`.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        for i, (ses, ses_weights) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_max_len]
            y[i, -len(ses) + 1 :] = ses[1:]  # ses: [session_len] -> y[i]: [session_max_len]
            yw[i, -len(ses) + 1 :] = ses_weights[1:]  # ses_weights: [session_len] -> yw[i]: [session_max_len]

        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, self.session_max_len, self.n_negatives),
            )  # [batch_size, session_max_len, n_negatives]
            batch_dict["negatives"] = negatives
        return batch_dict

    def _collate_fn_val(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, 1))  # Only leave-one-strategy is supported for losses
        yw = np.zeros((batch_size, 1))  # Only leave-one-strategy is supported for losses
        for i, (ses, ses_weights) in enumerate(batch):
            input_session = [ses[idx] for idx, weight in enumerate(ses_weights) if weight == 0]

            # take only first target for leave-one-strategy
            target_idx = [idx for idx, weight in enumerate(ses_weights) if weight != 0][0]

            # ses: [session_len] -> x[i]: [session_max_len]
            x[i, -len(input_session) :] = input_session[-self.session_max_len :]
            y[i, -1:] = ses[target_idx]  # y[i]: [1]
            yw[i, -1:] = ses_weights[target_idx]  # yw[i]: [1]

        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, 1, self.n_negatives),
            )  # [batch_size, 1, n_negatives]
            batch_dict["negatives"] = negatives
        return batch_dict

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len))
        for i, (ses, _) in enumerate(batch):
            x[i, -len(ses) :] = ses[-self.session_max_len :]
        return {"x": torch.LongTensor(x)}


class SASRecTransformerLayers(TransformerLayersBase):
    """
    Exactly SASRec author's transformer blocks architecture but with pytorch Multi-Head Attention realisation.

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
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.multi_head_attn = nn.ModuleList(
            [torch.nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True) for _ in range(n_blocks)]
        )  # important: original architecture had another version of MHA
        self.q_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.ff_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.feed_forward = nn.ModuleList(
            [PointWiseFeedForward(n_factors, n_factors, dropout_rate, torch.nn.ReLU()) for _ in range(n_blocks)]
        )
        self.dropout = nn.ModuleList([torch.nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        self.last_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)

    def forward(
        self,
        seqs: torch.Tensor,
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
        seqs *= timeline_mask  # [batch_size, session_max_len, n_factors]
        for i in range(self.n_blocks):
            q = self.q_layer_norm[i](seqs)
            mha_output, _ = self.multi_head_attn[i](
                q, seqs, seqs, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
            )
            seqs = q + mha_output
            ff_input = self.ff_layer_norm[i](seqs)
            seqs = self.feed_forward[i](ff_input)
            seqs = self.dropout[i](seqs)
            seqs += ff_input
            seqs *= timeline_mask

        seqs = self.last_layernorm(seqs)

        return seqs


class SASRecModelConfig(TransformerModelConfig):
    """SASRecModel config."""

    data_preparator_type: TransformerDataPreparatorType = SASRecDataPreparator
    transformer_layers_type: TransformerLayersType = SASRecTransformerLayers
    use_causal_attn: bool = True


class SASRecModel(TransformerModelBase[SASRecModelConfig]):
    """
    SASRec model: transformer-based sequential model with unidirectional attention mechanism and
    "Shifted Sequence" training objective.
    Our implementation covers multiple loss functions and a variable number of negatives for them.

    See RecTools theory&practice tutorial on transformers:
        https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_tutorial.html
    See RecTools advanced training guide for transformers:
        https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_training_guide.html
    See original SASRec paper:
        https://arxiv.org/abs/1808.09781

    n_blocks : int, default 1
        Number of transformer blocks.
    n_heads : int, default 1
        Number of attention heads.
    n_factors : int, default 128
        Latent embeddings size.
    use_pos_emb : bool, default ``True``
        If ``True``, learnable positional encoding will be added to session item embeddings.
    use_causal_attn : bool, default ``True``
        If ``True``, causal mask will be added as attn_mask in Multi-head Attention. Please note that default
        SASRec training task ("Shifted Sequence") does not work without causal masking. Set this
        parameter to ``False`` only when you change the training task with custom
        `data_preparator_type` or if you are absolutely sure of what you are doing.
    use_key_padding_mask : bool, default ``False``
        If ``True``, key_padding_mask will be added in Multi-head Attention.
    dropout_rate : float, default 0.2
        Probability of a hidden unit to be zeroed.
    session_max_len : int, default 32
        Maximum length of user sequence.
    train_min_user_interactions : int, default 2
        Minimum number of interactions user should have to be used for training. Should be greater
        than 1.
    dataloader_num_workers : int, default 0
        Number of loader worker processes.
    batch_size : int, default 128
        How many samples per batch to load.
    loss : {"softmax", "BCE", "gBCE"}, default "softmax"
        Loss function.
        See https://arxiv.org/pdf/2308.07192 for details on gBCE loss.
    n_negatives : int, default 1
        Number of negatives for BCE and gBCE losses.
    gbce_t : float, default 0.2
        Calibration parameter for gBCE loss.
    lr : float, default 0.01
        Learning rate.
    epochs : int, default 3
        Exact number of training epochs.
        Will be omitted if `get_trainer_func` is specified.
    verbose : int, default 0
        Verbosity level.
        Enables progress bar, model summary and logging in default lightning trainer when set to a
        positive integer.
        Enables automatic lightning checkpointing when set to 100 or higher. This will save the most
        the most recent model to a single checkpoint after each epoch.
        Will be omitted if `get_trainer_func` is specified.
    deterministic : bool, default ``False``
        `deterministic` flag passed to lightning trainer during initialization.
        Use `pytorch_lightning.seed_everything` together with this parameter to fix the random seed.
        Will be omitted if `get_trainer_func` is specified.
    recommend_batch_size : int, default 256
        How many samples per batch to load during `recommend`.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_batch_size` attribute.
    recommend_accelerator : {"cpu", "gpu", "tpu", "hpu", "mps", "auto"}, default "auto"
        Accelerator type for `recommend`. Used at predict_step of lightning module.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_accelerator` attribute.
    recommend_devices : int | List[int], default 1
        Devices for `recommend`. Please note that multi-device inference is not supported!
        Do not specify more then one device. For ``gpu`` accelerator you can pass which device to
        use, e.g. ``[1]``.
        Used at predict_step of lightning module.
        Multi-device recommendations are not supported.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_device` attribute.
    recommend_n_threads : int, default 0
        Number of threads to use in ranker if GPU ranking is turned off or unavailable.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_n_threads` attribute.
    recommend_use_gpu_ranking : bool, default ``True``
        If ``True`` and HAS_CUDA ``True``, set use_gpu=True in ImplicitRanker.rank.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_use_gpu_ranking` attribute.
    item_net_block_types : sequence of `type(ItemNetBase)`, default `(IdEmbeddingsItemNet, CatFeaturesItemNet)`
        Type of network returning item embeddings.
        (IdEmbeddingsItemNet,) - item embeddings based on ids.
        (CatFeaturesItemNet,) - item embeddings based on categorical features.
        (IdEmbeddingsItemNet, CatFeaturesItemNet) - item embeddings based on ids and categorical features.
    pos_encoding_type : type(PositionalEncodingBase), default `LearnableInversePositionalEncoding`
        Type of positional encoding.
    transformer_layers_type : type(TransformerLayersBase), default `SasRecTransformerLayers`
        Type of transformer layers architecture.
    data_preparator_type : type(TransformerDataPreparatorBase), default `SasRecDataPreparator`
        Type of data preparator used for dataset processing and dataloader creation.
    lightning_module_type : type(TransformerLightningModuleBase), default `TransformerLightningModule`
        Type of lightning module defining training procedure.
    get_val_mask_func : Callable, default ``None``
        Function to get validation mask.
    get_trainer_func : Callable, default ``None``
        Function for get custom lightning trainer.
        If `get_trainer_func` is None, default trainer will be created based on `epochs`,
        `deterministic` and `verbose` argument values. Model will be trained for the exact number of
        epochs. Checkpointing will be disabled.
        If you want to assign custom trainer after model is initialized, you can manually assign new
        value to model `_trainer` attribute.
    """

    config_class = SASRecModelConfig

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 2,
        n_heads: int = 4,
        n_factors: int = 256,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        dropout_rate: float = 0.2,
        session_max_len: int = 100,
        dataloader_num_workers: int = 0,
        batch_size: int = 128,
        loss: str = "softmax",
        n_negatives: int = 1,
        gbce_t: float = 0.2,
        lr: float = 0.001,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_batch_size: int = 256,
        recommend_accelerator: str = "auto",
        recommend_devices: tp.Union[int, tp.List[int]] = 1,
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        train_min_user_interactions: int = 2,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = SASRecTransformerLayers,  # SASRec authors net
        data_preparator_type: tp.Type[TransformerDataPreparatorBase] = SASRecDataPreparator,
        lightning_module_type: tp.Type[TransformerLightningModuleBase] = TransformerLightningModule,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        get_trainer_func: tp.Optional[TrainerCallable] = None,
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
            recommend_accelerator=recommend_accelerator,
            recommend_devices=recommend_devices,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
            train_min_user_interactions=train_min_user_interactions,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
            get_val_mask_func=get_val_mask_func,
            get_trainer_func=get_trainer_func,
        )

    def _init_data_preparator(self) -> None:
        self.data_preparator = self.data_preparator_type(
            session_max_len=self.session_max_len,
            n_negatives=self.n_negatives if self.loss != "softmax" else None,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            get_val_mask_func=self.get_val_mask_func,
        )
