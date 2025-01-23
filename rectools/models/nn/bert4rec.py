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
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator

from .item_net import CatFeaturesItemNet, IdEmbeddingsItemNet, ItemNetBase
from .transformer_base import (
    PADDING_VALUE,
    SessionEncoderLightningModule,
    SessionEncoderLightningModuleBase,
    TransformerModelBase,
)
from .transformer_data_preparator import SessionEncoderDataPreparatorBase
from .transformer_net_blocks import (
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    PreLNTransformerLayers,
    TransformerLayersBase,
)

MASKING_VALUE = "MASK"


class BERT4RecDataPreparator(SessionEncoderDataPreparatorBase):
    """Data Preparator for BERT4RecModel."""

    def __init__(
        self,
        session_max_len: int,
        n_negatives: tp.Optional[int],
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        mask_prob: float,
        item_extra_tokens: tp.Sequence[tp.Hashable],
        shuffle_train: bool = True,
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            item_extra_tokens=item_extra_tokens,
            shuffle_train=shuffle_train,
        )
        self.mask_prob = mask_prob

    def _mask_session(self, ses: List[int]) -> Tuple[List[int], List[int]]:
        masked_session = ses.copy()
        target = ses.copy()
        random_probs = np.random.rand(len(ses))
        for j in range(len(ses)):
            if random_probs[j] < self.mask_prob:
                random_probs[j] /= self.mask_prob
                if random_probs[j] < 0.8:
                    masked_session[j] = self.extra_token_ids[MASKING_VALUE]
                elif random_probs[j] < 0.9:
                    masked_session[j] = np.random.randint(low=self.n_item_extra_tokens, high=self.item_id_map.size)
            else:
                target[j] = 0
        return masked_session, target

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Mask session elements to receive `x`.
        Get target by replacing session elements with a MASK token with probability `mask_prob`.
        Truncate each session and target from right to keep (`session_max_len` + 1) last items.
        Do left padding until  (session_max_len + 1) is reached.
        If `n_negatives` is not None, generate negative items from uniform distribution.
        (session_max_len + 1) is used here to keep the logic that session_max_len is maximum sessions length
        that model will process during inference. But one extra "MASK" token will be added for making predictions.
        Thus we need to see (session_max_len + 1) elements during training.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len + 1))
        y = np.zeros((batch_size, self.session_max_len + 1))
        yw = np.zeros((batch_size, self.session_max_len + 1))
        for i, (ses, ses_weights) in enumerate(batch):
            masked_session, target = self._mask_session(ses)
            x[i, -len(ses) :] = masked_session  # ses: [session_len] -> x[i]: [session_max_len + 1]
            y[i, -len(ses) :] = target  # ses: [session_len] -> y[i]: [session_max_len + 1]
            yw[i, -len(ses) :] = ses_weights  # ses_weights: [session_len] -> yw[i]: [session_max_len + 1]

        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, self.session_max_len + 1, self.n_negatives),
            )  # [batch_size, session_max_len, n_negatives]
            batch_dict["negatives"] = negatives
        return batch_dict

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len + 1))
        for i, (ses, _) in enumerate(batch):
            session = ses.copy()
            session = session + [self.extra_token_ids[MASKING_VALUE]]
            x[i, -len(ses) - 1 :] = session[-self.session_max_len - 1 :]
        return {"x": torch.LongTensor(x)}


class BERT4RecModel(TransformerModelBase):
    """
    BERT4Rec model.

    n_blocks : int, default 1
        Number of transformer blocks.
    n_heads : int, default 1
        Number of attention heads.
    n_factors : int, default 128
        Latent embeddings size.
    use_pos_emb : bool, default ``True``
        If ``True``, learnable positional encoding will be added to session item embeddings.
    use_causal_attn : bool, default ``False``
        If ``True``, causal mask will be added as attn_mask in Multi-head Attention. Please note that default
        BERT4Rec training task (MLM) does not match well with causal masking. Set this parameter to
        ``True`` only when you change the training task with custom `data_preparator_type` or if you
        are absolutely sure of what you are doing.
    use_key_padding_mask : bool, default ``False``
        If ``True``, key_padding_mask will be added in Multi-head Attention.
    dropout_rate : float, default 0.2
        Probability of a hidden unit to be zeroed.
    session_max_len : int, default 32
        Maximum length of user sequence that model will accept during inference.
    train_min_user_interactions : int, default 2
        Minimum number of interactions user should have to be used for training. Should be greater than 1.
    mask_prob : float, default 0.15
        Probability of masking an item in interactions sequence.
    dataloader_num_workers : int, default 0
        Number of loader worker processes.
    batch_size : int, default 128
        How many samples per batch to load.
    loss : {"softmax", "BCE", "gBCE"}, default "softmax"
        Loss function.
    n_negatives : int, default 1
        Number of negatives for BCE and gBCE losses.
    gbce_t : float, default 0.2
        Calibration parameter for gBCE loss.
    lr : float, default 0.01
        Learning rate.
    epochs : int, default 3
        Number of training epochs.
    verbose : int, default 0
        Verbosity level.
    deterministic : bool, default ``False``
        If ``True``, set deterministic algorithms for PyTorch operations.
        Use `pytorch_lightning.seed_everything` together with this parameter to fix the random state.
    recommend_device : {"cpu", "gpu", "tpu", "hpu", "mps", "auto"} or Accelerator, default "auto"
        Device for recommend. Used at predict_step of lightning module.
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
    trainer : Trainer, optional, default ``None``
        Which trainer to use for training.
        If trainer is None, default pytorch_lightning Trainer is created.
    item_net_block_types : sequence of `type(ItemNetBase)`, default `(IdEmbeddingsItemNet, CatFeaturesItemNet)`
        Type of network returning item embeddings.
        (IdEmbeddingsItemNet,) - item embeddings based on ids.
        (CatFeaturesItemNet,) - item embeddings based on categorical features.
        (IdEmbeddingsItemNet, CatFeaturesItemNet) - item embeddings based on ids and categorical features.
    pos_encoding_type : type(PositionalEncodingBase), default `LearnableInversePositionalEncoding`
        Type of positional encoding.
    transformer_layers_type : type(TransformerLayersBase), default `PreLNTransformerLayers`
        Type of transformer layers architecture.
    data_preparator_type : type(SessionEncoderDataPreparatorBase), default `BERT4RecDataPreparator`
        Type of data preparator used for dataset processing and dataloader creation.
    lightning_module_type : type(SessionEncoderLightningModuleBase), default `SessionEncoderLightningModule`
        Type of lightning module defining training procedure.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        use_causal_attn: bool = False,
        use_key_padding_mask: bool = True,
        dropout_rate: float = 0.2,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_device: Union[str, Accelerator] = "auto",
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        session_max_len: int = 32,
        n_negatives: int = 1,
        batch_size: int = 128,
        loss: str = "softmax",
        gbce_t: float = 0.2,
        lr: float = 0.01,
        dataloader_num_workers: int = 0,
        train_min_user_interactions: int = 2,
        mask_prob: float = 0.15,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = PreLNTransformerLayers,
        data_preparator_type: tp.Type[BERT4RecDataPreparator] = BERT4RecDataPreparator,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
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
            epochs=epochs,
            verbose=verbose,
            deterministic=deterministic,
            recommend_device=recommend_device,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
            loss=loss,
            gbce_t=gbce_t,
            lr=lr,
            session_max_len=session_max_len + 1,
            trainer=trainer,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
        )
        self.data_preparator = data_preparator_type(
            session_max_len=session_max_len,
            n_negatives=n_negatives if loss != "softmax" else None,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            item_extra_tokens=(PADDING_VALUE, MASKING_VALUE),
            mask_prob=mask_prob,
        )
