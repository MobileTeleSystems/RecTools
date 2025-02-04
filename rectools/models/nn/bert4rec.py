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
from collections.abc import Hashable
from typing import Dict, List, Tuple

import numpy as np
import torch

from .constants import MASKING_VALUE, PADDING_VALUE
from .item_net import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructorBase,
    SumOfEmbeddingsConstructor,
)
from .transformer_base import (
    TrainerCallable,
    TransformerDataPreparatorType,
    TransformerLightningModule,
    TransformerLightningModuleBase,
    TransformerModelBase,
    TransformerModelConfig,
    ValMaskCallable,
)
from .transformer_data_preparator import TransformerDataPreparatorBase
from .transformer_net_blocks import (
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    PreLNTransformerLayers,
    TransformerLayersBase,
)


class BERT4RecDataPreparator(TransformerDataPreparatorBase):
    """Data Preparator for BERT4RecModel."""

    train_session_max_len_addition: int = 0

    item_extra_tokens: tp.Sequence[Hashable] = (PADDING_VALUE, MASKING_VALUE)

    def __init__(
        self,
        session_max_len: int,
        n_negatives: tp.Optional[int],
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        mask_prob: float,
        shuffle_train: bool = True,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            shuffle_train=shuffle_train,
            get_val_mask_func=get_val_mask_func,
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
        Truncate each session and target from right to keep `session_max_len` last items.
        Do left padding until `session_max_len` is reached.
        If `n_negatives` is not None, generate negative items from uniform distribution.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        for i, (ses, ses_weights) in enumerate(batch):
            masked_session, target = self._mask_session(ses)
            x[i, -len(ses) :] = masked_session  # ses: [session_len] -> x[i]: [session_max_len]
            y[i, -len(ses) :] = target  # ses: [session_len] -> y[i]: [session_max_len]
            yw[i, -len(ses) :] = ses_weights  # ses_weights: [session_len] -> yw[i]: [session_max_len]

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
        y = np.zeros((batch_size, 1))  # until only leave-one-strategy
        yw = np.zeros((batch_size, 1))  # until only leave-one-strategy
        for i, (ses, ses_weights) in enumerate(batch):
            input_session = [ses[idx] for idx, weight in enumerate(ses_weights) if weight == 0]
            session = input_session.copy()

            # take only first target for leave-one-strategy
            session = session + [self.extra_token_ids[MASKING_VALUE]]
            target_idx = [idx for idx, weight in enumerate(ses_weights) if weight != 0][0]

            # ses: [session_len] -> x[i]: [session_max_len]
            x[i, -len(input_session) - 1 :] = session[-self.session_max_len :]
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
        """
        Right truncation, left padding to `session_max_len`
        During inference model will use (`session_max_len` - 1) interactions
        and one extra "MASK" token will be added for making predictions.
        """
        x = np.zeros((len(batch), self.session_max_len))
        for i, (ses, _) in enumerate(batch):
            session = ses.copy()
            session = session + [self.extra_token_ids[MASKING_VALUE]]
            x[i, -len(ses) - 1 :] = session[-self.session_max_len :]
        return {"x": torch.LongTensor(x)}


class BERT4RecModelConfig(TransformerModelConfig):
    """BERT4RecModel config."""

    data_preparator_type: TransformerDataPreparatorType = BERT4RecDataPreparator
    use_key_padding_mask: bool = True
    mask_prob: float = 0.15


class BERT4RecModel(TransformerModelBase[BERT4RecModelConfig]):
    """
    BERT4Rec model: transformer-based sequential model with bidirectional attention mechanism and
    "MLM" (masked item in user sequence) training objective.
    Our implementation covers multiple loss functions and a variable number of negatives for them.

    References
    ----------
    Transformers tutorial: https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_tutorial.html
    Advanced training guide:
    https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_advanced_training_guide.html
    Public benchmark: https://github.com/blondered/bert4rec_repro
    Original BERT4Rec paper: https://arxiv.org/abs/1904.06690
    gBCE loss paper: https://arxiv.org/pdf/2308.07192

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
    mask_prob : float, default 0.15
        Probability of masking an item in interactions sequence.
    session_max_len : int, default 100
        Maximum length of user sequence.
    train_min_user_interactions : int, default 2
        Minimum number of interactions user should have to be used for training. Should be greater
        than 1.
    loss : {"softmax", "BCE", "gBCE"}, default "softmax"
        Loss function.
    n_negatives : int, default 1
        Number of negatives for BCE and gBCE losses.
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
    use_key_padding_mask : bool, default ``True``
        If ``True``, key_padding_mask will be added in Multi-head Attention.
    use_causal_attn : bool, default ``False``
        If ``True``, causal mask will be added as attn_mask in Multi-head Attention. Please note that default
        BERT4Rec training task ("MLM") does not work with causal masking. Set this
        parameter to ``True`` only when you change the training task with custom
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
    transformer_layers_type : type(TransformerLayersBase), default `PreLNTransformerLayers`
        Type of transformer layers architecture.
    data_preparator_type : type(TransformerDataPreparatorBase), default `BERT4RecDataPreparator`
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
    recommend_batch_size : int, default 256
        How many samples per batch to load during `recommend`.
        If you want to change this parameter after model is initialized,
        you can manually assign new value to model `recommend_batch_size` attribute.
    recommend_device : {"cpu", "cuda", "cuda:0", ...}, default ``None``
        String representation for `torch.device` used for recommendations.
        When set to ``None``, "cuda" will be used if it is available, "cpu" otherwise.
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
    """

    config_class = BERT4RecModelConfig

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 2,
        n_heads: int = 4,
        n_factors: int = 256,
        dropout_rate: float = 0.2,
        mask_prob: float = 0.15,
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
        use_key_padding_mask: bool = True,
        use_causal_attn: bool = False,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        item_net_constructor_type: tp.Type[ItemNetConstructorBase] = SumOfEmbeddingsConstructor,
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = PreLNTransformerLayers,
        data_preparator_type: tp.Type[TransformerDataPreparatorBase] = BERT4RecDataPreparator,
        lightning_module_type: tp.Type[TransformerLightningModuleBase] = TransformerLightningModule,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        get_trainer_func: tp.Optional[TrainerCallable] = None,
        recommend_batch_size: int = 256,
        recommend_device: tp.Optional[str] = None,
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,  # TODO: remove after TorchRanker
    ):
        self.mask_prob = mask_prob

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
            recommend_device=recommend_device,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
            train_min_user_interactions=train_min_user_interactions,
            item_net_block_types=item_net_block_types,
            item_net_constructor_type=item_net_constructor_type,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
            get_val_mask_func=get_val_mask_func,
            get_trainer_func=get_trainer_func,
        )

    def _init_data_preparator(self) -> None:
        self.data_preparator: TransformerDataPreparatorBase = self.data_preparator_type(
            session_max_len=self.session_max_len,
            n_negatives=self.n_negatives if self.loss != "softmax" else None,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            mask_prob=self.mask_prob,
            get_val_mask_func=self.get_val_mask_func,
        )
