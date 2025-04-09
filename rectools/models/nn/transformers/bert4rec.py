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

from ..item_net import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructorBase,
    SumOfEmbeddingsConstructor,
)
from .base import (
    InitKwargs,
    TrainerCallable,
    TransformerDataPreparatorType,
    TransformerLightningModule,
    TransformerLightningModuleBase,
    TransformerModelBase,
    TransformerModelConfig,
    ValMaskCallable,
)
from .constants import MASKING_VALUE, PADDING_VALUE
from .data_preparator import TransformerDataPreparatorBase
from .negative_sampler import CatalogUniformSampler, TransformerNegativeSamplerBase
from .net_blocks import (
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    PreLNTransformerLayers,
    TransformerLayersBase,
)
from .similarity import DistanceSimilarityModule, SimilarityModuleBase
from .torch_backbone import TransformerBackboneBase, TransformerTorchBackbone


class BERT4RecDataPreparator(TransformerDataPreparatorBase):
    """Data Preparator for BERT4RecModel.

    Parameters
    ----------
    session_max_len : int
        Maximum length of user sequence.
    batch_size : int
        How many samples per batch to load.
    dataloader_num_workers : int
        Number of loader worker processes.
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
    mask_prob : float, default 0.15
        Probability of masking an item in interactions sequence.
    """

    train_session_max_len_addition: int = 0
    item_extra_tokens: tp.Sequence[Hashable] = (PADDING_VALUE, MASKING_VALUE)

    def __init__(
        self,
        session_max_len: int,
        n_negatives: tp.Optional[int],
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        negative_sampler: tp.Optional[TransformerNegativeSamplerBase] = None,
        mask_prob: float = 0.15,
        shuffle_train: bool = True,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            negative_sampler=negative_sampler,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            shuffle_train=shuffle_train,
            get_val_mask_func=get_val_mask_func,
        )
        self.mask_prob = mask_prob

    def _mask_session(
        self,
        ses: List[int],
        first_border: float = 0.8,
        second_border: float = 0.9,
    ) -> Tuple[List[int], List[int]]:
        masked_session = ses.copy()
        target = ses.copy()
        random_probs = np.random.rand(len(ses))
        for j in range(len(ses)):
            if random_probs[j] < self.mask_prob:
                random_probs[j] /= self.mask_prob
                if random_probs[j] < first_border:
                    masked_session[j] = self.extra_token_ids[MASKING_VALUE]
                elif random_probs[j] < second_border:
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
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size
            )
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
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict, lowest_id=self.n_item_extra_tokens, highest_id=self.item_id_map.size, session_len_limit=1
            )
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
        negative_sampler_type: tp.Type[TransformerNegativeSamplerBase] = CatalogUniformSampler,
        similarity_module_type: tp.Type[SimilarityModuleBase] = DistanceSimilarityModule,
        backbone_type: tp.Type[TransformerBackboneBase] = TransformerTorchBackbone,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        get_trainer_func: tp.Optional[TrainerCallable] = None,
        recommend_batch_size: int = 256,
        recommend_torch_device: tp.Optional[str] = None,
        recommend_use_torch_ranking: bool = True,
        recommend_n_threads: int = 0,
        data_preparator_kwargs: tp.Optional[InitKwargs] = None,
        transformer_layers_kwargs: tp.Optional[InitKwargs] = None,
        item_net_block_kwargs: tp.Optional[InitKwargs] = None,
        item_net_constructor_kwargs: tp.Optional[InitKwargs] = None,
        pos_encoding_kwargs: tp.Optional[InitKwargs] = None,
        lightning_module_kwargs: tp.Optional[InitKwargs] = None,
        negative_sampler_kwargs: tp.Optional[InitKwargs] = None,
        similarity_module_kwargs: tp.Optional[InitKwargs] = None,
        backbone_kwargs: tp.Optional[InitKwargs] = None,
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
            data_preparator_kwargs=data_preparator_kwargs,
            transformer_layers_kwargs=transformer_layers_kwargs,
            item_net_block_kwargs=item_net_block_kwargs,
            item_net_constructor_kwargs=item_net_constructor_kwargs,
            pos_encoding_kwargs=pos_encoding_kwargs,
            lightning_module_kwargs=lightning_module_kwargs,
            negative_sampler_kwargs=negative_sampler_kwargs,
            similarity_module_kwargs=similarity_module_kwargs,
            backbone_kwargs=backbone_kwargs,
        )

    def _init_data_preparator(self) -> None:
        requires_negatives = self.lightning_module_type.requires_negatives(self.loss)
        self.data_preparator: TransformerDataPreparatorBase = self.data_preparator_type(
            session_max_len=self.session_max_len,
            n_negatives=self.n_negatives if requires_negatives else None,
            negative_sampler=self._init_negative_sampler() if requires_negatives else None,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            mask_prob=self.mask_prob,
            get_val_mask_func=self.get_val_mask_func,
            shuffle_train=True,
            **self._get_kwargs(self.data_preparator_kwargs),
        )
