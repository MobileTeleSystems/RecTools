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
import warnings
from typing import Dict

import torch
from torch import nn

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
from .data_preparator import InitKwargs, TransformerDataPreparatorBase
from .negative_sampler import CatalogUniformSampler, TransformerNegativeSamplerBase
from .net_blocks import LearnableInversePositionalEncoding, PositionalEncodingBase, TransformerLayersBase
from .sasrec import SASRecDataPreparator
from .similarity import DistanceSimilarityModule, SimilarityModuleBase
from .torch_backbone import TransformerBackboneBase, TransformerTorchBackbone


class RelativeAttentionBias(torch.nn.Module):
    """
    Computes relative time and positional attention biases for STU.

    Parameters
    ----------
    session_max_len : int
        Maximum sequence length for user interactions (padded/truncated)
    relative_time_attention : bool
        Whether to compute relative time attention from timestamps
    relative_pos_attention : bool
        Whether to compute relative positional attention
    num_buckets : int
        Number of buckets for quantizing timestamp differences
    """

    def __init__(
        self,
        session_max_len: int,
        relative_time_attention: bool,
        relative_pos_attention: bool,
        num_buckets: int = 128,
    ) -> None:
        super().__init__()
        self.session_max_len = session_max_len
        self.num_buckets = num_buckets
        self.relative_time_attention = relative_time_attention
        self.relative_pos_attention = relative_pos_attention
        if relative_time_attention:
            self.time_weights = torch.nn.Parameter(
                torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
            )
        if relative_pos_attention:
            self.pos_weights = torch.nn.Parameter(
                torch.empty(2 * session_max_len - 1).normal_(mean=0, std=0.02),
            )

    def _quantization_func(self, diff_timestamps: torch.Tensor) -> torch.Tensor:
        """Quantizes the differences between timestamps into discrete buckets."""
        return (torch.log(torch.abs(diff_timestamps).clamp(min=1)) / 0.301).long()

    def forward_time_attention(self, all_timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---------
        all_timestamps: torch.Tensor (batch_size, session_max_len+1)
            User interaction timestamps including the target item timestamp
        Returns
        ---------
        torch.Tensor (batch_size, session_max_len, session_max_len)
            relative time attention
        """
        len_expanded = self.session_max_len + 1  # 1 for target item time, needed for time aware
        batch_size = all_timestamps.size(0)
        extended_timestamps = torch.cat([all_timestamps, all_timestamps[:, len_expanded - 1 : len_expanded]], dim=1)
        early_time_binding = extended_timestamps[:, 1:].unsqueeze(2) - extended_timestamps[:, :-1].unsqueeze(1)
        bucketed_timestamps = torch.clamp(
            self._quantization_func(early_time_binding),
            min=0,
            max=self.num_buckets,
        ).detach()
        rel_time_attention = torch.index_select(self.time_weights, dim=0, index=bucketed_timestamps.view(-1)).view(
            batch_size, len_expanded, len_expanded
        )
        # reducted target time
        rel_time_attention = rel_time_attention[:, :-1, :-1]
        return rel_time_attention  # (batch_size, session_max_len, session_max_len)

    def forward_pos_attention(self) -> torch.Tensor:
        """
        Compute and return the relative positional attention bias matrix.

        Returns
        -------
        torch.Tensor (1, session_max_len, session_max_len)
        """
        n = self.session_max_len
        t = nn.functional.pad(self.pos_weights[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        rel_pos_attention = t[:, :, r:-r]
        return rel_pos_attention

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute relative attention biases.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Could contain payload information, in particular sequence timestamps.

        Returns
        -------
        torch.Tensor (batch_size, session_max_len, session_max_len)
            Variate of sum relative pos/time attention
        """
        batch_size = batch["x"].size(0)
        rel_attn = torch.zeros((batch_size, self.session_max_len, self.session_max_len)).to(batch["x"].device)
        if self.relative_time_attention:
            rel_attn += self.forward_time_attention(batch["unix_ts"])
        if self.relative_pos_attention:
            rel_attn += self.forward_pos_attention()
        return rel_attn


class STULayer(nn.Module):
    """
    HSTU author's encoder block architecture rewritten from jagged tensor to dense.

    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    linear_hidden_dim : int
        U, V size.
    attention_dim : int
        Q, K size.
    session_max_len : int
        Maximum length of user sequence padded or truncated to.
    relative_time_attention : bool
        Whether to use relative time attention.
    relative_pos_attention : bool
        Whether to use relative positional attention
    attn_dropout_rate : float
        Probability of an attention unit to be zeroed.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    epsilon  : float
        A value passed to LayerNorm for numerical stability.
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        linear_hidden_dim: int,
        attention_dim: int,
        session_max_len: int,
        relative_time_attention: bool,
        relative_pos_attention: bool,
        attn_dropout_rate: float,
        dropout_rate: float,
        epsilon: float,
    ):
        super().__init__()
        self.rel_attn = RelativeAttentionBias(
            session_max_len=session_max_len,
            relative_time_attention=relative_time_attention,
            relative_pos_attention=relative_pos_attention,
        )
        self.n_heads = n_heads
        self.linear_hidden_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.session_max_len = session_max_len
        self.uvqk_proj: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                (
                    n_factors,
                    linear_hidden_dim * 2 * n_heads + attention_dim * n_heads * 2,
                )
            ),
        )
        self.output_mlp = torch.nn.Linear(
            in_features=linear_hidden_dim * n_heads,
            out_features=n_factors,
        )
        self.norm_input = nn.LayerNorm(n_factors, eps=epsilon)
        self.norm_attn_output = nn.LayerNorm(linear_hidden_dim * n_heads, eps=epsilon)
        self.dropout_mlp = nn.Dropout(dropout_rate)
        self.dropout_attn = nn.Dropout(attn_dropout_rate)
        self.silu = nn.SiLU()

    def forward(
        self,
        seqs: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        attn_mask: torch.Tensor,
        timeline_mask: torch.Tensor,
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through STU.

        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.
        batch : torch.Tensor
            Could contain payload information, in particular sequence timestamps.
        attn_mask : torch.Tensor
            Mask to use in forward pass of multi-head attention as `attn_mask`.
        timeline_mask : torch.Tensor
            Mask marked padding items.
        key_padding_mask : torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `key_padding_mask`.


        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        batch_size, _, _ = seqs.shape
        normed_x = self.norm_input(seqs) * timeline_mask  # prevent null emb convert to not null
        general_transform = torch.matmul(normed_x, self.uvqk_proj)
        batched_mm_output = self.silu(general_transform)
        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self.linear_hidden_dim * self.n_heads,
                self.linear_hidden_dim * self.n_heads,
                self.attention_dim * self.n_heads,
                self.attention_dim * self.n_heads,
            ],
            dim=-1,
        )
        # (batch_size, n_head, session_max_len, session_max_len), attention on Q, K
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q.view(batch_size, self.session_max_len, self.n_heads, self.attention_dim),
            k.view(batch_size, self.session_max_len, self.n_heads, self.attention_dim),
        )
        # (batch_size, session_max_len, session_max_len).unsqueeze(1) for broadcast
        qk_attn = qk_attn + self.rel_attn(batch).unsqueeze(1)
        qk_attn = self.silu(qk_attn) / self.session_max_len

        time_line_mask_reducted = timeline_mask.squeeze(-1)
        time_line_mask_fix = time_line_mask_reducted.unsqueeze(1) * timeline_mask

        qk_attn = qk_attn * attn_mask.unsqueeze(0).unsqueeze(0) * time_line_mask_fix.unsqueeze(1)

        attn_output = torch.einsum(
            "bhnm,bmhd->bnhd",
            qk_attn,
            v.reshape(batch_size, self.session_max_len, self.n_heads, self.linear_hidden_dim),
        ).reshape(batch_size, self.session_max_len, self.n_heads * self.linear_hidden_dim)

        attn_output = self.dropout_attn(attn_output)
        o_input = u * self.norm_attn_output(attn_output) * timeline_mask

        new_outputs = self.output_mlp(self.dropout_mlp(o_input)) + seqs

        return new_outputs


class STULayers(TransformerLayersBase):
    """
    STULayers transformer blocks.

    Parameters
    ----------
    n_blocks : int
        Numbers of stacked STU.
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    linear_hidden_dim : int
        U, V size.
    attention_dim : int
        Q, K size.
    session_max_len : int
        Maximum length of user sequence padded or truncated to.
    relative_time_attention : bool
        Whether to use relative time attention.
    relative_pos_attention : bool
        Whether to use relative positional attention
    attn_dropout_rate : float, default 0.2
        Probability of an attention unit to be zeroed.
    dropout_rate : float, default 0.2
        Probability of a hidden unit to be zeroed.
    epsilon : float, default 1e-6
        A value passed to LayerNorm for numerical stability.
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
        attn_dropout_rate: float = 0.0,
        dropout_rate: float = 0.2,
        epsilon: float = 1e-6,
        **kwargs: tp.Any,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.epsilon = epsilon
        self.stu_blocks = nn.ModuleList(
            [
                STULayer(
                    n_factors=n_factors,
                    n_heads=n_heads,
                    dropout_rate=dropout_rate,
                    linear_hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    relative_time_attention=relative_time_attention,
                    relative_pos_attention=relative_pos_attention,
                    attn_dropout_rate=attn_dropout_rate,
                    session_max_len=session_max_len,
                    epsilon=epsilon,
                )
                for _ in range(self.n_blocks)
            ]
        )

    def forward(  # type: ignore
        self,
        seqs: torch.Tensor,
        timeline_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: tp.Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """
        Forward pass through STU blocks.

        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.
        timeline_mask : torch.Tensor
            Mask indicating padding elements.
        attn_mask : torch.Tensor, optional
            Mask to use in forward pass of multi-head attention as `attn_mask`.
        key_padding_mask : torch.Tensor, optional
            Mask to use in forward pass of multi-head attention as `key_padding_mask`.
        batch : Dict[str, torch.Tensor]
            Could contain payload information,in particular sequence timestamps.

        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        attn_mask = (~attn_mask).int()
        for i in range(self.n_blocks):
            seqs *= timeline_mask  # [batch_size, session_max_len, n_factors]
            seqs = self.stu_blocks[i](seqs, batch, attn_mask, timeline_mask, key_padding_mask)
        seqs *= timeline_mask
        return seqs


class HSTUModelConfig(TransformerModelConfig):
    """HSTU model config."""

    data_preparator_type: TransformerDataPreparatorType = SASRecDataPreparator
    transformer_layers_type: TransformerLayersType = STULayers
    use_causal_attn: bool = True
    relative_time_attention: bool = True
    relative_pos_attention: bool = True


class HSTUModel(TransformerModelBase[HSTUModelConfig]):
    """
    HSTU model: transformer-based sequential model with unidirectional pointwise aggregated attention mechanism,
    combined with "Shifted Sequence" training objective.
    Our implementation covers multiple loss functions and a variable number of negatives for them.

    References
    ----------
    HSTU tutorial: https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_HSTU_tutorial.html
    Original paper: https://arxiv.org/abs/2402.17152


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
    relative_time_attention : bool
        Whether to use relative time attention.
    relative_pos_attention : bool
        Whether to use relative positional attention
    item_net_block_types : sequence of `type(ItemNetBase)`, default `(IdEmbeddingsItemNet, CatFeaturesItemNet)`
        Type of network returning item embeddings.
        (IdEmbeddingsItemNet,) - item embeddings based on ids.
        (CatFeaturesItemNet,) - item embeddings based on categorical features.
        (IdEmbeddingsItemNet, CatFeaturesItemNet) - item embeddings based on ids and categorical features.
    item_net_constructor_type : type(ItemNetConstructorBase), default `SumOfEmbeddingsConstructor`
        Type of item net blocks aggregation constructor.
    pos_encoding_type : type(PositionalEncodingBase), default `LearnableInversePositionalEncoding`
        Type of positional encoding.
    transformer_layers_type : type(TransformerLayersBase), default `STULayers`
        Type of transformer layers architecture.
    data_preparator_type : type(TransformerDataPreparatorBase), default `HSTUDataPreparator`
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
        Let's add comment about our changes for default module kwargs:

    To precisely follow the original authors implementations of the model,
    the following kwargs for specific modules will be replaced from their default versions
    used in other Transformer models:
    1)use_scale_factor in pos_encoding_kwargs will be set to True
    2)distance in similarity_module_kwargs will be set to cosine
     if not explicitly provided as others options

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
        relative_time_attention: bool = True,
        relative_pos_attention: bool = True,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        item_net_constructor_type: tp.Type[ItemNetConstructorBase] = SumOfEmbeddingsConstructor,
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = STULayers,
        data_preparator_type: tp.Type[TransformerDataPreparatorBase] = SASRecDataPreparator,
        lightning_module_type: tp.Type[TransformerLightningModuleBase] = TransformerLightningModule,
        negative_sampler_type: tp.Type[TransformerNegativeSamplerBase] = CatalogUniformSampler,
        similarity_module_type: tp.Type[SimilarityModuleBase] = DistanceSimilarityModule,
        backbone_type: tp.Type[TransformerBackboneBase] = TransformerTorchBackbone,
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
        if n_factors % n_heads != 0:
            raise ValueError("n_factors must be divisible by n_heads without remainder")
        if use_key_padding_mask:
            warnings.warn(
                "'use_key_padding_mask' is not supported for HSTU and enforced to False.", UserWarning, stacklevel=2
            )
            use_key_padding_mask = False
        self.relative_time_attention = relative_time_attention
        self.relative_pos_attention = relative_pos_attention
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

    def _init_transformer_layers(self) -> TransformerLayersBase:
        head_dim = self.n_factors // self.n_heads
        return self.transformer_layers_type(
            n_blocks=self.n_blocks,
            n_factors=self.n_factors,
            n_heads=self.n_heads,
            session_max_len=self.session_max_len,
            attention_dim=head_dim,
            linear_hidden_dim=head_dim,
            dropout_rate=self.dropout_rate,
            relative_time_attention=self.relative_time_attention,
            relative_pos_attention=self.relative_pos_attention,
            **self._get_kwargs(self.transformer_layers_kwargs),
        )

    def _init_data_preparator(self) -> None:
        requires_negatives = self.lightning_module_type.requires_negatives(self.loss)
        if self.data_preparator_kwargs is None:
            data_preparator_kwargs = {}
        else:
            data_preparator_kwargs = self.data_preparator_kwargs.copy()
        if self.relative_time_attention:
            data_preparator_kwargs["add_unix_ts"] = True
        self.data_preparator = self.data_preparator_type(
            session_max_len=self.session_max_len,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            negative_sampler=self._init_negative_sampler() if requires_negatives else None,
            n_negatives=self.n_negatives if requires_negatives else None,
            get_val_mask_func=self.get_val_mask_func,
            get_val_mask_func_kwargs=self.get_val_mask_func_kwargs,
            **data_preparator_kwargs,
        )

    def _init_similarity_module(self) -> SimilarityModuleBase:
        if self.similarity_module_kwargs is None:
            similarity_module_kwargs = {}
        else:
            similarity_module_kwargs = self.similarity_module_kwargs.copy()
        if "distance" not in similarity_module_kwargs:
            similarity_module_kwargs["distance"] = "cosine"
        return self.similarity_module_type(**similarity_module_kwargs)

    def _init_pos_encoding_layer(self) -> PositionalEncodingBase:
        if self.pos_encoding_kwargs is None:
            pos_encoding_kwargs = {}
        else:
            pos_encoding_kwargs = self.pos_encoding_kwargs.copy()
        if "use_scale_factor" not in pos_encoding_kwargs:
            pos_encoding_kwargs["use_scale_factor"] = True
        return self.pos_encoding_type(
            self.use_pos_emb,
            self.session_max_len,
            self.n_factors,
            **pos_encoding_kwargs,
        )

    @property
    def require_recommend_context(self) -> bool:
        """
        Indicates whether the model requires context for accurate recommendations.

        -------
        bool
        """
        if self.relative_time_attention:
            return True
        return False
