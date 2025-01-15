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
from copy import deepcopy

import numpy as np
import torch
from implicit.gpu import HAS_CUDA
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator

from rectools import ExternalIds
from rectools.dataset import Dataset
from rectools.models.base import ErrorBehaviour, InternalRecoTriplet, ModelBase
from rectools.models.rank import Distance, ImplicitRanker
from rectools.types import InternalIdsArray

from .item_net import CatFeaturesItemNet, IdEmbeddingsItemNet, ItemNetBase, ItemNetConstructor
from .net_blocks import (
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    PreLNTransformerLayers,
    TransformerLayersBase,
)
from .transformer_data_preparator import SessionEncoderDataPreparatorBase

PADDING_VALUE = "PAD"


class TransformerBasedSessionEncoder(torch.nn.Module):
    """
    Torch model for recommendations.

    Parameters
    ----------
    n_blocks : int
        Number of transformer blocks.
    n_factors : int
        Latent embeddings size.
    n_heads : int
        Number of attention heads.
    session_max_len : int
        Maximum length of user sequence.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    use_pos_emb : bool, default True
        If ``True``, learnable positional encoding will be added to session item embeddings.
    use_causal_attn : bool, default True
        If ``True``, causal mask is used in multi-head self-attention.
    transformer_layers_type : Type(TransformerLayersBase), default `PreLNTransformerLayers`
        Type of transformer layers architecture.
    item_net_type : Type(ItemNetBase), default IdEmbeddingsItemNet
        Type of network returning item embeddings.
    pos_encoding_type : Type(PositionalEncodingBase), default LearnableInversePositionalEncoding
        Type of positional encoding.
    """

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        session_max_len: int,
        dropout_rate: float,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        transformer_layers_type: tp.Type[TransformerLayersBase] = PreLNTransformerLayers,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
    ) -> None:
        super().__init__()

        self.item_model: ItemNetConstructor
        self.pos_encoding = pos_encoding_type(use_pos_emb, session_max_len, n_factors)
        self.emb_dropout = torch.nn.Dropout(dropout_rate)
        self.transformer_layers = transformer_layers_type(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
        )
        self.use_causal_attn = use_causal_attn
        self.use_key_padding_mask = use_key_padding_mask
        self.n_factors = n_factors
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads

        self.item_net_block_types = item_net_block_types

    def construct_item_net(self, dataset: Dataset) -> None:
        """
        Construct network for item embeddings from dataset.

        Parameters
        ----------
        dataset : Dataset
            RecTools dataset with user-item interactions.
        """
        self.item_model = ItemNetConstructor.from_dataset(
            dataset, self.n_factors, self.dropout_rate, self.item_net_block_types
        )

    @staticmethod
    def _convert_mask_to_float(mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(mask, dtype=query.dtype).masked_fill_(mask, float("-inf"))

    def _merge_masks(
        self, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge `attn_mask` and `key_padding_mask` as a new `attn_mask`.
        Both masks are expanded to shape ``(batch_size * n_heads, session_max_len, session_max_len)``
        and combined with logical ``or``.
        Diagonal elements in last two dimensions are set equal to ``0``.
        This prevents nan values in gradients for pytorch < 2.5.0 when both masks are present in forward pass of
        `torch.nn.MultiheadAttention` (https://github.com/pytorch/pytorch/issues/41508).

        Parameters
        ----------
        attn_mask:  torch.Tensor. [session_max_len, session_max_len]
            Boolean causal attention mask.
        key_padding_mask: torch.Tensor. [batch_size, session_max_len]
            Boolean padding mask.
        query: torch.Tensor
            Query tensor used to acquire correct shapes and dtype for new `attn_mask`.

        Returns
        -------
        torch.Tensor. [batch_size * n_heads, session_max_len, session_max_len]
            Merged mask to use as new `attn_mask` with zeroed diagonal elements in last 2 dimensions.
        """
        batch_size, seq_len, _ = query.shape

        key_padding_mask_expanded = self._convert_mask_to_float(  # [batch_size, session_max_len]
            key_padding_mask, query
        ).view(
            batch_size, 1, seq_len
        )  # [batch_size, 1, session_max_len]

        attn_mask_expanded = (
            self._convert_mask_to_float(attn_mask, query)  # [session_max_len, session_max_len]
            .view(1, seq_len, seq_len)
            .expand(batch_size, -1, -1)
        )  # [batch_size, session_max_len, session_max_len]

        merged_mask = attn_mask_expanded + key_padding_mask_expanded
        res = (
            merged_mask.view(batch_size, 1, seq_len, seq_len)
            .expand(-1, self.n_heads, -1, -1)
            .view(-1, seq_len, seq_len)
        )  # [batch_size * n_heads, session_max_len, session_max_len]
        torch.diagonal(res, dim1=1, dim2=2).zero_()
        return res

    def encode_sessions(self, sessions: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings.
        Add positional encoding.
        Pass history through transformer blocks.

        Parameters
        ----------
        sessions :  torch.Tensor
            User sessions in the form of sequences of items ids.
        item_embs : torch.Tensor
            Item embeddings.

        Returns
        -------
        torch.Tensor. [batch_size, session_max_len, n_factors]
            Encoded session embeddings.
        """
        session_max_len = sessions.shape[1]
        attn_mask = None
        key_padding_mask = None

        timeline_mask = (sessions != 0).unsqueeze(-1)  # [batch_size, session_max_len, 1]

        seqs = item_embs[sessions]  # [batch_size, session_max_len, n_factors]
        seqs = self.pos_encoding(seqs)
        seqs = self.emb_dropout(seqs)

        if self.use_causal_attn:
            attn_mask = ~torch.tril(
                torch.ones((session_max_len, session_max_len), dtype=torch.bool, device=sessions.device)
            )
        if self.use_key_padding_mask:
            key_padding_mask = sessions == 0
            if attn_mask is not None:  # merge masks to prevent nan gradients for torch < 2.5.0
                attn_mask = self._merge_masks(attn_mask, key_padding_mask, seqs)
                key_padding_mask = None

        seqs = self.transformer_layers(seqs, timeline_mask, attn_mask, key_padding_mask)
        return seqs

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_max_len]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get item and session embeddings.
        Get item embeddings.
        Pass user sessions through transformer blocks.

        Parameters
        ----------
        sessions : torch.Tensor
            User sessions in the form of sequences of items ids.

        Returns
        -------
        (torch.Tensor, torch.Tensor)
        """
        item_embs = self.item_model.get_all_embeddings()  # [n_items + n_item_extra_tokens, n_factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_max_len, n_factors]
        return item_embs, session_embs


# ####  --------------  Lightning Model  --------------  #### #


class SessionEncoderLightningModuleBase(LightningModule):
    """
    Base class for lightning module. To change train procedure inherit
    from this class and pass your custom LightningModule to your model parameters.

    Parameters
    ----------
    torch_model : TransformerBasedSessionEncoder
        Torch model to make recommendations.
    lr : float
        Learning rate.
    loss : str, default "softmax"
        Loss function.
    adam_betas : Tuple[float, float], default (0.9, 0.98)
        Coefficients for running averages of gradient and its square.
    """

    def __init__(
        self,
        torch_model: TransformerBasedSessionEncoder,
        lr: float,
        gbce_t: float,
        n_item_extra_tokens: int,
        loss: str = "softmax",
        adam_betas: tp.Tuple[float, float] = (0.9, 0.98),
    ):
        super().__init__()
        self.lr = lr
        self.loss = loss
        self.torch_model = torch_model
        self.adam_betas = adam_betas
        self.gbce_t = gbce_t
        self.n_item_extra_tokens = n_item_extra_tokens
        self.item_embs: torch.Tensor

    def configure_optimizers(self) -> torch.optim.Adam:
        """Choose what optimizers and learning-rate schedulers to use in optimization"""
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        raise NotImplementedError()

    def predict_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        raise NotImplementedError()


class SessionEncoderLightningModule(SessionEncoderLightningModuleBase):
    """Lightning module to train SASRec model."""

    def on_train_start(self) -> None:
        """Initialize parameters with values from Xavier normal distribution."""
        # TODO: init padding embedding with zeros
        self._xavier_normal_init()

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y, w = batch["x"], batch["y"], batch["yw"]
        if self.loss == "softmax":
            logits = self._get_full_catalog_logits(x)
            return self._calc_softmax_loss(logits, y, w)
        if self.loss == "BCE":
            negatives = batch["negatives"]
            logits = self._get_pos_neg_logits(x, y, negatives)
            return self._calc_bce_loss(logits, y, w)
        if self.loss == "gBCE":
            negatives = batch["negatives"]
            logits = self._get_pos_neg_logits(x, y, negatives)
            return self._calc_gbce_loss(logits, y, w, negatives)

        raise ValueError(f"loss {self.loss} is not supported")

    def _get_full_catalog_logits(self, x: torch.Tensor) -> torch.Tensor:
        item_embs, session_embs = self.torch_model(x)
        logits = session_embs @ item_embs.T
        return logits

    def _get_pos_neg_logits(self, x: torch.Tensor, y: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        # [n_items + n_item_extra_tokens, n_factors], [batch_size, session_max_len, n_factors]
        item_embs, session_embs = self.torch_model(x)
        pos_neg = torch.cat([y.unsqueeze(-1), negatives], dim=-1)  # [batch_size, session_max_len, n_negatives + 1]
        pos_neg_embs = item_embs[pos_neg]  # [batch_size, session_max_len, n_negatives + 1, n_factors]
        # [batch_size, session_max_len, n_negatives + 1]
        logits = (pos_neg_embs @ session_embs.unsqueeze(-1)).squeeze(-1)
        return logits

    def _get_reduced_overconfidence_logits(self, logits: torch.Tensor, n_items: int, n_negatives: int) -> torch.Tensor:
        # https://arxiv.org/pdf/2308.07192.pdf
        alpha = n_negatives / (n_items - 1)  # sampling rate
        beta = alpha * (self.gbce_t * (1 - 1 / alpha) + 1 / alpha)

        pos_logits = logits[:, :, 0:1].to(torch.float64)
        neg_logits = logits[:, :, 1:].to(torch.float64)

        epsilon = 1e-10
        pos_probs = torch.clamp(torch.sigmoid(pos_logits), epsilon, 1 - epsilon)
        pos_probs_adjusted = torch.clamp(pos_probs.pow(-beta), 1 + epsilon, torch.finfo(torch.float64).max)
        pos_probs_adjusted = torch.clamp(
            torch.div(1, (pos_probs_adjusted - 1)), epsilon, torch.finfo(torch.float64).max
        )
        pos_logits_transformed = torch.log(pos_probs_adjusted)
        logits = torch.cat([pos_logits_transformed, neg_logits], dim=-1)
        return logits

    @classmethod
    def _calc_softmax_loss(cls, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # We are using CrossEntropyLoss with a multi-dimensional case

        # Logits must be passed in form of [batch_size, n_items + n_item_extra_tokens, session_max_len],
        #  where n_items + n_item_extra_tokens is number of classes

        # Target label indexes must be passed in a form of [batch_size, session_max_len]
        # (`0` index for "PAD" ix excluded from loss)

        # Loss output will have a shape of [batch_size, session_max_len]
        # and will have zeros for every `0` target label
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), y, ignore_index=0, reduction="none"
        )  # [batch_size, session_max_len]
        loss = loss * w
        n = (loss > 0).to(loss.dtype)
        loss = torch.sum(loss) / torch.sum(n)
        return loss

    @classmethod
    def _calc_bce_loss(cls, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mask = y != 0
        target = torch.zeros_like(logits)
        target[:, :, 0] = 1

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )  # [batch_size, session_max_len, n_negatives + 1]
        loss = loss.mean(-1) * mask * w  # [batch_size, session_max_len]
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

    def _calc_gbce_loss(
        self, logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        n_actual_items = self.torch_model.item_model.n_items - self.n_item_extra_tokens
        n_negatives = negatives.shape[2]
        logits = self._get_reduced_overconfidence_logits(logits, n_actual_items, n_negatives)
        loss = self._calc_bce_loss(logits, y, w)
        return loss

    def on_train_end(self) -> None:
        """Save item embeddings"""
        self.eval()
        self.item_embs = self.torch_model.item_model.get_all_embeddings()

    def predict_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Prediction step.
        Encode user sessions.
        """
        encoded_sessions = self.torch_model.encode_sessions(batch["x"], self.item_embs.to(self.device))[:, -1, :]
        return encoded_sessions

    def _xavier_normal_init(self) -> None:
        for _, param in self.torch_model.named_parameters():
            # ValueError is raised if `param.data` is a tensor with fewer than 2 dimensions.
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError:
                pass


# ####  --------------  Transformer Model Base  --------------  #### #


class TransformerModelBase(ModelBase):  # pylint: disable=too-many-instance-attributes
    """
    Base model for all recommender algorithms that work on transformer architecture (e.g. SASRec, Bert4Rec).
    To create a custom transformer model it is necessary to inherit from this class
    and write self.data_preparator initialization logic.
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        transformer_layers_type: tp.Type[TransformerLayersBase],
        data_preparator_type: tp.Type[SessionEncoderDataPreparatorBase],
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        dropout_rate: float = 0.2,
        session_max_len: int = 32,
        loss: str = "softmax",
        gbce_t: float = 0.5,
        lr: float = 0.01,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_device: tp.Union[str, Accelerator] = "auto",
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(verbose)
        self.recommend_n_threads = recommend_n_threads
        self.recommend_device = recommend_device
        self.recommend_use_gpu_ranking = recommend_use_gpu_ranking
        self._torch_model = TransformerBasedSessionEncoder(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            session_max_len=session_max_len,
            dropout_rate=dropout_rate,
            use_pos_emb=use_pos_emb,
            use_causal_attn=use_causal_attn,
            use_key_padding_mask=use_key_padding_mask,
            transformer_layers_type=transformer_layers_type,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
        )
        self.lightning_model: SessionEncoderLightningModuleBase
        self.lightning_module_type = lightning_module_type
        self.fit_trainer: Trainer
        if trainer is None:
            self._trainer = Trainer(
                max_epochs=epochs,
                min_epochs=epochs,
                deterministic=deterministic,
                enable_progress_bar=verbose > 0,
                enable_model_summary=verbose > 0,
                logger=verbose > 0,
            )
        else:
            self._trainer = trainer
        self.data_preparator: SessionEncoderDataPreparatorBase
        self.u2i_dist = Distance.DOT
        self.i2i_dist = Distance.COSINE
        self.lr = lr
        self.loss = loss
        self.gbce_t = gbce_t

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        processed_dataset = self.data_preparator.process_dataset_train(dataset)
        train_dataloader = self.data_preparator.get_dataloader_train(processed_dataset)

        torch_model = deepcopy(self._torch_model)
        torch_model.construct_item_net(processed_dataset)

        n_item_extra_tokens = self.data_preparator.n_item_extra_tokens
        self.lightning_model = self.lightning_module_type(
            torch_model=torch_model,
            lr=self.lr,
            loss=self.loss,
            gbce_t=self.gbce_t,
            n_item_extra_tokens=n_item_extra_tokens,
        )

        self.fit_trainer = deepcopy(self._trainer)
        self.fit_trainer.fit(self.lightning_model, train_dataloader)

    def _custom_transform_dataset_u2i(
        self, dataset: Dataset, users: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        return self.data_preparator.transform_dataset_u2i(dataset, users)

    def _custom_transform_dataset_i2i(
        self, dataset: Dataset, target_items: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        return self.data_preparator.transform_dataset_i2i(dataset)

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        dataset: Dataset,  # [n_rec_users x n_items + n_item_extra_tokens]
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],  # model_internal
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()  # model internal

        recommend_trainer = Trainer(devices=1, accelerator=self.recommend_device)
        recommend_dataloader = self.data_preparator.get_dataloader_recommend(dataset)
        session_embs = recommend_trainer.predict(model=self.lightning_model, dataloaders=recommend_dataloader)
        if session_embs is None:
            explanation = """Received empty recommendations."""
            raise ValueError(explanation)
        user_embs = np.concatenate(session_embs, axis=0)
        user_embs = user_embs[user_ids]
        item_embs = self.lightning_model.item_embs
        item_embs_np = item_embs.detach().cpu().numpy()

        ranker = ImplicitRanker(
            self.u2i_dist,
            user_embs,  # [n_rec_users, n_factors]
            item_embs_np,  # [n_items + n_item_extra_tokens, n_factors]
        )
        if filter_viewed:
            user_items = dataset.get_user_item_matrix(include_weights=False)
            ui_csr_for_filter = user_items[user_ids]
        else:
            ui_csr_for_filter = None

        # TODO: When filter_viewed is not needed and user has GPU, torch DOT and topk should be faster
        user_ids_indices, all_reco_ids, all_scores = ranker.rank(
            subject_ids=np.arange(user_embs.shape[0]),  # n_rec_users
            k=k,
            filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + n_item_extra_tokens]
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model_internal
            num_threads=self.recommend_n_threads,
            use_gpu=self.recommend_use_gpu_ranking and HAS_CUDA,
        )
        all_target_ids = user_ids[user_ids_indices]
        return all_target_ids, all_reco_ids, all_scores  # n_rec_users, model_internal, scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,  # model internal
        dataset: Dataset,
        k: int,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()

        item_embs = self.lightning_model.item_embs.detach().cpu().numpy()
        # TODO: i2i reco do not need filtering viewed. And user most of the times has GPU
        # Should we use torch dot and topk? Should be faster

        ranker = ImplicitRanker(
            self.i2i_dist,
            item_embs,  # [n_items + n_item_extra_tokens, n_factors]
            item_embs,  # [n_items + n_item_extra_tokens, n_factors]
        )
        return ranker.rank(
            subject_ids=target_ids,  # model internal
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model internal
            num_threads=self.recommend_n_threads,
            use_gpu=self.recommend_use_gpu_ranking and HAS_CUDA,
        )

    @property
    def torch_model(self) -> TransformerBasedSessionEncoder:
        """Pytorch model."""
        return self.lightning_model.torch_model
