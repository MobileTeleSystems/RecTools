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

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from rectools import ExternalIds
from rectools.dataset.dataset import Dataset, DatasetSchemaDict
from rectools.models.base import InternalRecoTriplet
from rectools.models.rank import Distance, ImplicitRanker, Ranker, TorchRanker
from rectools.types import InternalIdsArray

from .torch_backbone import TransformerTorchBackbone

# ####  --------------  Lightning Base Model  --------------  #### #


class TransformerLightningModuleBase(LightningModule):  # pylint: disable=too-many-instance-attributes
    """
    Base class for transfofmers lightning module. To change train procedure inherit
    from this class and pass your custom LightningModule to your model parameters.

    Parameters
    ----------
    torch_model : TransformerTorchBackbone
        Torch model to make recommendations.
    lr : float
        Learning rate.
    loss : str, default "softmax"
        Loss function.
    adam_betas : Tuple[float, float], default (0.9, 0.98)
        Coefficients for running averages of gradient and its square.
    data_preparator : TransformerDataPreparatorBase
        Data preparator.
    verbose : int, default 0
        Verbosity level.
    train_loss_name : str, default "train_loss"
        Name of the training loss.
    val_loss_name : str, default "val_loss"
        Name of the training loss.
    """

    def __init__(
        self,
        torch_model: TransformerTorchBackbone,
        model_config: tp.Dict[str, tp.Any],
        dataset_schema: DatasetSchemaDict,
        item_external_ids: ExternalIds,
        item_extra_tokens: tp.Sequence[Hashable],
        lr: float,
        gbce_t: float,
        loss: str,
        verbose: int = 0,
        train_loss_name: str = "train_loss",
        val_loss_name: str = "val_loss",
        adam_betas: tp.Tuple[float, float] = (0.9, 0.98),
    ):
        super().__init__()
        self.torch_model = torch_model
        self.model_config = model_config
        self.dataset_schema = dataset_schema
        self.item_external_ids = item_external_ids
        self.item_extra_tokens = item_extra_tokens
        self.lr = lr
        self.loss = loss
        self.adam_betas = adam_betas
        self.gbce_t = gbce_t
        self.verbose = verbose
        self.train_loss_name = train_loss_name
        self.val_loss_name = val_loss_name
        self.item_embs: torch.Tensor

        self.save_hyperparameters(ignore=["torch_model"])

    def configure_optimizers(self) -> torch.optim.Adam:
        """Choose what optimizers and learning-rate schedulers to use in optimization"""
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        raise NotImplementedError()

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> tp.Dict[str, torch.Tensor]:
        """Validate step."""
        raise NotImplementedError()

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        recommend_dataloader: DataLoader,
        sorted_item_ids_to_recommend: InternalIdsArray,
        k: int,
        dataset: Dataset,  # [n_rec_users x n_items + n_item_extra_tokens]
        filter_viewed: bool,
        recommend_use_torch_ranking: bool,
        recommend_n_threads: int,
        recommend_device: tp.Optional[str],
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> InternalRecoTriplet:
        """Recommending to users."""
        raise NotImplementedError()

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        sorted_item_ids_to_recommend: InternalIdsArray,
        k: int,
        recommend_use_torch_ranking: bool,
        recommend_n_threads: int,
        recommend_device: tp.Optional[str],
        *args: tp.Any,
        **kwargs: tp.Any,
    ) -> InternalRecoTriplet:
        """Recommending to items."""
        raise NotImplementedError()


# ####  --------------  Lightning Model  --------------  #### #


class TransformerLightningModule(TransformerLightningModuleBase):
    """Lightning module to train transformer models."""

    i2i_dist = Distance.COSINE

    def on_train_start(self) -> None:
        """Initialize parameters with values from Xavier normal distribution."""
        self._xavier_normal_init()

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y, w = batch["x"], batch["y"], batch["yw"]
        if self.loss == "softmax":
            logits = self._get_full_catalog_logits(x)
            loss = self._calc_softmax_loss(logits, y, w)
        elif self.loss == "BCE":
            negatives = batch["negatives"]
            logits = self._get_pos_neg_logits(x, y, negatives)
            loss = self._calc_bce_loss(logits, y, w)
        elif self.loss == "gBCE":
            negatives = batch["negatives"]
            logits = self._get_pos_neg_logits(x, y, negatives)
            loss = self._calc_gbce_loss(logits, y, w, negatives)
        else:
            loss = self._calc_custom_loss(batch, batch_idx)

        self.log(self.train_loss_name, loss, on_step=False, on_epoch=True, prog_bar=self.verbose > 0)

        return loss

    def _calc_custom_loss(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        raise ValueError(f"loss {self.loss} is not supported")

    def on_validation_start(self) -> None:
        """Save item embeddings"""
        self.eval()
        with torch.no_grad():
            self.item_embs = self.torch_model.item_model.get_all_embeddings()

    def on_validation_end(self) -> None:
        """Clear item embeddings"""
        del self.item_embs
        torch.cuda.empty_cache()

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int) -> tp.Dict[str, torch.Tensor]:
        """Validate step."""
        # x: [batch_size, session_max_len]
        # y: [batch_size, 1]
        # yw: [batch_size, 1]
        x, y, w = batch["x"], batch["y"], batch["yw"]
        outputs = {}
        if self.loss == "softmax":
            logits = self._get_full_catalog_logits(x)[:, -1:, :]
            outputs["loss"] = self._calc_softmax_loss(logits, y, w)
            outputs["logits"] = logits.squeeze()
        elif self.loss == "BCE":
            negatives = batch["negatives"]
            pos_neg_logits = self._get_pos_neg_logits(x, y, negatives)[:, -1:, :]
            outputs["loss"] = self._calc_bce_loss(pos_neg_logits, y, w)
            outputs["pos_neg_logits"] = pos_neg_logits.squeeze()
        elif self.loss == "gBCE":
            negatives = batch["negatives"]
            pos_neg_logits = self._get_pos_neg_logits(x, y, negatives)[:, -1:, :]
            outputs["loss"] = self._calc_gbce_loss(pos_neg_logits, y, w, negatives)
            outputs["pos_neg_logits"] = pos_neg_logits.squeeze()
        else:
            outputs = self._calc_custom_loss_outputs(batch, batch_idx)  # pragma: no cover

        self.log(self.val_loss_name, outputs["loss"], on_step=False, on_epoch=True, prog_bar=self.verbose > 0)
        return outputs

    def _calc_custom_loss_outputs(
        self, batch: tp.Dict[str, torch.Tensor], batch_idx: int
    ) -> tp.Dict[str, torch.Tensor]:
        raise ValueError(f"loss {self.loss} is not supported")  # pragma: no cover

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
        n_actual_items = self.torch_model.item_model.n_items - len(self.item_extra_tokens)
        n_negatives = negatives.shape[2]
        logits = self._get_reduced_overconfidence_logits(logits, n_actual_items, n_negatives)
        loss = self._calc_bce_loss(logits, y, w)
        return loss

    def _xavier_normal_init(self) -> None:
        for _, param in self.torch_model.named_parameters():
            if param.data.dim() > 1:
                torch.nn.init.xavier_normal_(param.data)

    def _prepare_for_inference(self, recommend_device: tp.Optional[str]) -> None:
        if recommend_device is None:
            recommend_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(recommend_device)
        self.torch_model.to(device)
        self.torch_model.eval()

    def _get_user_item_embeddings(
        self,
        recommend_dataloader: DataLoader,
        recommend_device: tp.Optional[str],
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare user embeddings for all user interaction sequences in `recommend_dataloader`.
        Prepare item embeddings for full items catalog.
        """
        self._prepare_for_inference(recommend_device)
        device = self.torch_model.item_model.device

        with torch.no_grad():
            item_embs = self.torch_model.item_model.get_all_embeddings()
            user_embs = []
            for batch in recommend_dataloader:
                batch_embs = self.torch_model.encode_sessions(batch["x"].to(device), item_embs)[:, -1, :]
                user_embs.append(batch_embs)

        return torch.cat(user_embs), item_embs

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,
        recommend_dataloader: DataLoader,
        sorted_item_ids_to_recommend: InternalIdsArray,
        k: int,
        dataset: Dataset,  # [n_rec_users x n_items + n_item_extra_tokens]
        filter_viewed: bool,
        recommend_use_torch_ranking: bool,
        recommend_n_threads: int,
        recommend_device: tp.Optional[str],
    ) -> InternalRecoTriplet:
        """Recommend to users."""
        ui_csr_for_filter = None
        if filter_viewed:
            ui_csr_for_filter = dataset.get_user_item_matrix(include_weights=False, include_warm_items=True)[user_ids]

        user_embs, item_embs = self._get_user_item_embeddings(recommend_dataloader, recommend_device)

        ranker: Ranker
        if recommend_use_torch_ranking:
            ranker = TorchRanker(
                distance=Distance.DOT,
                device=item_embs.device,
                subjects_factors=user_embs[user_ids],
                objects_factors=item_embs,
            )
        else:
            user_embs_np = user_embs.detach().cpu().numpy()
            item_embs_np = item_embs.detach().cpu().numpy()
            del user_embs, item_embs
            torch.cuda.empty_cache()
            ranker = ImplicitRanker(
                Distance.DOT,
                user_embs_np[user_ids],  # [n_rec_users, n_factors]
                item_embs_np,  # [n_items + n_item_extra_tokens, n_factors]
                num_threads=recommend_n_threads,
                use_gpu=False,
            )

        user_ids_indices, all_reco_ids, all_scores = ranker.rank(
            subject_ids=np.arange(len(user_ids)),  # n_rec_users
            k=k,
            filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + n_item_extra_tokens]
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model_internal
        )
        all_user_ids = user_ids[user_ids_indices]
        return all_user_ids, all_reco_ids, all_scores

    def _recommend_i2i(
        self,
        target_ids: InternalIdsArray,
        sorted_item_ids_to_recommend: InternalIdsArray,
        k: int,
        recommend_use_torch_ranking: bool,
        recommend_n_threads: int,
        recommend_device: tp.Optional[str],
    ) -> InternalRecoTriplet:
        """Recommend to items."""
        self._prepare_for_inference(recommend_device)
        with torch.no_grad():
            item_embs = self.torch_model.item_model.get_all_embeddings()

        ranker: Ranker
        if recommend_use_torch_ranking:
            ranker = TorchRanker(
                distance=self.i2i_dist, device=item_embs.device, subjects_factors=item_embs, objects_factors=item_embs
            )
        else:
            item_embs_np = item_embs.detach().cpu().numpy()
            del item_embs
            ranker = ImplicitRanker(
                self.i2i_dist,
                item_embs_np,  # [n_items + n_item_extra_tokens, n_factors]
                item_embs_np,  # [n_items + n_item_extra_tokens, n_factors]
                num_threads=recommend_n_threads,
                use_gpu=False,
            )

        torch.cuda.empty_cache()
        return ranker.rank(
            subject_ids=target_ids,  # model internal
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model internal
        )
