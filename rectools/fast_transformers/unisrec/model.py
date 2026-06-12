#  Copyright 2026 MTS (Mobile Telesystems)
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

"""UniSRecModel: standalone sequential recommender with pretrained text embeddings."""

import logging
import typing as tp
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader

from ..preprocessing import SequenceBatchDataset, align_embeddings, build_sequences
from .lightning import SUPPORTED_LOSSES, SUPPORTED_OPTIMIZERS, SUPPORTED_SCHEDULERS, UniSRecLightning
from .net import UniSRecNet

logger = logging.getLogger(__name__)

# Keys saved in checkpoint for architecture validation
_ARCH_KEYS = (
    "n_factors",
    "projection_hidden",
    "n_blocks",
    "n_heads",
    "session_max_len",
    "dropout",
    "adaptor_dropout",
    "adaptor_type",
    "use_adaptor_ffn",
    "ffn_type",
    "ffn_expansion",
)


class _NegativeSampler:
    """Add ``negatives`` field to a batch, avoiding positive collisions."""

    def __init__(self, n_items: int, n_negatives: int) -> None:
        self.n_items = n_items
        self.n_negatives = n_negatives

    def __call__(self, batch: tp.Dict[str, torch.Tensor]) -> tp.Dict[str, torch.Tensor]:
        y = batch["y"]
        negs = torch.randint(1, self.n_items + 1, (*y.shape, self.n_negatives), device=y.device)
        # Resample positions where negative == positive
        collisions = negs == y.unsqueeze(-1)
        if collisions.any():
            negs[collisions] = torch.randint(1, self.n_items + 1, (int(collisions.sum()),), device=y.device)
        batch["negatives"] = negs
        return batch


class _ProjectAllWrapper(torch.nn.Module):
    def __init__(self, net: UniSRecNet) -> None:
        super().__init__()
        self.net = net

    def forward(self) -> torch.Tensor:
        return self.net.project_all()


class UniSRecModel:  # pylint: disable=too-many-instance-attributes
    """UniSRec sequential recommender with pretrained text embeddings.

    Joint training of the adaptor and transformer encoder on
    frozen pretrained embeddings (e.g. from a sentence-transformer).

    Parameters
    ----------
    pretrained_item_embeddings : Tensor
        Shape ``(max_external_item_id + 1, D_text)`` or
        ``(max_external_item_id + 1, n_variants, D_text)``.
        Index *i* holds the text embedding for the item whose **external** ID
        equals *i*.  Index 0 is padding (zeros).
    patience : int, optional
        Number of epochs with no improvement on validation loss before
        stopping training.  **Note:** validation is currently computed on
        the training data (last-item prediction on the same sequences),
        so ``patience`` monitors training stability rather than true
        generalization.  Pass an explicit held-out split via a future API
        to get real early stopping.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        pretrained_item_embeddings: torch.Tensor,
        # architecture
        n_factors: int = 256,
        projection_hidden: int = 512,
        n_blocks: int = 2,
        n_heads: int = 1,
        session_max_len: int = 200,
        dropout: float = 0.1,
        adaptor_dropout: float = 0.2,
        adaptor_type: str = "pca",
        use_adaptor_ffn: bool = True,
        ffn_type: str = "conv1d",
        ffn_expansion: int = 1,
        # training
        epochs: int = 10,
        lr: float = 1e-4,
        lr_head: float = 0.3,
        lr_wp: float = 0.1,
        lr_transformer: float = 3.0,
        # optimizer / scheduler
        optimizer: str = "adamw",
        scheduler: tp.Optional[str] = None,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        grad_clip: float = 1.0,
        weight_decay: float = 0.01,
        # loss
        loss: str = "softmax",
        gbce_t: float = 0.2,
        n_negatives: tp.Optional[int] = None,
        # early stopping
        patience: tp.Optional[int] = None,
        # data
        batch_size: int = 128,
        dataloader_num_workers: int = 0,
        train_min_user_interactions: int = 2,
        device: tp.Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss '{loss}'. Choose from {SUPPORTED_LOSSES}")
        if loss in ("BCE", "gBCE", "sampled_softmax"):
            if not isinstance(n_negatives, int) or n_negatives <= 0:
                raise ValueError(f"Loss '{loss}' requires n_negatives to be a positive integer")
        if optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose from {SUPPORTED_OPTIMIZERS}")
        if scheduler not in SUPPORTED_SCHEDULERS:
            raise ValueError(f"Unsupported scheduler '{scheduler}'. Choose from {SUPPORTED_SCHEDULERS}")

        self.pretrained_item_embeddings = pretrained_item_embeddings
        self.n_factors = n_factors
        self.projection_hidden = projection_hidden
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.session_max_len = session_max_len
        self.dropout = dropout
        self.adaptor_dropout = adaptor_dropout
        self.adaptor_type = adaptor_type
        self.use_adaptor_ffn = use_adaptor_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion = ffn_expansion
        self.epochs = epochs
        self.lr = lr
        self.lr_head = lr_head
        self.lr_wp = lr_wp
        self.lr_transformer = lr_transformer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.loss = loss
        self.gbce_t = gbce_t
        self.n_negatives = n_negatives
        self.patience = patience
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions
        self.device = device
        self.verbose = verbose

        self._net: tp.Optional[UniSRecNet] = None
        self._unique_items: tp.Optional[torch.Tensor] = None
        self._unique_users: tp.Optional[torch.Tensor] = None
        self.is_fitted: bool = False

    # ── helpers ──

    def _arch_kwargs(self) -> tp.Dict[str, tp.Any]:
        """Return architecture hyperparameters as a dict."""
        return {k: getattr(self, k) for k in _ARCH_KEYS}

    def _make_trainer(self, max_epochs: int, val_dl: tp.Any = None) -> pl.Trainer:
        """Create a PyTorch Lightning Trainer."""
        callbacks: tp.List[Callback] = []
        if self.patience is not None and val_dl is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=self.patience, mode="min"))

        return pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=self.grad_clip,
            callbacks=callbacks or None,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=self.verbose > 0,
            enable_progress_bar=self.verbose > 0,
        )

    def _make_lightning(
        self,
        net: UniSRecNet,
        param_groups: tp.List[tp.Dict],
        max_epochs: int,
        train_dl: tp.Any,
    ) -> UniSRecLightning:
        """Build the Lightning wrapper for training."""
        total_steps = len(train_dl) * max_epochs if self.scheduler else None
        return UniSRecLightning(
            net=net,
            param_groups=param_groups,
            loss=self.loss,
            n_negatives=self.n_negatives,
            gbce_t=self.gbce_t,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
            total_steps=total_steps,
        )

    # ── param groups ──

    def _param_groups(self, net: UniSRecNet) -> tp.List[tp.Dict[str, tp.Any]]:
        """Build per-layer parameter groups with individual learning rates."""
        if self.adaptor_type == "pca":
            adaptor: tp.List[tp.Dict[str, tp.Any]] = [
                {"params": [net.whitening_proj], "lr": self.lr * self.lr_wp, "weight_decay": 0.0},
                {"params": [net.whitening_bias], "lr": self.lr * 10.0, "weight_decay": 0.0},
            ]
        else:
            adaptor = [
                {"params": list(net.bn_input.parameters()), "lr": self.lr, "weight_decay": 0.0},
                {"params": list(net.bn_score.parameters()), "lr": self.lr, "weight_decay": 0.0},
            ]
        head: tp.List[tp.Dict[str, tp.Any]] = []
        if net.head is not None:
            head = [
                {
                    "params": list(net.head.parameters()),
                    "lr": self.lr * self.lr_head,
                    "weight_decay": self.weight_decay,
                }
            ]
        transformer = [
            {"params": list(net.pos_emb.parameters()), "lr": self.lr * self.lr_transformer, "weight_decay": 0.0},
            {
                "params": (
                    [p for layer in net.attention_layers for p in layer.parameters()]
                    + [p for layer in net.forward_layers for p in layer.parameters()]
                ),
                "lr": self.lr * self.lr_transformer,
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    [p for layer in net.attention_layernorms for p in layer.parameters()]
                    + [p for layer in net.forward_layernorms for p in layer.parameters()]
                    + list(net.last_layernorm.parameters())
                ),
                "lr": self.lr,
                "weight_decay": 0.0,
            },
        ]
        return adaptor + head + transformer

    # ── fit ──

    def fit(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> "UniSRecModel":
        """Train the model on interaction data.

        Parameters
        ----------
        user_ids : LongTensor (N,)
            External user IDs for each interaction.
        item_ids : LongTensor (N,)
            External item IDs for each interaction.
        timestamps : LongTensor (N,)
            Timestamps (any monotonic int64 values).

        Returns
        -------
        self
        """
        if self.patience is not None:
            warnings.warn(
                "Early stopping currently monitors loss on the training data "
                "(last-item prediction on the same sequences), not on a "
                "held-out validation set. `patience` therefore tracks training "
                "stability rather than true generalization.",
                UserWarning,
                stacklevel=2,
            )

        x, y, unique_items, unique_users = build_sequences(
            user_ids,
            item_ids,
            timestamps,
            max_len=self.session_max_len,
            min_interactions=self.train_min_user_interactions,
            device=self.device,
        )
        if len(x) == 0:
            raise ValueError(
                f"No users with >= {self.train_min_user_interactions} interactions. " "Cannot train on empty data."
            )
        self._unique_items = unique_items.cpu()
        self._unique_users = unique_users.cpu()
        n_items = len(unique_items)

        aligned_emb = align_embeddings(self.pretrained_item_embeddings, unique_items, n_items)

        net = UniSRecNet(
            n_items=n_items,
            pretrained_embeddings=aligned_emb,
            n_factors=self.n_factors,
            projection_hidden=self.projection_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            session_max_len=self.session_max_len,
            dropout=self.dropout,
            adaptor_dropout=self.adaptor_dropout,
            adaptor_type=self.adaptor_type,
            use_adaptor_ffn=self.use_adaptor_ffn,
            ffn_type=self.ffn_type,
            ffn_expansion=self.ffn_expansion,
        )

        # DataLoader with num_workers>0 requires CPU tensors
        if x.is_cuda:
            x, y = x.cpu(), y.cpu()

        neg_transform = None
        if self.loss in ("BCE", "gBCE", "sampled_softmax"):
            assert self.n_negatives is not None  # validated in __init__
            neg_transform = _NegativeSampler(n_items, self.n_negatives)

        train_dl = DataLoader(
            SequenceBatchDataset(x, y, transform=neg_transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

        val_dl = None
        if self.patience is not None:
            val_y_last = y[:, -1:]
            val_dl = DataLoader(
                SequenceBatchDataset(x, val_y_last, transform=neg_transform),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.dataloader_num_workers,
            )

        lm = self._make_lightning(net, self._param_groups(net), self.epochs, train_dl)
        trainer = self._make_trainer(self.epochs, val_dl)
        trainer.fit(lm, train_dl, val_dl)

        self._net = net
        self.is_fitted = True
        return self

    # ── save / load ──

    def save_checkpoint(self, path: tp.Union[str, Path]) -> None:
        """Save model weights, ID mappings, and architecture hyperparameters.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        assert self._net is not None and self._unique_items is not None
        torch.save(
            {
                "net": self._net.state_dict(),
                "unique_items": self._unique_items,
                "unique_users": self._unique_users,
                "n_items": len(self._unique_items),
                "arch": self._arch_kwargs(),
            },
            path,
        )

    def load_checkpoint(self, path: tp.Union[str, Path], device: tp.Optional[str] = None) -> None:
        """Load model weights from a checkpoint.

        Parameters
        ----------
        path : str or Path
            Checkpoint file path (created by :meth:`save_checkpoint`).
        device : str, optional
            Device to load the model onto.  Defaults to CUDA if available.

        Raises
        ------
        ValueError
            If the checkpoint was saved with different architecture
            hyperparameters than this model instance.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self._unique_items = ckpt["unique_items"].cpu()
        self._unique_users = ckpt["unique_users"].cpu()
        n_items = ckpt["n_items"]

        # Validate architecture hyperparameters if present
        saved_arch = ckpt.get("arch")
        if saved_arch is not None:
            current_arch = self._arch_kwargs()
            mismatches = {
                k: (saved_arch[k], current_arch[k])
                for k in _ARCH_KEYS
                if k in saved_arch and saved_arch[k] != current_arch[k]
            }
            if mismatches:
                msg = "; ".join(f"{k}: saved={sv}, current={cv}" for k, (sv, cv) in mismatches.items())
                raise ValueError(f"Architecture mismatch between checkpoint and model: {msg}")

        assert self._unique_items is not None
        aligned_emb = align_embeddings(self.pretrained_item_embeddings, self._unique_items, n_items)

        self._net = UniSRecNet(
            n_items=n_items,
            pretrained_embeddings=aligned_emb,
            n_factors=self.n_factors,
            projection_hidden=self.projection_hidden,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            session_max_len=self.session_max_len,
            dropout=self.dropout,
            adaptor_dropout=self.adaptor_dropout,
            adaptor_type=self.adaptor_type,
            use_adaptor_ffn=self.use_adaptor_ffn,
            ffn_type=self.ffn_type,
            ffn_expansion=self.ffn_expansion,
        )
        self._net.load_state_dict(ckpt["net"])
        self._net.to(device).eval()
        self.is_fitted = True

    # ── ONNX export ──

    def export_to_onnx(
        self,
        encoder_path: tp.Union[str, Path],
        items_path: tp.Optional[tp.Union[str, Path]] = None,
        opset_version: int = 18,
    ) -> None:
        """Export the model to ONNX.

        Parameters
        ----------
        encoder_path : str or Path
            Path for the encoder graph (input_ids -> hidden states).
        items_path : str or Path, optional
            If given, also exports project_all (-> item embeddings).
        opset_version : int
            ONNX opset version (default 18).
        """
        assert self._net is not None, "Model not fitted or loaded"
        net = self._net
        was_training = net.training
        net.eval()

        try:
            device = next(net.parameters()).device
            dummy = torch.zeros(1, 5, dtype=torch.long, device=device)

            torch.onnx.export(
                net,
                (dummy,),
                str(encoder_path),
                input_names=["input_ids"],
                output_names=["hidden"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "hidden": {0: "batch", 1: "seq"}},
                opset_version=opset_version,
            )

            if items_path is not None:
                wrapper = _ProjectAllWrapper(net)
                wrapper.eval()
                torch.onnx.export(
                    wrapper,
                    (),
                    str(items_path),
                    input_names=[],
                    output_names=["item_embs"],
                    opset_version=opset_version,
                )
        finally:
            if was_training:
                net.train()

    def map_item_ids(self, external_ids: torch.Tensor) -> torch.Tensor:
        """Map external item IDs to internal IDs used by the model.

        Parameters
        ----------
        external_ids : LongTensor
            External item IDs.

        Returns
        -------
        LongTensor
            Internal IDs in ``[0, n_items]``.  0 means unknown item.
        """
        assert self._unique_items is not None, "Model not fitted or loaded"
        input_device = external_ids.device
        external_cpu = external_ids.cpu()
        sorted_items, sort_idx = self._unique_items.sort()
        pos = torch.searchsorted(sorted_items, external_cpu)
        pos = pos.clamp(max=len(sorted_items) - 1)
        found = sorted_items[pos] == external_cpu
        result = torch.zeros_like(external_cpu, dtype=torch.long)
        result[found] = sort_idx[pos[found]] + 1
        return result.to(input_device)

    def _internal_to_external(self, internal_ids: torch.Tensor) -> torch.Tensor:
        """Map internal item IDs (1-based) back to external IDs."""
        assert self._unique_items is not None, "Model not fitted or loaded"
        # internal_ids are 1-based; index 0 maps to 0 (unknown)
        mapping = torch.zeros(len(self._unique_items) + 1, dtype=torch.long)
        mapping[1:] = self._unique_items
        return mapping[internal_ids.cpu()].to(internal_ids.device)

    def recommend(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:  # noqa: D102
        """Not supported. Use :meth:`predict_topk` instead.

        ``UniSRecModel`` operates on raw tensor sequences, not on
        ``Dataset`` / user IDs expected by ``ModelBase.recommend()``.
        Keeping the same name with a different signature would silently
        break code that relies on the RecTools ``recommend`` contract.
        """
        raise NotImplementedError(
            "UniSRecModel does not implement recommend(). "
            "Use predict_topk(input_ids, k) instead — it accepts "
            "left-padded internal ID sequences and returns (scores, item_ids) tensors."
        )

    @torch.no_grad()
    def predict_topk(
        self,
        input_ids: torch.Tensor,
        k: int = 10,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Encode user sequences and return top-k items in a single GPU pass.

        This is the inference entry point for ``UniSRecModel``.  It fuses
        sequence encoding and dot-product ranking into one call, keeping
        everything on GPU without intermediate numpy / scipy conversions.

        Parameters
        ----------
        input_ids : LongTensor (B, L)
            Left-padded internal item ID sequences (0 = padding).
            Use :meth:`map_item_ids` to convert external IDs to internal.
        k : int
            Number of items to return per user.

        Returns
        -------
        scores : Tensor (B, k)
            Dot-product scores, descending.
        item_ids : LongTensor (B, k)
            **External** item IDs corresponding to the top-k scores.
        """
        assert self._net is not None, "Model not fitted or loaded"
        net = self._net
        was_training = net.training
        net.eval()
        try:
            device = next(net.parameters()).device
            h = net.encode_last(input_ids.to(device))
            item_embs = net.project_all()
            scores = h @ item_embs.T
            scores[:, 0] = float("-inf")
            top_scores, top_ids = scores.topk(k, dim=1)
            top_ids = self._internal_to_external(top_ids)
        finally:
            if was_training:
                net.train()
        return top_scores, top_ids

    @property
    def net(self) -> UniSRecNet:
        """Return the underlying network. Raises if model is not fitted."""
        assert self._net is not None, "Model not fitted or loaded"
        return self._net

    @property
    def item_id_mapping(self) -> tp.Optional[torch.Tensor]:
        """Return the external item ID mapping tensor, or None if not fitted."""
        return self._unique_items
