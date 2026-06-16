import os
import typing as tp
from collections import Counter, defaultdict
from datetime import datetime
from os import PathLike
from os.path import join
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from rectools.semantic import OptimizerType, QuantizerType

from .base_quantizer import Quantizer
from .emb_dataset import EmbDataset
from .rkmeans import RKmeans
from .rqvae import RQVAE


class SIDTokenizer:
    """A class implementing semantic ID tokenizer, which can use either RK-Means or RQ-VAE as its backbone.

    The tokenizer owns the quantizer lifecycle: it constructs the appropriate
    ``Quantizer`` from configuration parameters, delegates training-related
    calls (``init_codebooks``, forward pass) to it, and maintains the
    ``id2sid`` / ``sid2id`` vocabulary mappings.

    Parameters
    ----------
    input_dim : int
        Dimensionality of incoming item embeddings.
    codebook_sizes : tp.List[int]
        Number of codes per codebook level.
    codebook_dim : int
        Dimensionality of codebook embeddings (used by RQ-VAE;
        ignored for RK-Means, where codebook dim equals ``input_dim``).
    hidden_dims : tp.List[int] | None
        Hidden layer sizes for the RQ-VAE encoder/decoder.
        Pass ``None`` for RK-Means.
    quantizer : QuantizerType
        Which quantizer backend to use: ``"rkmeans"`` or ``"rqvae"``.
    device : tp.Optional[str], optional
        Device for quantizer parameters, by default None.
    adapter_proj_dim : int, optional
        Number of principal components for the RQ-VAE whitening adapter,
        by default 512. Ignored for RK-Means.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_sizes: tp.List[int],
        codebook_dim: int = 32,
        hidden_dims: tp.Optional[tp.List[int]] = None,
        quantizer: QuantizerType = "rkmeans",
        device: tp.Optional[str] = None,
        adapter_proj_dim: int = 512,
    ) -> None:
        self.input_dim = input_dim
        self.codebook_sizes = list(codebook_sizes)
        self.codebook_dim = codebook_dim
        self.hidden_dims = hidden_dims
        self.quantizer_name: QuantizerType = quantizer
        self.device = device
        self.adapter_proj_dim = adapter_proj_dim

        self._quantizer = self._build_quantizer()
        self._quantizer.eval()

        # SIDs are tuples because they are hashable -> it is easier to check for conflicts
        # since there can be conflicting SIDs, sid2id maps to a list
        self.id2sid: tp.Dict[int, tp.Tuple[int, ...]] = {}
        self.sid2id: tp.Dict[tp.Tuple[int, ...], tp.List[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Quantizer construction
    # ------------------------------------------------------------------

    def _build_quantizer(self) -> Quantizer:
        """Construct the quantizer from stored configuration and move it to ``self.device``."""
        quantizer: Quantizer
        if self.quantizer_name == "rqvae":
            quantizer = RQVAE(
                input_dim=self.input_dim,
                codebook_sizes=self.codebook_sizes,
                hidden_dims=self.hidden_dims if self.hidden_dims is not None else [],
                codebook_dim=self.codebook_dim,
                adapter_proj_dim=self.adapter_proj_dim,
            )
        else:
            quantizer = RKmeans(
                input_dim=self.input_dim,
                codebook_sizes=self.codebook_sizes,
            )
        return quantizer.to(self.device)

    # ------------------------------------------------------------------
    # Training helpers (delegated to the quantizer)
    # ------------------------------------------------------------------

    @property
    def quantizer(self) -> Quantizer:
        """Access the underlying quantizer module."""
        return self._quantizer

    def init_codebooks(self, dataset: EmbDataset) -> None:
        """Initialize quantizer codebooks from an ``EmbDataset``.

        Parameters
        ----------
        dataset : EmbDataset
            Dataset whose embeddings are used for codebook initialization.
        """
        embs = torch.Tensor(dataset[:]["embed"]).to(self.device)
        self._quantizer.init_codebooks(embs)

    def __call__(self, batch: tp.Dict[str, tp.Any]) -> torch.Tensor:
        """Run a forward pass through the quantizer on a batch and update ``id2sid``.

        Parameters
        ----------
        batch : tp.Dict[str, tp.Any]
            A dict with ``"item_id"`` and ``"embed"`` keys, as returned by
            ``EmbDataset.__getitem__``.

        Returns
        -------
        torch.Tensor
            Scalar loss from the quantizer.
        """
        item_ids = batch["item_id"]
        if isinstance(item_ids, int):
            item_ids = [item_ids]

        embs = torch.Tensor(batch["embed"]).to(self.device)
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)

        out = self._quantizer(embs)

        for idx, item in enumerate(item_ids):
            self.id2sid[item] = out.sem_ids[idx]

        return out.loss

    def train(self) -> "SIDTokenizer":
        """Set the quantizer to training mode."""
        self._quantizer.train()
        return self

    def eval(self) -> "SIDTokenizer":
        """Set the quantizer to evaluation mode."""
        self._quantizer.eval()
        return self

    def parameters(self) -> tp.Any:
        """Return quantizer parameters (for optimizer construction)."""
        return self._quantizer.parameters()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @staticmethod
    def _get_collision_rate(id2sid: tp.Dict[int, tp.Tuple[int, ...]]) -> float:
        """Fraction of items whose SID collides with at least one other item."""
        sid_counts = Counter(id2sid.values())
        n_colliding = sum(c for c in sid_counts.values() if c > 1)
        return n_colliding / len(id2sid) if id2sid else 1.0

    @torch.no_grad()
    def _get_id2sid(self, dataset: EmbDataset, batch_size: int = 2048) -> tp.Dict[int, tp.Tuple[int, ...]]:
        """Run the quantizer over ``dataset`` in eval mode and return an id2sid mapping."""
        was_training = self._quantizer.training
        self._quantizer.eval()
        id2sid: tp.Dict[int, tp.Tuple[int, ...]] = {}
        for start_idx in range(0, len(dataset), batch_size):
            end_idx = min(len(dataset), start_idx + batch_size)
            batch = dataset[start_idx:end_idx]
            batch_embeds = torch.Tensor(batch["embed"]).to(self.device)
            out = self._quantizer(batch_embeds)
            for idx, item in enumerate(batch["item_id"]):
                id2sid[item] = out.sem_ids[idx]
        if was_training:
            self._quantizer.train()
        return id2sid

    def fit(  # pylint: disable=too-many-locals
        self,
        item_ids: tp.List[int],
        embeddings: np.ndarray,
        init_max_items: int = 10_000,
        max_epochs: int = 1_000,
        patience: int = 100,
        optimizer: OptimizerType = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        save_dir: str = "checkpoints",
    ) -> "SIDTokenizer":
        """Train the quantizer on pre-computed item embeddings.

        After training, ``id2sid`` is populated for every item in ``item_ids``.

        Parameters
        ----------
        item_ids : tp.List[int]
            Item identifiers matching rows of ``embeddings``.
        embeddings : np.ndarray
            Embedding matrix of shape ``(len(item_ids), embed_dim)``.
        init_max_items : int, optional
            Maximum number of items used for codebook initialization,
            by default 10_000.
        max_epochs : int, optional
            Maximum training epochs, by default 1_000.
        patience : int, optional
            Early stopping patience (epochs without collision rate
            improvement), by default 100.
        optimizer : OptimizerType, optional
            Optimizer name, by default ``"adamw"``.
        lr : float, optional
            Learning rate, by default 1e-3.
        weight_decay : float, optional
            Weight decay, by default 1e-4.
        batch_size : int, optional
            Training batch size, by default 2048.
        save_dir : str, optional
            Directory for saving best checkpoint, by default ``"checkpoints"``.

        Returns
        -------
        SIDTokenizer
            ``self``, with trained quantizer and populated ``id2sid``.
        """
        dataset = EmbDataset(item_ids, embeddings)

        # Initialize codebooks from a (possibly smaller) subset
        init_dataset = EmbDataset(item_ids[:init_max_items], embeddings[:init_max_items])
        self.init_codebooks(init_dataset)

        opt: torch.optim.Optimizer
        if optimizer.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == "adagrad":
            opt = torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_collision_rate = float("inf")
        steps_no_improve = 0
        best_path = Path(
            join(
                save_dir,
                f"tokenizer_{self.quantizer_name}",
                f"{datetime.now().strftime('%d%b%Y-%H.%M.%S.%f')}",
                "best_perf.pt",
            )
        )
        os.makedirs(best_path.parent, exist_ok=True)

        self.train()
        with tqdm(
            total=max_epochs * (len(dataset) // batch_size + (len(dataset) % batch_size > 0)),
        ) as pbar:
            for epoch_no in range(max_epochs):
                pbar.set_description(f"Tokenizer train {epoch_no} / {max_epochs}")
                for start_idx in range(0, len(dataset), batch_size):
                    end_idx = min(start_idx + batch_size, len(dataset))
                    batch = dataset[start_idx:end_idx]
                    batch_embeds = torch.Tensor(batch["embed"]).to(self.device)
                    out = self._quantizer(batch_embeds)
                    loss = out.loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    metrics_dict = {"loss": loss.item()}
                    pbar.set_postfix({metric: f"{value:.4f}" for metric, value in metrics_dict.items()})
                    pbar.update(1)

                # Compute collision rate
                id2sid = self._get_id2sid(dataset, batch_size=batch_size)
                collision_rate = self._get_collision_rate(id2sid)

                improved = collision_rate < best_collision_rate
                if improved:
                    best_collision_rate = collision_rate
                    steps_no_improve = 0
                    torch.save(self._quantizer.state_dict(), best_path)
                else:
                    steps_no_improve += 1

                self.train()

                if steps_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch_no} (patience exceeded {patience})")
                    break

        print(f"Best model's collision rate: {best_collision_rate * 100:.2f}%")
        self._quantizer.load_state_dict(torch.load(best_path, map_location=self.device))
        self.eval()
        self.id2sid = self._get_id2sid(dataset, batch_size=batch_size)
        return self

    # ------------------------------------------------------------------
    # Vocabulary extension
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extend(self, dataset: EmbDataset) -> None:
        """Tokenize new items from an EmbDataset and add them to the vocabulary.

        Items already present in id2sid are skipped.
        """
        for start_idx in range(0, len(dataset), 256):
            end_idx = min(start_idx + 256, len(dataset))
            batch = dataset[start_idx:end_idx]

            item_ids = batch["item_id"]
            if isinstance(item_ids, int):
                item_ids = [item_ids]

            new_mask = [i for i, item in enumerate(item_ids) if item not in self.id2sid]
            if not new_mask:
                continue

            new_embs = torch.Tensor(batch["embed"]).to(self.device)
            if new_embs.dim() == 1:
                new_embs = new_embs.unsqueeze(0)
            new_embs = new_embs[new_mask]

            out = self._quantizer(new_embs)
            sids = out.sem_ids
            for j, mask_idx in enumerate(new_mask):
                item = item_ids[mask_idx]
                sid = sids[j]
                self.id2sid[item] = sid
                if self.sid2id:
                    self.sid2id[sid].append(item)

    # ------------------------------------------------------------------
    # Tokenize / decode
    # ------------------------------------------------------------------

    def _get_sid(self, item: int) -> tp.Tuple[int, ...]:
        if item in self.id2sid:
            return self.id2sid[item]
        raise ValueError(
            f"Item with ID {item} is out of vocabulary. " "Use extend() with an EmbDataset containing this item first."
        )

    def _get_id(
        self,
        sid: tp.Tuple[int, ...],
        default_value: tp.Optional[int] = None,
        rng: tp.Optional[np.random.RandomState] = None,
    ) -> tp.Optional[int]:
        if len(self.sid2id) == 0:
            for item_id, item_sid in self.id2sid.items():
                self.sid2id[item_sid].append(item_id)
        if sid in self.sid2id:
            if rng is None:
                rng = np.random
            return int(rng.choice(self.sid2id[sid]))
        return default_value

    def tokenize(  # pylint: disable=redefined-builtin
        self, input: int | tp.Iterable[int]
    ) -> tp.Tuple[int, ...] | tp.List[tp.Tuple[int, ...]]:
        """Convert incoming item IDs to corresponding semantic IDs.

        Parameters
        ----------
        input : int | tp.Iterable[int]
            Item ID(s).

        Returns
        -------
        tp.Tuple[int, ...] | tp.List[tp.Tuple[int, ...]]
            Item SID(s).
        """
        if isinstance(input, int):
            return self._get_sid(input)

        output = []
        for item in input:
            output.append(self._get_sid(item))
        return output

    def decode(  # pylint: disable=redefined-builtin
        self,
        input: tp.Tuple[int, ...] | tp.Iterable[tp.Tuple[int, ...]],
        default_value: tp.Optional[int] = None,
        rng: tp.Optional[np.random.RandomState] = None,
    ) -> tp.Optional[int] | tp.List[tp.Optional[int]]:
        """Decode incoming SID(s) to corresponding item IDs.

        Parameters
        ----------
        input : tp.Tuple[int, ...] | tp.Iterable[tp.Tuple[int, ...]]
            Either a single SID or an iterable of SIDs.
        default_value : tp.Optional[int], optional
            What to return when there is no item corresponding to given SID, by default None
        rng : np.random.RandomState, optional
            Random generator used to resolve SID collisions. If omitted, the
            global NumPy random generator is used.

        Returns
        -------
        tp.Optional[int] | tp.List[tp.Optional[int]]
            Item ID(s) corresponding to input SID(s).
        """
        # check if the input is only a single SID
        if isinstance(input, tuple) and all(map(lambda x: isinstance(x, int), input)):
            return self._get_id(input, default_value=default_value, rng=rng)
        # type: ignore[arg-type]
        return list(map(lambda item: self._get_id(item, default_value=default_value, rng=rng), input))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: tp.Union[str, PathLike[str]]) -> None:
        """Save tokenizer to the specified path.

        The checkpoint is a dict containing quantizer configuration,
        quantizer ``state_dict``, and the ``id2sid`` vocabulary mapping.

        Parameters
        ----------
        path : str or PathLike
            File path to write the checkpoint to.
        """
        checkpoint = {
            "quantizer_name": self.quantizer_name,
            "input_dim": self.input_dim,
            "codebook_sizes": self.codebook_sizes,
            "codebook_dim": self.codebook_dim,
            "hidden_dims": self.hidden_dims,
            "adapter_proj_dim": self.adapter_proj_dim,
            "quantizer_state_dict": self._quantizer.state_dict(),
            "id2sid": self.id2sid,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: tp.Union[str, PathLike[str]],
        device: tp.Optional[str] = None,
        map_location: tp.Optional[str] = None,
    ) -> "SIDTokenizer":
        """Load serialized SIDTokenizer.

        Parameters
        ----------
        path : str or PathLike
            Location of the serialized tokenizer.
        device : tp.Optional[str], optional
            Device to load tokenizer's quantizer to, by default None.
        map_location : tp.Optional[str], optional
            ``map_location`` forwarded to ``torch.load``. If provided,
            also used as the device. By default None.

        Returns
        -------
        SIDTokenizer
            A de-serialized tokenizer.
        """
        effective_device = map_location or device
        checkpoint = torch.load(path, map_location=effective_device, weights_only=False)

        tokenizer = cls(
            input_dim=checkpoint["input_dim"],
            codebook_sizes=checkpoint["codebook_sizes"],
            codebook_dim=checkpoint["codebook_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            quantizer=checkpoint["quantizer_name"],
            device=effective_device,
            adapter_proj_dim=checkpoint["adapter_proj_dim"],
        )
        tokenizer._quantizer.load_state_dict(checkpoint["quantizer_state_dict"])
        tokenizer._quantizer.eval()
        tokenizer.id2sid = checkpoint["id2sid"]

        return tokenizer

    def __len__(self) -> int:
        return len(set(self.id2sid.values()))
