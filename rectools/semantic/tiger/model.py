import json
import os
import typing as tp

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from tqdm.auto import trange

from rectools.semantic import LRScheduleType, OptimizerType
from rectools.semantic.data_handling import PaddingCollateFn, TIGERDataset
from rectools.semantic.tokenizer import SIDTokenizer

from .lightning import TIGERLightning
from .module import TIGERNet


class TIGERModel:  # pylint: disable=too-many-instance-attributes
    """TIGER generative recommender model with semantic IDs.

    Given a pretrained SIDTokenizer, trains a TIGER generative recommender model.

    Parameters
    ----------
    tokenizer : SIDTokenizer
        Pretrained semantic ID tokenizer.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        tokenizer: SIDTokenizer,
        # TIGER model hyperparams
        hidden_units: int = 256,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        max_length: int = 200,
        ff_dim: tp.Optional[int] = None,
        d_kv: tp.Optional[int] = None,
        # Training hyperparams
        optimizer: OptimizerType = "adamw",
        lr: float = 1e-3,
        lr_schedule: tp.Optional[LRScheduleType] = None,
        warmup_steps: int = 300,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: tp.Optional[int] = 10,
        batch_size: int = 256,
        eval_batch_size: int = 64,
        num_workers: int = 4,
        beam_size: int = 20,
        top_k: int = 10,
        train_only_last: bool = True,
        random_seed: tp.Optional[int] = None,
        # General
        device: tp.Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = tokenizer
        self.codebook_sizes = list(tokenizer.quantizer.codebook_sizes)

        # Training hyperparams
        self.optimizer = optimizer
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.beam_size = beam_size
        self.top_k = top_k
        self.train_only_last = train_only_last
        self.random_seed = random_seed
        self._decode_rng = np.random.RandomState(random_seed) if random_seed is not None else None

        # TIGER model hyperparams
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.ff_dim = ff_dim
        self.d_kv = d_kv

        self.model = TIGERNet(
            codebook_sizes=self.codebook_sizes,
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_length=max_length,
            ff_dim=ff_dim,
            d_kv=d_kv,
        ).to(self.device)

    def _tiger_dataset_kwargs(self) -> dict:
        return {
            "tokenizer": self.tokenizer,
            "codebook_sizes": self.codebook_sizes,
            "max_length": self.max_length,
            "codeword_offset": TIGERNet.CODEWORD_OFFSET,
            "bos_token_id": TIGERNet.BOS_TOKEN_ID,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        monitor: tp.Literal["Hit", "MRR", "NDCG"] = "Hit",
    ) -> None:
        """Train the TIGER model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions (user_id, item_id, optionally timestamp).
        val_df : pd.DataFrame
            Validation interactions (same schema).

        Notes
        -----
        If multiple items share the same Semantic ID, decoding during
        validation remains stochastic. Set ``random_seed`` at model
        initialization to make this collision resolution reproducible.
        """
        num_items = len(set(train_df["item_id"]).union(val_df["item_id"]))

        ds_kwargs = self._tiger_dataset_kwargs()

        train_dataset = TIGERDataset(
            interactions=train_df,
            only_last=self.train_only_last,
            eval_mode=False,
            **ds_kwargs,
        )

        val_dataset = TIGERDataset(
            interactions=val_df,
            only_last=True,
            eval_mode=True,
            **ds_kwargs,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=PaddingCollateFn(padding_value=TIGERNet.PAD_TOKEN_ID),
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=PaddingCollateFn(padding_value=TIGERNet.PAD_TOKEN_ID),
            pin_memory=True,
        )

        lightning_model = TIGERLightning(
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer_name=self.optimizer,
            lr=self.lr,
            lr_schedule=self.lr_schedule,
            warmup_steps=self.warmup_steps,
            max_iters=self.max_epochs * len(train_loader) - self.warmup_steps,
            weight_decay=self.weight_decay,
            num_items=num_items,
            beam_size=self.beam_size,
            top_k=self.top_k,
        )

        callbacks = [LearningRateMonitor(logging_interval="step"), ModelSummary()]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor=f"val_{monitor}@{self.top_k}",
                    patience=self.patience,
                    mode="max",
                )
            )

        trainer = pl.Trainer(max_epochs=self.max_epochs, callbacks=callbacks)
        trainer.fit(
            lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        self.model = lightning_model.model.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        interactions: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Generate top-k recommendations for each user.

        Parameters
        ----------
        interactions : pd.DataFrame
            User interaction histories (user_id, item_id, optionally timestamp).
        top_k : int
            Number of recommendations per user.

        Returns
        -------
        pd.DataFrame
            Recommendations with columns: user_id, item_id, score, rank.

        Notes
        -----
        If multiple items share the same Semantic ID, recommendation
        deduplication depends on stochastic SID decoding. Set
        ``random_seed`` at model initialization to make outputs
        reproducible.
        """
        self.model.eval()

        df = interactions
        if "timestamp" in df.columns:
            df = df.sort_values(["user_id", "timestamp"])

        user_sequences = df.groupby("user_id")["item_id"].agg(list)
        user_ids = user_sequences.index.tolist()
        sequences = user_sequences.tolist()

        offsets = np.cumsum([TIGERNet.CODEWORD_OFFSET] + self.codebook_sizes)[:-1]

        enc_tokens_list = []
        for seq in sequences:
            seq = seq[-self.max_length :]
            sids: tp.List[tp.Tuple[int]] = self.tokenizer.tokenize(seq)  # type: ignore[assignment]
            tokens: tp.List[int] = []
            for sid in sids:
                tokens.extend(code + offsets[d] for d, code in enumerate(sid))
            enc_tokens_list.append(torch.tensor(tokens, dtype=torch.long))

        rows: tp.List[tp.Tuple] = []
        n_users = len(enc_tokens_list)

        for start in trange(0, n_users, self.eval_batch_size, desc="Generating predictions"):
            end = min(start + self.eval_batch_size, n_users)
            batch_tokens = enc_tokens_list[start:end]
            batch_user_ids = user_ids[start:end]

            batch_max_len = max(t.size(0) for t in batch_tokens)
            enc_input = torch.full(
                (len(batch_tokens), batch_max_len),
                TIGERNet.PAD_TOKEN_ID,
                dtype=torch.long,
                device=self.device,
            )
            for i, t in enumerate(batch_tokens):
                enc_input[i, : t.size(0)] = t.to(self.device)

            enc_padding_mask = enc_input == TIGERNet.PAD_TOKEN_ID

            codes, scores = self.model.generate(enc_input, enc_padding_mask=enc_padding_mask, beam_size=top_k)

            batch_size, num_beams = codes.size(0), codes.size(1)
            codes_cpu = codes.cpu().tolist()
            scores_cpu = scores.cpu()

            all_sids = [tuple(codes_cpu[i][j]) for i in range(batch_size) for j in range(num_beams)]
            all_decoded: tp.List[tp.Optional[int]] = self.tokenizer.decode(  # type: ignore[assignment]
                all_sids,
                rng=self._decode_rng,
            )

            for i in range(batch_size):
                uid = batch_user_ids[i]
                row_decoded = all_decoded[i * num_beams : (i + 1) * num_beams]
                row_scores = scores_cpu[i]
                seen: tp.Set[int] = set()
                rank = 0
                for j, item_id in enumerate(row_decoded):
                    if item_id is not None and item_id not in seen:
                        seen.add(item_id)
                        rank += 1
                        rows.append((uid, item_id, row_scores[j].item(), rank))
                    if rank >= top_k:
                        break

        return pd.DataFrame(rows, columns=["user_id", "item_id", "score", "rank"])

    def evaluate(
        self,
        test_df: pd.DataFrame,
        top_k: tp.Optional[int] = None,
    ) -> tp.Mapping[str, float]:
        """Evaluate the model on a test set.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test interactions (user_id, item_id, optionally timestamp).
            For each user, the last item is treated as the ground-truth target.
        top_k : int, optional
            Number of recommendations to generate. Defaults to ``self.top_k``.

        Returns
        -------
        dict
            Metric name -> value (Hit@k, NDCG@k, MRR@k, etc.).

        Notes
        -----
        If multiple items share the same Semantic ID, decoding during
        evaluation remains stochastic. Set ``random_seed`` at model
        initialization to make metric computation reproducible.
        """
        if top_k is None:
            top_k = self.top_k

        num_items = test_df["item_id"].nunique()

        test_dataset = TIGERDataset(
            interactions=test_df,
            eval_mode=True,
            only_last=True,
            **self._tiger_dataset_kwargs(),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=PaddingCollateFn(padding_value=TIGERNet.PAD_TOKEN_ID),
            pin_memory=True,
        )

        lightning_model = TIGERLightning(
            model=self.model,
            tokenizer=self.tokenizer,
            num_items=num_items,
            beam_size=self.beam_size,
            top_k=top_k,
            decode_rng=self._decode_rng,
        )

        trainer = pl.Trainer()
        results = trainer.test(lightning_model, dataloaders=test_loader)[0]
        self.model = self.model.to(self.device)
        return results

    def save(self, directory: str) -> None:
        """Save the model to a directory.

        Writes ``tokenizer.pt``, ``model.pt``, and ``config.json``.
        """
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save(os.path.join(directory, "tokenizer.pt"))
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))

        config = {
            "hidden_units": self.hidden_units,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "max_length": self.max_length,
            "ff_dim": self.ff_dim,
            "d_kv": self.d_kv,
            "random_seed": self.random_seed,
        }
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str, device: tp.Optional[str] = None) -> "TIGERModel":
        """Load a saved TIGERModel from a directory."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(directory, "config.json"), encoding="utf-8") as f:
            config = json.load(f)

        tokenizer = SIDTokenizer.load(os.path.join(directory, "tokenizer.pt"), device=device)

        tiger_model = cls(tokenizer=tokenizer, device=device, **config)

        state_dict = torch.load(os.path.join(directory, "model.pt"), map_location=device)
        tiger_model.model.load_state_dict(state_dict)

        return tiger_model
