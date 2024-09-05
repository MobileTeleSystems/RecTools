import logging
import typing as tp
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
import typing_extensions as tpe
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset, Interactions
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ErrorBehaviour, InternalRecoTriplet, ModelBase
from rectools.models.rank import Distance, ImplicitRanker
from rectools.types import InternalIdsArray

PADDING_VALUE = "PAD"

logger = logging.getLogger(__name__)  # TODO: remove

# ####  --------------  Net blocks  --------------  #### #


class ItemNetBase(nn.Module):
    """Base class ItemNet. Used only for type hinting."""

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """TODO"""
        raise NotImplementedError()

    @classmethod
    def from_dataset(cls, dataset: Dataset, *args: tp.Any, **kwargs: tp.Any) -> tpe.Self:
        """TODO"""
        raise NotImplementedError()

    def get_all_embeddings(self) -> torch.Tensor:
        """TODO"""
        raise NotImplementedError()


class TransformerLayersBase(nn.Module):
    """Base class for transformer layers. Used only for type hinting."""

    def forward(self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Forward"""
        raise NotImplementedError()


class SessionEncoderDataPreparatorBase:
    """Base class for data preparator. Used only for type hinting."""

    def __init__(
        self,
        session_maxlen: int,
        batch_size: int,
        item_extra_tokens: tp.Sequence[tp.Hashable] = (PADDING_VALUE,),
        shuffle_train: bool = True,
        train_min_user_interactions: int = 2,
    ) -> None:
        """TODO"""
        raise NotImplementedError()

    def get_known_items_sorted_internal_ids(self) -> np.ndarray:
        """TODO"""
        raise NotImplementedError()

    def process_dataset_train(self, dataset: Dataset) -> Dataset:
        """TODO"""
        raise NotImplementedError()

    def get_dataloader_train(self, processed_dataset: Dataset) -> DataLoader:
        """TODO"""
        raise NotImplementedError()

    def get_dataloader_recommend(self, dataset: Dataset) -> DataLoader:
        """TODO"""
        raise NotImplementedError()

    def transform_dataset_u2i(self, dataset: Dataset, users: ExternalIds) -> Dataset:
        """TODO"""
        raise NotImplementedError()

    def transform_dataset_i2i(self, dataset: Dataset) -> Dataset:
        """TODO"""
        raise NotImplementedError()


class SessionEncoderLightningModuleBase(LightningModule):
    """Base class for lightning module. Used only for type hinting."""

    def forward(self, sessions: torch.Tensor) -> torch.Tensor:
        """TODO"""
        raise NotImplementedError()

    def on_train_start(self) -> None:
        """TODO"""
        raise NotImplementedError()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """TODO"""
        raise NotImplementedError()

    def configure_optimizers(self) -> torch.optim.Adam:
        """TODO"""
        raise NotImplementedError()


class IdEmbeddingsItemNet(ItemNetBase):
    """
    Base class for item embeddings. To use more complicated logic then just id embeddings inherit
    from this class and pass your custom ItemNet to your model params
    """

    def __init__(self, n_factors: int, n_items: int, dropout_rate: float):
        super().__init__()

        self.n_items = n_items
        self.item_emb = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=n_factors,
            padding_idx=0,
        )
        self.drop_layer = nn.Dropout(dropout_rate)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_emb(items)
        item_embs = self.drop_layer(item_embs)
        return item_embs

    @property
    def catalogue(self) -> torch.Tensor:
        """TODO"""
        return torch.arange(0, self.n_items, device=self.item_emb.weight.device)

    def get_all_embeddings(self) -> torch.Tensor:
        """TODO"""
        return self.forward(self.catalogue)

    @classmethod
    def from_dataset(cls, dataset: Dataset, n_factors: int, dropout_rate: float) -> tpe.Self:
        """TODO"""
        n_items = dataset.item_id_map.size
        return cls(n_factors, n_items, dropout_rate)


class PointWiseFeedForward(nn.Module):
    """TODO"""

    def __init__(self, n_factors: int, n_factors_ff: int, dropout_rate: float) -> None:
        """TODO"""
        super().__init__()
        self.ff_linear1 = nn.Linear(n_factors, n_factors_ff)
        self.ff_dropout1 = torch.nn.Dropout(dropout_rate)
        self.ff_relu = torch.nn.ReLU()
        self.ff_linear2 = nn.Linear(n_factors_ff, n_factors)
        self.ff_dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """TODO"""
        output = self.ff_relu(self.ff_dropout1(self.ff_linear1(seqs)))
        fin = self.ff_dropout2(self.ff_linear2(output))
        return fin


class SasRecTransformerLayers(TransformerLayersBase):
    """Exactly SASRec authors architecture but with torch MHA realisation"""

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
        )  # TODO: original architecture had another version of MHA
        self.q_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.ff_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.feed_forward = nn.ModuleList(
            [PointWiseFeedForward(n_factors, n_factors, dropout_rate) for _ in range(n_blocks)]
        )
        self.last_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)

    def forward(self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """TODO"""
        for i in range(self.n_blocks):
            q = self.q_layer_norm[i](seqs)
            mha_output, _ = self.multi_head_attn[i](q, seqs, seqs, attn_mask=attn_mask, need_weights=False)
            seqs = q + mha_output
            ff_input = self.ff_layer_norm[i](seqs)
            seqs = self.feed_forward[i](ff_input)
            seqs += ff_input
            seqs *= timeline_mask

        seqs = self.last_layernorm(seqs)

        return seqs


class PreLNTransformerLayers(TransformerLayersBase):
    """
    Based on https://arxiv.org/pdf/2002.04745
    On Kion open dataset didn't change metrics, even got a bit worse
    But let's keep it for now
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
        )
        self.mha_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.mha_dropout = nn.Dropout(dropout_rate)
        self.ff_layer_norm = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.feed_forward = nn.ModuleList(
            [PointWiseFeedForward(n_factors, n_factors, dropout_rate) for _ in range(n_blocks)]
        )

    def forward(self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """TODO"""
        for i in range(self.n_blocks):
            mha_input = self.mha_layer_norm[i](seqs)
            mha_output, _ = self.multi_head_attn[i](
                mha_input, mha_input, mha_input, attn_mask=attn_mask, need_weights=False
            )
            mha_output = self.mha_dropout(mha_output)
            seqs = seqs + mha_output
            ff_input = self.ff_layer_norm[i](seqs)
            ff_output = self.feed_forward[i](ff_input)
            seqs = seqs + ff_output
            seqs *= timeline_mask

        return seqs


class LearnableInversePositionalEncoding(torch.nn.Module):
    """TODO"""

    def __init__(self, use_pos_emb: bool, session_maxlen: int, n_factors: int):
        super().__init__()
        self.pos_emb = torch.nn.Embedding(session_maxlen, n_factors) if use_pos_emb else None

    def forward(self, sessions: torch.Tensor, timeline_mask: torch.Tensor) -> torch.Tensor:
        """TODO"""
        batch_size, session_maxlen, _ = sessions.shape

        if self.pos_emb is not None:
            # Inverse positions are appropriate for variable length sequences across different batches
            # They are equal to absolute positions for fixed sequence length across different batches
            positions = torch.tile(
                torch.arange(session_maxlen - 1, -1, -1), (batch_size, 1)
            )  # [batch_size, session_maxlen]
            sessions += self.pos_emb(positions.to(sessions.device))

        # TODO: do we need to fill padding embeds in sessions to all zeros
        # or should we use the learnt padding embedding? Should we make it an option for user to decide?
        sessions *= timeline_mask  # [batch_size, session_maxlen, n_factors]

        return sessions


# ####  --------------  Session Encoder  --------------  #### #


class TransformerBasedSessionEncoder(torch.nn.Module):
    """TODO"""

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        session_maxlen: int,
        dropout_rate: float,
        use_pos_emb: bool = True,  # TODO: add pos_encoding_type option for user to pass
        use_causal_attn: bool = True,
        transformer_layers_type: tp.Type[TransformerLayersBase] = SasRecTransformerLayers,
        item_net_type: tp.Type[ItemNetBase] = IdEmbeddingsItemNet,
    ) -> None:
        super().__init__()

        self.item_model: ItemNetBase
        self.pos_encoding = LearnableInversePositionalEncoding(use_pos_emb, session_maxlen, n_factors)
        self.emb_dropout = torch.nn.Dropout(dropout_rate)
        self.transformer_layers = transformer_layers_type(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
        )
        self.use_causal_attn = use_causal_attn
        self.item_net_type = item_net_type
        self.n_factors = n_factors
        self.dropout_rate = dropout_rate

    def construct_item_net(self, dataset: Dataset) -> None:
        """TODO"""
        self.item_model = self.item_net_type.from_dataset(dataset, self.n_factors, self.dropout_rate)

    def encode_sessions(self, sessions: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings and transformer blocks.

        Returns
        -------
            torch.Tensor. [batch_size, session_maxlen, n_factors]

        """
        session_maxlen = sessions.shape[1]
        attn_mask = None
        if self.use_causal_attn:
            attn_mask = ~torch.tril(
                torch.ones((session_maxlen, session_maxlen), dtype=torch.bool, device=sessions.device)
            )
        timeline_mask = (sessions != 0).unsqueeze(-1)  # [batch_size, session_maxlen, 1]
        seqs = item_embs[sessions]  # [batch_size, session_maxlen, n_factors]
        seqs = self.pos_encoding(seqs, timeline_mask)
        seqs = self.emb_dropout(seqs)
        seqs = self.transformer_layers(seqs, timeline_mask, attn_mask)
        return seqs

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_maxlen]
    ) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_model.get_all_embeddings()  # [n_items + 1, n_factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_maxlen, n_factors]
        logits = session_embs @ item_embs.T  # [batch_size, session_maxlen, n_items + 1]
        return logits


# ####  --------------  Data Processor  --------------  #### #


class SequenceDataset(TorchDataset):
    """TODO"""

    def __init__(self, sessions: List[List[int]], weights: List[List[float]]):
        self.sessions = sessions
        self.weights = weights

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int) -> Tuple[List[int], List[float]]:
        session = self.sessions[index]  # [session_len]
        weights = self.weights[index]  # [session_len]
        return session, weights

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
    ) -> "SequenceDataset":
        """TODO"""
        sessions = (
            interactions.sort_values(Columns.Datetime)
            .groupby(Columns.User, sort=True)[[Columns.Item, Columns.Weight]]
            .agg(list)
        )
        sessions, weights = (
            sessions[Columns.Item].to_list(),
            sessions[Columns.Weight].to_list(),
        )

        return cls(sessions=sessions, weights=weights)


class SasRecDataPreparator(SessionEncoderDataPreparatorBase):
    """TODO"""

    def __init__(
        self,
        session_maxlen: int,
        batch_size: int,
        item_extra_tokens: tp.Sequence[tp.Hashable] = (PADDING_VALUE,),
        shuffle_train: bool = True,  # not shuffling train dataloader hurts performance
        train_min_user_interactions: int = 2,
    ) -> None:
        self.session_maxlen = session_maxlen
        self.batch_size = batch_size
        self.item_extra_tokens = item_extra_tokens
        self.shuffle_train = shuffle_train
        self.train_min_user_interactions = train_min_user_interactions
        self.item_id_map: IdMap
        # TODO: add SequenceDatasetType for fit and recommend

    @property
    def n_item_extra_tokens(self) -> int:
        """TODO"""
        return len(self.item_extra_tokens)

    def get_known_item_ids(self) -> np.ndarray:
        """TODO"""
        return self.item_id_map.get_external_sorted_by_internal()[self.n_item_extra_tokens :]

    def get_known_items_sorted_internal_ids(self) -> np.ndarray:
        """TODO"""
        return self.item_id_map.get_sorted_internal()[self.n_item_extra_tokens :]

    def process_dataset_train(self, dataset: Dataset) -> Dataset:
        """TODO"""
        interactions = dataset.get_raw_interactions()

        # Filter interactions
        user_stats = interactions[Columns.User].value_counts()
        users = user_stats[user_stats >= self.train_min_user_interactions].index
        interactions = interactions[(interactions[Columns.User].isin(users))]
        interactions = interactions.sort_values(Columns.Datetime).groupby(Columns.User).tail(self.session_maxlen + 1)

        # Construct dataset
        # TODO: user features and item features are dropped for now
        user_id_map = IdMap.from_values(interactions[Columns.User].values)
        item_id_map = IdMap.from_values(self.item_extra_tokens)
        item_id_map = item_id_map.add_ids(interactions[Columns.Item])
        interactions = Interactions.from_raw(interactions, user_id_map, item_id_map)
        dataset = Dataset(user_id_map, item_id_map, interactions)

        self.item_id_map = dataset.item_id_map
        return dataset

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        """
        Truncate each session from right to keep (session_maxlen+1) last items.
        Do left padding until  (session_maxlen+1) is reached.
        Split to `x`, `y`, and `yw`.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_maxlen))
        y = np.zeros((batch_size, self.session_maxlen))
        yw = np.zeros((batch_size, self.session_maxlen))
        for i, (ses, ses_weights) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_maxlen]
            y[i, -len(ses) + 1 :] = ses[1:]  # ses: [session_len] -> y[i]: [session_maxlen]
            yw[i, -len(ses) + 1 :] = ses_weights[1:]  # ses_weights: [session_len] -> yw[i]: [session_maxlen]
        return torch.LongTensor(x), torch.LongTensor(y), torch.FloatTensor(yw)

    def get_dataloader_train(self, processed_dataset: Dataset) -> DataLoader:
        """TODO"""
        sequence_dataset = SequenceDataset.from_interactions(processed_dataset.interactions.df)
        train_dataloader = DataLoader(
            sequence_dataset, collate_fn=self._collate_fn_train, batch_size=self.batch_size, shuffle=self.shuffle_train
        )
        return train_dataloader

    def transform_dataset_u2i(self, dataset: Dataset, users: ExternalIds) -> Dataset:
        """
        Filter out interactions and adapt id maps.
        Final dataset will consist only of model known items during fit and only of required
        (and supported) target users for recommendations.
        All users beyond target users for recommendations are dropped.
        All target users that do not have at least one known item in interactions are dropped.
        Final user_id_map is an enumerated list of supported (filtered) target users
        Final item_id_map is model item_id_map constructed during training
        """
        # Filter interactions in dataset internal ids
        interactions = dataset.interactions.df
        users_internal = dataset.user_id_map.convert_to_internal(users, strict=False)
        items_internal = dataset.item_id_map.convert_to_internal(self.get_known_item_ids(), strict=False)
        interactions = interactions[interactions[Columns.User].isin(users_internal)]  # todo: fast_isin
        interactions = interactions[interactions[Columns.Item].isin(items_internal)]

        # Convert to external ids
        interactions[Columns.Item] = dataset.item_id_map.convert_to_external(interactions[Columns.Item])
        interactions[Columns.User] = dataset.user_id_map.convert_to_external(interactions[Columns.User])

        # Prepare new user id mapping
        rec_user_id_map = IdMap.from_values(interactions[Columns.User])

        # Construct dataset
        # TODO: For now features are dropped because model doesn't support them
        n_filtered = len(users) - rec_user_id_map.size
        if n_filtered > 0:
            explanation = f"""{n_filtered} target users were considered cold
            because of missing known items"""
            warnings.warn(explanation)
        filtered_interactions = Interactions.from_raw(interactions, rec_user_id_map, self.item_id_map)
        filtered_dataset = Dataset(rec_user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def transform_dataset_i2i(self, dataset: Dataset) -> Dataset:
        """
        Filter out interactions and adapt id maps.
        Final dataset will consist only of model known items during fit.
        Final user_id_map is the same as dataset original
        Final item_id_map is model item_id_map constructed during training
        """
        # TODO: optimize by filtering in internal ids
        # TODO: For now features are dropped because model doesn't support them
        interactions = dataset.get_raw_interactions()
        interactions = interactions[interactions[Columns.Item].isin(self.get_known_item_ids())]
        filtered_interactions = Interactions.from_raw(interactions, dataset.user_id_map, self.item_id_map)
        filtered_dataset = Dataset(dataset.user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> torch.LongTensor:
        """Right truncation, left padding to session_maxlen"""
        x = np.zeros((len(batch), self.session_maxlen))
        for i, (ses, _) in enumerate(batch):
            x[i, -len(ses) :] = ses[-self.session_maxlen :]
        return torch.LongTensor(x)

    def get_dataloader_recommend(self, dataset: Dataset) -> DataLoader:
        """TODO"""
        sequence_dataset = SequenceDataset.from_interactions(dataset.interactions.df)
        recommend_dataloader = DataLoader(
            sequence_dataset, batch_size=self.batch_size, collate_fn=self._collate_fn_recommend, shuffle=False
        )
        return recommend_dataloader


# ####  --------------  Lightning Model  --------------  #### #


class SessionEncoderLightningModule(SessionEncoderLightningModuleBase):
    """TODO"""

    def __init__(
        self,
        torch_model: TransformerBasedSessionEncoder,
        lr: float,
        loss: str = "softmax",  # TODO: move loss func to `SasRec` class
        adam_betas: Tuple[float, float] = (0.9, 0.98),
    ):
        super().__init__()
        self.lr = lr
        self.loss_func = self._init_loss_func(loss)
        self.torch_model = torch_model
        self.adam_betas = adam_betas

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_maxlen]
    ) -> torch.Tensor:
        """TODO"""
        return self.torch_model(sessions)

    def on_train_start(self) -> None:
        """TODO"""
        self._xavier_normal_init()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """TODO"""
        x, y, w = batch
        logits = self.forward(x)  # [batch_size, session_maxlen, n_items + 1]
        # We are using CrossEntropyLoss with a multi-dimensional case

        # Logits must be passed in form of [batch_size, n_items + 1, session_maxlen],
        #  where n_items + 1 is number of classes

        # Target label indexes must be passed in a form of [batch_size, session_maxlen]
        # (`0` index for "PAD" ix excluded from loss)

        # Loss output will have a shape of [batch_size, session_maxlen]
        # and will have zeros for every `0` target label
        loss = self.loss_func(logits.transpose(1, 2), y)  # [batch_size, session_maxlen]
        loss = loss * w
        n = (loss > 0).to(loss.dtype)
        loss = torch.sum(loss) / torch.sum(n)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """TODO"""
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...

    def _init_loss_func(self, loss: str) -> nn.CrossEntropyLoss:
        """TODO"""
        if loss == "softmax":
            return nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        raise ValueError(f"loss {loss} is not supported")

    def _xavier_normal_init(self) -> None:
        """TODO"""
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError:
                pass


# ####  --------------  SASRec Model  --------------  #### #


class SasRecModel(ModelBase):  # pylint: disable=too-many-instance-attributes
    """TODO"""

    def __init__(
        self,
        session_maxlen: int,
        lr: float,
        batch_size: int,
        epochs: int,
        device: str,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        use_pos_emb: bool = True,
        loss: str = "softmax",
        verbose: int = 0,
        cpu_n_threads: int = 0,
        deterministic: bool = False,
        trainer: tp.Optional[Trainer] = None,
        transformer_layers_type: tp.Type[TransformerLayersBase] = SasRecTransformerLayers,  # SASRec authors net
        item_net_type: tp.Type[ItemNetBase] = IdEmbeddingsItemNet,  # item embeddings on ids
        data_preparator_type: tp.Type[SessionEncoderDataPreparatorBase] = SasRecDataPreparator,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
    ):
        super().__init__(verbose=verbose)
        self.device = torch.device(device)
        self.n_threads = cpu_n_threads
        self.model: TransformerBasedSessionEncoder
        self._model = TransformerBasedSessionEncoder(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            session_maxlen=session_maxlen,
            dropout_rate=dropout_rate,
            use_pos_emb=use_pos_emb,
            use_causal_attn=True,
            transformer_layers_type=transformer_layers_type,
            item_net_type=item_net_type,
        )
        self.lightning_module_type = lightning_module_type
        self.trainer: Trainer
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
        self.data_preparator = data_preparator_type(session_maxlen, batch_size)
        self.u2i_dist = Distance.DOT
        self.i2i_dist = Distance.COSINE
        self.lr = lr
        self.loss = loss

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        processed_dataset = self.data_preparator.process_dataset_train(dataset)
        train_dataloader = self.data_preparator.get_dataloader_train(processed_dataset)

        self.model = deepcopy(self._model)  # TODO: check that it works
        self.model.construct_item_net(processed_dataset)

        lightning_model = self.lightning_module_type(self.model, self.lr, self.loss)
        self.trainer = deepcopy(self._trainer)
        self.trainer.fit(lightning_model, train_dataloader)

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
        dataset: Dataset,  # [n_rec_users x n_items + 1]
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],  # model_internal
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:  # TODO: move to _get_sorted_item_ids_to_recommend
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()  # model internal

        self.model = self.model.eval()
        self.model.to(self.device)

        # Dataset has already been filtered and adapted to known item_id_map
        recommend_dataloader = self.data_preparator.get_dataloader_recommend(dataset)

        session_embs = []
        item_embs = self.model.item_model.get_all_embeddings()  # [n_items + 1, n_factors]
        with torch.no_grad():
            for x_batch in tqdm.tqdm(recommend_dataloader):  # TODO: from tqdm.auto import tqdm. Also check `verbose``
                x_batch = x_batch.to(self.device)  # [batch_size, session_maxlen]
                encoded = self.model.encode_sessions(x_batch, item_embs)[:, -1, :]  # [batch_size, n_factors]
                encoded = encoded.detach().cpu().numpy()
                session_embs.append(encoded)

        user_embs = np.concatenate(session_embs, axis=0)
        user_embs = user_embs[user_ids]
        item_embs_np = item_embs.detach().cpu().numpy()

        ranker = ImplicitRanker(
            self.u2i_dist,
            user_embs,  # [n_rec_users, n_factors]
            item_embs_np,  # [n_items + 1, n_factors]
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
            filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + 1]
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model_internal
            num_threads=self.n_threads,
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
        item_embs = self.model.item_model.get_all_embeddings().detach().cpu().numpy()  # [n_items + 1, n_factors]

        # TODO: i2i reco do not need filtering viewed. And user most of the times has GPU
        # Should we use torch dot and topk? Should be faster

        ranker = ImplicitRanker(
            self.i2i_dist,
            item_embs,  # [n_items + 1, n_factors]
            item_embs,  # [n_items + 1, n_factors]
        )
        return ranker.rank(
            subject_ids=target_ids,  # model internal
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model internal
            num_threads=0,
        )

    @property
    def torch_model(self) -> TransformerBasedSessionEncoder:
        """TODO"""
        return self.model
