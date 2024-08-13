import logging
import typing as tp
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset, Interactions
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ErrorBehaviour, InternalRecoTriplet, ModelBase
from rectools.models.rank import Distance, ImplicitRanker
from rectools.types import ExternalIdsArray, InternalIdsArray

PADDING_VALUE = "PAD"

logger = logging.getLogger(__name__)


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


class SasRecDataPreparator:
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
        item_id_map = item_id_map.add_ids(np.unique(interactions[Columns.Item]))  # TODO: remove unique
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

    def process_dataset_recommend(self, dataset: Dataset, users: ExternalIds) -> Dataset:
        """
        Filter out interactions and adapt id maps.
        Final dataset will consist only of model known items during fit and only of required
        (and supported) target users for recommendations.
        All users beyond target users for recommendations are dropped.
        All target users that do not have at least one known item in interactions are dropped.
        Final user_id_map is an enumerated list of supported (filtered) target users
        Final item_id_map is model item_id_map constructed during training
        """
        # Filter interactions
        interactions = dataset.get_raw_interactions()
        interactions = interactions[interactions[Columns.User].isin(users)]
        interactions = interactions[interactions[Columns.Item].isin(self.get_known_item_ids())]

        # Construct dataset
        # TODO: For now features are dropped because model doesn't support them
        rec_user_id_map = IdMap.from_values(interactions[Columns.User])
        n_filtered = len(users) - rec_user_id_map.size
        if n_filtered > 0:
            explanation = f"""{n_filtered} target users were considered cold
            because of missing known items"""
            warnings.warn(explanation)
        filtered_interactions = Interactions.from_raw(interactions, rec_user_id_map, self.item_id_map)
        filtered_dataset = Dataset(rec_user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def process_dataset_recommend_to_items(self, dataset: Dataset) -> Dataset:
        """
        Filter out interactions and adapt id maps.
        Final dataset will consist only of model known items during fit.
        Final user_id_map is the same as dataset original
        Final item_id_map is model item_id_map constructed during training
        """
        # TODO: For now features are dropped because model doesn't support them
        interactions = dataset.get_raw_interactions()
        interactions = interactions[interactions[Columns.Item].isin(self.get_known_item_ids())]
        filtered_interactions = Interactions.from_raw(interactions, dataset.user_id_map, self.item_id_map)
        filtered_dataset = Dataset(dataset.user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> torch.LongTensor:
        """TODO"""
        x = np.zeros((len(batch), self.session_maxlen))
        # left padding, left truncation
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


class SasRecRecommenderModel(ModelBase):
    """TODO"""

    def __init__(
        self,
        session_maxlen: int,
        lr: float,
        batch_size: int,
        epochs: int,
        device: str,
        n_blocks: int,
        factors: int,
        n_heads: int,
        dropout_rate: float,
        item_net_dropout_rate: float,
        use_pos_emb: bool = True,
        loss: str = "softmax",
        verbose: int = 0,
        random_state: int = False,
    ):  # pylint: disable=too-many-instance-attributes
        super().__init__(verbose=verbose)
        self.session_maxlen = session_maxlen
        self.n_blocks = n_blocks
        self.factors = factors
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.item_net_dropout_rate = item_net_dropout_rate
        self.use_pos_emb = use_pos_emb
        self.device = torch.device(device)
        self.model: SASRec
        self.trainer = Trainer(
            lr=lr,
            epochs=epochs,
            device=self.device,
            loss=loss,
        )
        self.data_preparator = SasRecDataPreparator(session_maxlen, batch_size)
        if random_state is not None:
            torch.use_deterministic_algorithms(True)
            seed_everything(random_state, workers=True)

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        processed_dataset = self.data_preparator.process_dataset_train(dataset)
        train_dataloader = self.data_preparator.get_dataloader_train(processed_dataset)

        n_items = self.data_preparator.item_id_map.size
        self.model = SASRec(
            n_blocks=self.n_blocks,
            factors=self.factors,
            n_heads=self.n_heads,
            session_maxlen=self.session_maxlen,
            dropout_rate=self.dropout_rate,
            use_pos_emb=self.use_pos_emb,
            n_items=n_items,  # TODO: can we init a SASRec net without knowing this?
            item_net_dropout_rate=self.item_net_dropout_rate,
        )

        self.trainer.fit(self.model, train_dataloader)
        self.model = self.trainer.model

    def _process_dataset_u2i(
        self, dataset: Dataset, users: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        filtered_dataset = self.data_preparator.process_dataset_recommend(dataset, users)
        return filtered_dataset

    @classmethod
    def _split_targets_by_hot_warm_cold(  # TODO: remove this
        cls,
        targets: ExternalIds,  # users for U2I or target items for I2I
        dataset: Dataset,
        entity: tp.Literal["user", "item"],
    ) -> tp.Tuple[InternalIdsArray, InternalIdsArray, ExternalIdsArray]:

        if entity == "user":
            # We already filtered out warm and cold user ids
            _, new_ids = dataset.user_id_map.convert_to_internal(targets, strict=False, return_missing=True)
            return dataset.user_id_map.get_sorted_internal(), np.asarray([]), np.asarray([])  # new_ids

        # Warm items were already filtered out from dataset
        known_ids, new_ids = dataset.item_id_map.convert_to_internal(targets, strict=False, return_missing=True)
        return known_ids, np.asarray([]), new_ids

    def _process_dataset_i2i(
        self, dataset: Dataset, target_items: ExternalIds, on_unsupported_targets: ErrorBehaviour
    ) -> Dataset:
        filtered_dataset = self.data_preparator.process_dataset_recommend_to_items(dataset)
        return filtered_dataset

    def _recommend_u2i(
        self,
        user_ids: InternalIdsArray,  # n_rec_users
        dataset: Dataset,  # [n_rec_users x n_items + 1]
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],  # model_internal
    ) -> InternalRecoTriplet:

        if sorted_item_ids_to_recommend is None:
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()  # model internal

        self.model = self.model.eval()
        self.model.to(self.device)

        # Dataset has already been filtered and adapted to known item_id_map
        recommend_dataloader = self.data_preparator.get_dataloader_recommend(dataset)

        session_embs = []
        device = self.model.get_model_device()
        self.model.item_model.to_device(device)
        item_embs = self.model.item_model.get_all_embeddings()  # [n_items + 1, factors]
        with torch.no_grad():
            for x_batch in tqdm.tqdm(recommend_dataloader):
                x_batch = x_batch.to(device)  # [batch_size, session_maxlen]
                encoded = self.model.encode_sessions(x_batch, item_embs)[:, -1, :]  # [batch_size, factors]
                encoded = encoded.detach().cpu().numpy()
                session_embs.append(encoded)

        user_embs = np.concatenate(session_embs, axis=0)
        user_embs = user_embs[user_ids]  # in case user_ids are not sorted properly
        item_embs_np = item_embs.detach().cpu().numpy()

        ranker = ImplicitRanker(
            Distance.DOT,
            user_embs,  # [n_rec_users, factors]
            item_embs_np,  # [n_items + 1, factors]
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
            num_threads=0,  # TODO: think about receiving CPU num_threads from user
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
        item_embs = self.model.item_model.get_all_embeddings().detach().cpu().numpy()  # [n_items + 1, factors]

        # TODO: i2i reco do not need filtering viewed. And user most of the times has GPU
        # Should we use torch dot and topk? Should be faster

        ranker = ImplicitRanker(
            Distance.COSINE,
            item_embs,  # [n_items + 1, factors]
            item_embs,  # [n_items + 1, factors]
        )
        return ranker.rank(
            subject_ids=target_ids,  # model internal
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model internal
            num_threads=0,
        )


class PointWiseFeedForward(torch.nn.Module):
    """TODO"""

    def __init__(self, factors: int, dropout_rate: float):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(factors, factors, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(factors, factors, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """TODO"""
        # [batch_size, session_maxlen, factors] -> [batch_size, factors, session_maxlen]
        inputs = inputs.transpose(-1, -2)
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs)))))
        # [batch_size, factors, session_maxlen] -> [batch_size, session_maxlen, factors]
        outputs = outputs.transpose(-1, -2)
        # [batch_size, factors, session_maxlen] -> [batch_size, session_maxlen, factors]
        inputs = inputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class ItemNet(nn.Module):
    """TODO"""

    def __init__(self, factors: int, n_items: int, item_net_dropout_rate: float):
        super().__init__()

        self.catalogue: torch.Tensor
        self.n_items = n_items
        self.item_emb = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=factors,
            padding_idx=0,
        )
        self.drop_layer = nn.Dropout(item_net_dropout_rate)

    def forward(self) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_emb(self.catalogue)
        item_embs = self.drop_layer(item_embs)
        return item_embs

    def get_all_embeddings(self) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_emb(self.catalogue)  # [n_items + 1, factors]
        item_embs = self.drop_layer(item_embs)
        return item_embs

    def to_device(self, device: torch.device) -> None:
        """TODO"""
        catalogue = torch.LongTensor(list(range(self.n_items)))
        self.catalogue = catalogue.to(device)

    def get_device(self) -> torch.device:
        """TODO"""
        return self.item_emb.weight.device


class TransformerDecoder(nn.Module):
    """TODO"""

    def __init__(
        self,
        n_blocks: int,
        factors: int,
        n_heads: int,
        dropout_rate: float,
    ):
        super().__init__()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(n_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(factors, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(factors, n_heads, dropout_rate, batch_first=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(factors, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(factors, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(
        self,
        seqs: torch.Tensor,
        attention_mask: torch.Tensor,
        timeline_mask: torch.Tensor,
    ) -> torch.Tensor:
        """TODO"""
        for i, _ in enumerate(self.attention_layers):
            q = self.attention_layernorms[i](seqs)  # [batch_size, session_maxlen, factors]
            mha_outputs, _ = self.attention_layers[i](q, seqs, seqs, attn_mask=attention_mask)
            seqs = q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= timeline_mask

        return seqs


class SASRec(torch.nn.Module):
    """TODO"""

    def __init__(
        self,
        n_blocks: int,
        factors: int,
        n_heads: int,
        session_maxlen: int,
        dropout_rate: float,
        n_items: int,
        item_net_dropout_rate: float,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.n_items = n_items
        self.session_maxlen = session_maxlen

        self.item_model = ItemNet(factors, n_items, item_net_dropout_rate)

        if self.use_pos_emb:
            self.pos_emb = torch.nn.Embedding(session_maxlen, factors)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.encoder = TransformerDecoder(
            n_blocks=n_blocks,
            factors=factors,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
        )
        self.last_layernorm = torch.nn.LayerNorm(factors, eps=1e-8)

    def get_model_device(self) -> torch.device:
        """TODO"""
        return self.item_model.get_device()

    def encode_sessions(self, sessions: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings and transformer blocks.

        Returns
        -------
            torch.Tensor. [batch_size, history_len, emdedding_dim]

        """
        session_maxlen = sessions.shape[1]
        seqs = item_embs[sessions]  # [batch_size, session_maxlen, factors]
        timeline_mask = (sessions != 0).unsqueeze(-1)  # [batch_size, session_maxlen, 1]

        # TODO: add inverse positional embedding (much talked about with good feedback)
        if self.use_pos_emb:
            positions = np.tile(np.array(range(session_maxlen)), [sessions.shape[0], 1])  # [batch_size, session_maxlen]
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.get_model_device()))

        # TODO: do we need to fill padding embeds in sessions to all zeros
        # or should we use the learnt padding embedding? Should we make it an option for user to decide?
        seqs *= timeline_mask  # [batch_size, session_maxlen, factors]
        seqs = self.emb_dropout(seqs)

        # TODO do we need to mask padding embs attention?
        # [session_maxlen, session_maxlen]
        attention_mask = ~torch.tril(
            torch.ones((session_maxlen, session_maxlen), dtype=torch.bool, device=self.get_model_device())
        )

        seqs = self.encoder(
            seqs=seqs,
            attention_mask=attention_mask,
            timeline_mask=timeline_mask,
        )  # [batch_size, session_maxlen, factors]

        seqs = self.last_layernorm(seqs)  # [batch_size, session_maxlen, factors]

        return seqs

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_maxlen]
    ) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_model.get_all_embeddings()  # [n_items + 1, factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_maxlen, factors]
        logits = session_embs @ item_embs.T  # [batch_size, session_maxlen, n_items + 1]
        return logits


class Trainer:
    """TODO"""

    def __init__(
        self,
        lr: float,
        epochs: int,
        device: torch.device,
        loss: str = "softmax",
    ):
        self.model: SASRec
        self.optimizer: torch.optim.Adam
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.loss_func = self._init_loss_func(loss)  # TODO: move loss func to `SasRec` class

    def fit(
        self,
        model: SASRec,
        fit_dataloader: DataLoader,
    ) -> None:
        """TODO"""
        self.model = model
        self.optimizer = self._init_optimizers()
        self.model.to(self.device)

        self.xavier_normal_init(self.model)
        self.model.train()  # enable model training
        self.model.item_model.to_device(self.device)

        epoch_start_idx = 1

        # ce_criterion = torch.nn.CrossEntropyLoss()
        # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...

        try:
            for epoch in range(epoch_start_idx, self.epochs + 1):
                logger.info("training epoch %s", epoch)
                for x, y, w in fit_dataloader:
                    x = x.to(self.device)  # [batch_size, session_maxlen]
                    y = y.to(self.device)  # [batch_size, session_maxlen]
                    w = w.to(self.device)  # [batch_size, session_maxlen]

                    self.train_step(x, y, w)

        except KeyboardInterrupt:
            logger.info("training interritem_model_inputupted")

    def train_step(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> None:
        """TODO"""
        self.optimizer.zero_grad()
        logits = self.model(x)  # [batch_size, session_maxlen, n_items + 1]
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
        joint_loss = loss  # + reg_loss
        joint_loss.backward()
        self.optimizer.step()

    def _init_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98))
        return optimizer

    def _init_loss_func(self, loss: str) -> nn.CrossEntropyLoss:

        if loss == "softmax":
            return nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        raise ValueError(f"loss {loss} is not supported")

    def xavier_normal_init(self, model: nn.Module) -> None:
        """TODO"""
        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError as err:
                logger.info("unable to init param %s with xavier: %s", name, err)
