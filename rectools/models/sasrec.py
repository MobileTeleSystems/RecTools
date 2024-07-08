import logging
import math
import typing as tp
from dataclasses import dataclass
from itertools import compress
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rectools import AnyIds, Columns
from rectools.dataset import Dataset as RecDataset
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ModelBase

logger = logging.getLogger(__name__)


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
        use_pos_emb: bool = True,
        loss: str = "bce",
        verbose: int = 0,
        random_state: int = None,
    ):
        super().__init__(verbose=verbose)
        self.session_maxlen = session_maxlen
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.loss = loss
        self.n_blocks = n_blocks
        self.factors = factors
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.use_pos_emb = use_pos_emb
        self.random_state = random_state

        if self.random_state is not None:
            torch.use_deterministic_algorithms(True)
            seed_everything(self.random_state, workers=True)

    def fit(
        self,
        dataset: RecDataset,
    ):
        """TODO"""
        user_item_interactions = dataset.get_raw_interactions()

        user_item_interactions[Columns.User] = user_item_interactions[Columns.User].astype(str)
        user_item_interactions[Columns.Item] = user_item_interactions[Columns.Item].astype(str)

        users = user_item_interactions[Columns.User].value_counts()
        users = users[users >= 2]
        users = users.index.to_list()
        user_item_interactions = user_item_interactions[(user_item_interactions[Columns.User].isin(users))]

        self.task_converter = SequenceTaskConverter()
        self.task_converter.build(self.session_maxlen, user_item_interactions)
        user_item_interactions[Columns.Item] = self.task_converter.item_id_encoder.convert_to_internal(
            user_item_interactions[Columns.Item]
        )
        logger.info("converting datasets to task format")
        train_x, train_y = self.task_converter.train_transform(
            user_item_interactions=user_item_interactions,
        )

        logger.info("building train dataset")
        train_dataset = SequenceDataset(x=train_x, y=train_y, batch_size=self.batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size=None)  # batching in dataset

        # init model
        logger.info("building model")
        self.model = SASRec(
            n_blocks=self.n_blocks,
            factors=self.factors,
            n_heads=self.n_heads,
            session_maxlen=self.session_maxlen,
            dropout_rate=self.dropout_rate,
            use_pos_emb=self.use_pos_emb,
            n_items=self.task_converter.item_id_encoder.size if self.task_converter.item_id_encoder is not None else -1,
        )

        logger.info("building trainer")
        trainer = Trainer(
            model=self.model,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            device=self.device,
            loss=self.loss,
        )
        trainer.fit(train_dataloader)
        self.model = trainer.model

    def recommend(
        self,
        users: List[str],
        dataset: RecDataset,
        k: int,
        items_to_recommend: tp.Optional[AnyIds] = None,
    ):
        """TODO"""
        train = dataset.get_raw_interactions()

        train[Columns.User] = train[Columns.User].astype(str)
        train[Columns.Item] = train[Columns.Item].astype(str)
        item_features = train[Columns.Item].drop_duplicates()

        rec_df = train[train[Columns.User].isin(users)]
        recommender = SASRecRecommender(self.model, self.task_converter)

        if items_to_recommend is None:
            items_to_recommend = item_features

        recs = recommender.recommend(
            user_item_interactions=rec_df,
            item_features=item_features,
            top_k=k,
            candidate_items=items_to_recommend,
        )
        return recs


@dataclass
class SASRecModelInput:
    """
    sessions (torch.LongTensor): [batch_size, maxlen] user history
    item_model_input (Tuple[torch.LongTensor, torch.LongTensor]): item features input to construct
    """

    sessions: torch.LongTensor
    item_model_input: torch.LongTensor
    users: List[str]


@dataclass
class SASRecTargetInput:
    """TODO"""

    next_watch: torch.LongTensor
    weights: Optional[torch.LongTensor] = None


class SequenceTaskConverter:
    """Convert database data to use in particular model"""

    def train_transform(
        self,
        user_item_interactions: pd.DataFrame,
    ) -> Tuple[SASRecModelInput, SASRecTargetInput]:
        """Convert user-item interaction to sessions and extract target.
        Prepare other features
        """
        users, sessions, weights = self._interactions2sessions(user_item_interactions)
        sessions_x, sessions_y, x_weights, y_weights = self._sessions2xy(sessions, weights)

        items = user_item_interactions[Columns.Item].drop_duplicates().sort_values().to_list()

        sessions, _ = self._sessions_transform(sessions_x, x_weights)
        sessions = torch.LongTensor(sessions)

        item_model_input = np.concatenate([np.zeros_like(items[:1]), items], axis=0)
        item_model_input = torch.LongTensor(item_model_input)

        next_watch, weights = self._sessions_transform(sessions_y, y_weights)
        next_watch = torch.LongTensor(next_watch)
        weights = torch.FloatTensor(weights)

        return (
            SASRecModelInput(
                sessions=sessions,
                item_model_input=item_model_input,
                users=users,
            ),
            SASRecTargetInput(
                next_watch=next_watch,
                weights=weights,
            ),
        )

    # TODO What if sessions contains items not present in item features?
    # TODO What if user's consist only of items not known by the model?
    def inference_transform(
        self,
        user_item_interactions: pd.DataFrame,
        item_features: pd.Series,
    ) -> SASRecModelInput:
        """Convert user-item interaction to sessions and prepare other features"""
        users, sessions, weights = self._interactions2sessions(user_item_interactions)

        items = item_features.sort_values().to_list()

        sessions, _ = self._sessions_transform(sessions, weights)
        sessions = torch.LongTensor(sessions)

        item_model_input = np.concatenate([np.zeros_like(items[:1]), items], axis=0)
        item_model_input = torch.LongTensor(item_model_input)

        return SASRecModelInput(
            sessions=sessions,
            item_model_input=item_model_input,
            users=users,
        )

    def _interactions2sessions(
        self, user_item_interactions: pd.DataFrame
    ) -> Tuple[List[str], List[List[str]], List[List[float]]]:
        sessions = (
            user_item_interactions.sort_values(Columns.Datetime)
            .groupby(Columns.User)[[Columns.Item, Columns.Weight]]
            .agg(list)
        )
        users, sessions, weights = (
            sessions.index.to_list(),
            sessions[Columns.Item].to_list(),
            sessions[Columns.Weight].to_list(),
        )
        return users, sessions, weights

    def _sessions2xy(
        self,
        sessions: List[List[str]],
        weights: List[List[float]],
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]], List[List[float]]]:
        x = []
        y = []
        xw = []
        yw = []

        for ses, ses_weights in zip(sessions, weights):
            cx = ses[:-1]
            cy = ses[1:]
            cxw = ses_weights[:-1]
            cyw = ses_weights[1:]

            assert len(cx) > 0, "too short session to generate target found"

            x.append(cx)
            y.append(cy)
            xw.append(cxw)
            yw.append(cyw)

        return x, y, xw, yw

    def _sessions_transform(
        self,
        sessions: List[List[str]],
        weights: List[List[float]],
    ) -> np.ndarray:
        x = np.zeros((len(sessions), self.session_maxlen))
        w = np.zeros((len(sessions), self.session_maxlen))
        # left padding, left truncation
        for i, (ses, ses_w) in enumerate(zip(sessions, weights)):
            cur_ses = ses[-self.session_maxlen :]
            cur_w = ses_w[-self.session_maxlen :]

            x[i, -len(cur_ses) :] = cur_ses
            w[i, -len(cur_ses) :] = cur_w

        return x, w

    def build(
        self,
        session_maxlen: int,
        user_item_interactions: pd.DataFrame,
    ):
        """TODO"""
        self.session_maxlen = session_maxlen

        items = user_item_interactions[Columns.Item].drop_duplicates().sort_values()
        self.item_id_encoder = IdMap.from_values("PAD")
        self.item_id_encoder = self.item_id_encoder.add_ids(list(items))


class SequenceDataset(Dataset):
    """TODO"""

    def __init__(
        self,
        x: SASRecModelInput,
        batch_size: int,
        y: Optional[SASRecTargetInput] = None,
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size

        self.samples_cnt = self.x.sessions.shape[0]
        self.batches_cnt = math.ceil(self.samples_cnt / self.batch_size)

    def __len__(self):
        return self.batches_cnt

    def __getitem__(self, index):
        batch_idx = index
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.samples_cnt)

        # inference case
        if self.y is None:
            return (
                self.x.sessions[start:end],
                self.x.item_model_input,
            )

        # train case
        return (
            (
                self.x.sessions[start:end],
                self.x.item_model_input,
            ),
            self.y.next_watch[start:end],
            self.y.weights[start:end],
        )


class PointWiseFeedForward(torch.nn.Module):
    """TODO"""

    def __init__(self, factors, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(factors, factors, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(factors, factors, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """TODO"""
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class ItemModelIdemb(nn.Module):
    """TODO"""

    def __init__(
        self,
        factors: int,
        n_items: int,
    ):
        super().__init__()

        self.item_emb = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=factors,
            padding_idx=0,
        )

        # TODO to config
        self.drop_layer = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_emb(x)
        item_embs = self.drop_layer(item_embs)

        return item_embs

    def get_device(self) -> torch.device:
        """TODO"""
        return self.item_emb.weight.device


class TransformerEncoder(nn.Module):
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

            new_attn_layer = torch.nn.MultiheadAttention(factors, n_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(factors, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(factors, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(
        self,
        seqs: torch.LongTensor,
        attention_mask: torch.Tensor,
        timeline_mask: torch.Tensor,
    ):
        """TODO"""
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](q, seqs, seqs, attn_mask=attention_mask)
            seqs = q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

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
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.use_pos_emb = use_pos_emb

        self.item_model = ItemModelIdemb(factors, n_items)

        if self.use_pos_emb:
            self.pos_emb = torch.nn.Embedding(session_maxlen, factors)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.encoder = TransformerEncoder(
            n_blocks=n_blocks,
            factors=factors,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
        )
        self.last_layernorm = torch.nn.LayerNorm(factors, eps=1e-8)

    def get_model_device(self) -> torch.device:
        """TODO"""
        return self.item_model.get_device()

    def log2feats(self, sessions: torch.LongTensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings and transformer blocks.

        Returns
        -------
            torch.Tensor. [batch_size, history_len, emdedding_dim]

        """
        seqs = item_embs[sessions]

        if self.use_pos_emb:
            positions = np.tile(np.array(range(sessions.shape[1])), [sessions.shape[0], 1])
            seqs += self.pos_emb(torch.LongTensor(positions).to(self.get_model_device()))

        seqs = self.emb_dropout(seqs)

        timeline_mask = sessions == 0
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        # TODO do we need to mask padding embs attention?
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.get_model_device()))

        seqs = self.encoder(
            seqs=seqs,
            attention_mask=attention_mask,
            timeline_mask=timeline_mask,
        )

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    # TODO check user_ids
    def forward(
        self,
        x: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ):
        """TODO"""
        sessions, item_model_input = x
        # TODO merge item model with log2feats
        item_embs = self.item_model(item_model_input)

        log_feats = self.log2feats(sessions, item_embs)

        logits = log_feats @ item_embs.T

        return logits

    def predict(
        self,
        x: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        item_indices: torch.LongTensor,
    ) -> torch.Tensor:
        # TODO fix docstring
        """
        Inference model function

        Args
        -------
            sessions (np.ndarray): [batch_size, maxlen] users' sessions
            item_indices (np.ndarray): [candidates_count] 1D array of candidate items

        Returns
        -------
            torch.Tensor: [batch_size, candidates_count] logits for each user

        """
        sessions, item_model_input = x

        item_embs = self.item_model(item_model_input)
        log_feats = self.log2feats(sessions, item_embs)

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = item_embs[item_indices]  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # preds # (U, I)


def to_device(x, device: torch.device):
    """TODO"""
    if isinstance(x, torch.Tensor):
        return x.to(device)

    if not isinstance(x, tuple) and not isinstance(x, list):
        raise Exception(f"expected Tuple, List or Tensor, found {type(x)}")

    return tuple(to_device(e, device) for e in x)


class Trainer:
    """TODO"""

    def __init__(
        self,
        model: SASRec,
        lr: float,
        batch_size: int,
        epochs: int,
        device: str,
        loss: str = "bce",
    ):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.loss = loss

        self.optimizer = None
        self.loss_func = None

        self.inited = False

    def init(self):
        """TODO"""
        self._init_optimizers()
        self._init_loss_func()

    def fit(
        self,
        train_dataloader: DataLoader,
    ) -> SASRec:
        """TODO"""
        if not self.inited:
            self.init()

        self.model.to(self.device)

        self.xavier_normal_init(self.model)
        self.model.train()  # enable model training

        epoch_start_idx = 1

        # ce_criterion = torch.nn.CrossEntropyLoss()
        # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...

        iteration = 0
        try:
            for epoch in range(epoch_start_idx, self.epochs + 1):
                logger.info("training epoch %s", epoch)

                for x, y, w in train_dataloader:
                    x = to_device(x, self.device)
                    y = to_device(y, self.device)
                    w = to_device(w, self.device)

                    self.train_step(x, y, w, iteration)

                    iteration += 1

        except KeyboardInterrupt:
            logger.info("training interrupted")

    def train_step(self, x, y, w, iteration):
        """TODO"""
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.loss_func(logits.transpose(1, 2), y)  # loss expects logits in form of [N, C, D1]
        loss = loss * w
        n = (loss > 0).to(loss.dtype)
        loss = torch.sum(loss) / torch.sum(n)
        joint_loss = loss  # + reg_loss
        joint_loss.backward()
        self.optimizer.step()

    def _init_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98))

    def _init_loss_func(self):

        if self.loss == "bce":
            self.loss_func = nn.BCEWithLogitsLoss()
        elif self.loss == "sm_ce":
            self.loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        else:
            raise Exception(f"loss {self.loss} is not supported")

        logger.info("used %s loss", self.loss)

    def xavier_normal_init(self, model: nn.Module) -> None:
        """TODO"""
        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except Exception as err:
                logger.info("undable to init param %s with xavier: %s", name, err)
                pass  # just ignore those failed init layers


class SASRecRecommender:
    """TODO"""

    def __init__(
        self,
        # processor: SASRecProcessor,
        model: SASRec,
        task_converter: SequenceTaskConverter,
        device: Union[torch.device, str] = "cpu",
    ):
        # self.processor = processor
        self.model = model.eval()
        # TODO merge with processor
        self.task_converter = task_converter
        self.device = device

        self.model.to(self.device)

    # TODO what if candidate_items has elements not in item_features
    def recommend(
        self,
        user_item_interactions: pd.DataFrame,
        item_features: pd.Series,
        top_k: int,
        candidate_items: Sequence[str],
        batch_size: int = 128,
    ) -> pd.DataFrame:
        """Accept 3 dataframes from DB and return dataset with recommends"""
        # TODO check that users with unsupported items in history have appropriate embeddings

        # filter out unsupported items
        supported_items = self.get_supported_items(item_features)
        candidate_items = self._filter_candidate_items(candidate_items, supported_items)

        # TODO candidate_items fix private method usage
        candidate_items_inds = list(self.task_converter.item_id_encoder.convert_to_internal(candidate_items))
        candidate_items_inds = torch.LongTensor(candidate_items_inds).to(self.device)

        # TODO here we can filter users completely.
        # We need to find out where to check if we can make recs.
        # If it is here, than just return error object with info about such users.
        user_item_interactions, item_features = self._filter_datasets(
            supported_items=supported_items,
            user_item_interactions=user_item_interactions,
            item_features=item_features,
        )

        # filter out unnecessary items to speed up calculation
        items = user_item_interactions[Columns.Item].drop_duplicates().to_list() + list(candidate_items)
        item_features = item_features[item_features.isin(items)]

        user_item_interactions[Columns.Item] = self.task_converter.item_id_encoder.convert_to_internal(
            user_item_interactions[Columns.Item]
        )
        item_features = pd.Series(self.task_converter.item_id_encoder.convert_to_internal(item_features))

        x = self.task_converter.inference_transform(
            user_item_interactions=user_item_interactions,
            item_features=item_features,
        )

        model_x = SequenceDataset(x=x, batch_size=batch_size)
        dataloader = DataLoader(model_x, batch_size=None)  # batching in dataset

        logits = []
        device = self.model.get_model_device()
        with torch.no_grad():
            for x_batch in tqdm.tqdm(dataloader):
                x_batch = to_device(x_batch, device)
                logits_batch = self.model.predict(x_batch, candidate_items_inds).detach().cpu().numpy()
                logits.append(logits_batch)

        logits = np.concatenate(logits, axis=0)
        user_item_interactions[Columns.Item] = self.task_converter.item_id_encoder.convert_to_external(
            user_item_interactions[Columns.Item]
        )
        recs = self.collect_recs(
            user_item_interactions=user_item_interactions,
            candidate_items=candidate_items,
            users=x.users,
            logits=logits,
            top_k=top_k,
        )
        return recs

    # TODO move implementation to processor
    def get_supported_items(self, item_features: pd.Series) -> List[str]:
        """TODO"""
        supported_ids = set(self.task_converter.item_id_encoder.get_external_sorted_by_internal())
        mask = item_features.isin(supported_ids)

        item_features = item_features[mask]

        return item_features.to_list()

    def _filter_candidate_items(self, candidate_items: Sequence[str], supported_items: Sequence[str]) -> List[str]:
        supported_candidate_items = list(set(candidate_items).intersection(supported_items))

        filtered_items_cnt = len(candidate_items) - len(supported_candidate_items)
        if filtered_items_cnt > 0:
            logger.warning("filtered out %s candidate items which are not supported by model", filtered_items_cnt)

        return supported_candidate_items

    def _filter_datasets(
        self,
        supported_items: Sequence[str],
        user_item_interactions: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        item_features = item_features[item_features.isin(supported_items)]
        user_item_interactions = user_item_interactions[user_item_interactions[Columns.Item].isin(supported_items)]

        return user_item_interactions, item_features

    def collect_recs(
        self,
        user_item_interactions: pd.DataFrame,
        candidate_items: List[str],
        users: List[str],
        logits: np.ndarray,  # [users, candidate_items]
        top_k: int,
    ):
        """TODO"""
        user2hist = user_item_interactions.groupby(Columns.User)[Columns.Item].agg(set).to_dict()
        candidate_items = np.array(candidate_items)
        inds = np.argsort(-logits, axis=1)

        records = []
        for i, user in enumerate(tqdm.tqdm(users)):
            cur_hist = user2hist[user]
            cur_top_k = top_k + len(cur_hist)

            cur_recs = candidate_items[inds[i, :cur_top_k]]
            cur_scores = logits[i, inds[i, :cur_top_k]]
            mask = [rec not in cur_hist for rec in cur_recs]
            cur_recs = list(compress(cur_recs, mask))[:top_k]
            cur_scores = list(compress(cur_scores, mask))[:top_k]

            records.append(
                (
                    user,
                    cur_recs,
                    cur_scores,
                )
            )

        recs = pd.DataFrame(
            records,
            columns=[
                Columns.User,
                Columns.Item,
                Columns.Weight,
            ],
        ).explode([Columns.Item, Columns.Weight])
        recs[Columns.Rank] = recs.groupby(Columns.User).cumcount() + 1

        return recs
