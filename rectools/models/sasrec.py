import logging
import typing as tp
from itertools import compress
from typing import List, Sequence, Tuple, Union

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
        item_net_dropout_rate: float,
        use_pos_emb: bool = True,
        loss: str = "softmax",
        verbose: int = 0,
        random_state: int = False,
    ):
        super().__init__(verbose=verbose)
        self.session_maxlen = session_maxlen
        self.batch_size = batch_size
        self.n_blocks = n_blocks
        self.factors = factors
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.item_net_dropout_rate = item_net_dropout_rate
        self.use_pos_emb = use_pos_emb
        self.model: SASRec
        self.item_id_map: IdMap
        self.trainer = Trainer(
            lr=lr,
            batch_size=self.batch_size,
            epochs=epochs,
            device=device,
            loss=loss,
        )
        if random_state is not None:
            torch.use_deterministic_algorithms(True)
            seed_everything(random_state, workers=True)

    def _fit(
        self,
        dataset: RecDataset,
    ) -> None:
        """TODO"""
        user_item_interactions = dataset.get_raw_interactions()

        users = user_item_interactions[Columns.User].value_counts()
        users = users[users >= 2]
        user_item_interactions = user_item_interactions[(user_item_interactions[Columns.User].isin(users.index))]

        user_item_interactions = (
            user_item_interactions.sort_values(Columns.Datetime).groupby(Columns.User).tail(self.session_maxlen + 1)
        )

        items = np.unique(user_item_interactions[Columns.Item])
        self.item_id_map = IdMap.from_values("PAD")
        self.item_id_map = self.item_id_map.add_ids(items)
        user_item_interactions[Columns.Item] = self.item_id_map.convert_to_internal(
            user_item_interactions[Columns.Item]
        )

        train_dataset = SequenceDataset.from_interactions(user_item_interactions=user_item_interactions)
        fit_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: collate_fn_train(batch, self.session_maxlen),
        )

        # init model
        logger.info("building model")
        self.model = SASRec(
            n_blocks=self.n_blocks,
            factors=self.factors,
            n_heads=self.n_heads,
            session_maxlen=self.session_maxlen,
            dropout_rate=self.dropout_rate,
            use_pos_emb=self.use_pos_emb,
            n_items=self.item_id_map.size,
            item_net_dropout_rate=self.item_net_dropout_rate,
        )

        logger.info("building trainer")
        self.trainer.fit(self.model, fit_dataloader)
        self.model = self.trainer.model

    def recommend(
        self,
        users: AnyIds,
        dataset: RecDataset,
        k: int,
        filter_viewed: bool = False,
        items_to_recommend: tp.Optional[AnyIds] = None,
        add_rank_col: bool = True,
        assume_external_ids: bool = True,
    ) -> pd.DataFrame:
        """TODO"""
        train = dataset.get_raw_interactions()

        item_features = train[Columns.Item].drop_duplicates()

        rec_df = train[train[Columns.User].isin(users)]
        recommender = SASRecRecommender(self.model, self.item_id_map)

        if items_to_recommend is None:
            items_to_recommend = item_features

        recs = recommender.recommend(
            user_item_interactions=rec_df,
            item_features=item_features,
            top_k=k,
            candidate_items=items_to_recommend,
        )
        return recs


# TODO: think about moving all data processing to SequenceDataset.from_dataset method.
# This would also mean creating different dataset types for fit and recommend
# If we have different dataset types for fit and recommend,
# we can also split to x/y/w and pad with zeros in dataset, not dataloader


class SequenceDataset(Dataset):
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
        user_item_interactions: pd.DataFrame,
    ) -> "SequenceDataset":
        """TODO"""
        sessions = (
            user_item_interactions.sort_values(Columns.Datetime)
            .groupby(Columns.User)[[Columns.Item, Columns.Weight]]
            .agg(list)
        )
        sessions, weights = (
            sessions[Columns.Item].to_list(),
            sessions[Columns.Weight].to_list(),
        )

        return cls(sessions=sessions, weights=weights)


def collate_fn_train(
    batch: List[Tuple[List[int], List[float]]],
    session_maxlen: int,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """TODO"""
    sessions = list(list(zip(*batch))[0])  # [batch_size, session_len]
    weights = list(list(zip(*batch))[1])  # [batch_size, session_len]

    x = np.zeros((len(sessions), session_maxlen))  # [batch_size, session_maxlen]
    y = np.zeros((len(sessions), session_maxlen))  # [batch_size, session_maxlen]
    yw = np.zeros((len(sessions), session_maxlen))  # [batch_size, session_maxlen]
    # left padding
    for i, (ses, ses_weights) in enumerate(zip(sessions, weights)):
        x[i, -len(ses) + 1 :] = ses[:-1]  # ses: [session_len] -> x[i]: [session_maxlen]
        y[i, -len(ses) + 1 :] = ses[1:]  # ses: [session_len] -> y[i]: [session_maxlen]
        yw[i, -len(ses) + 1 :] = ses_weights[1:]  # ses_weights: [session_len] -> yw[i]: [session_maxlen]

    return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(yw)


def collate_fn_recommend(
    batch: List[Tuple[List[int], List[float]]],
    session_maxlen: int,
) -> torch.LongTensor:
    """TODO"""
    sessions = list(list(zip(*batch))[0])

    x = np.zeros((len(sessions), session_maxlen))
    # left padding, left truncation
    for i, ses in enumerate(sessions):
        x[i, -len(ses) :] = ses[-session_maxlen:]

    return torch.LongTensor(x)


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

        session_embs = self.last_layernorm(seqs)  # [batch_size, session_maxlen, factors]

        return session_embs

    # TODO check user_ids
    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_maxlen]
    ) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_model.get_all_embeddings()  # [n_items + 1, factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_maxlen, factors]
        logits = session_embs @ item_embs.T  # [batch_size, session_maxlen, n_items + 1]
        return logits

    def predict(
        self,
        sessions: torch.Tensor,
        item_indices: torch.Tensor,
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
        item_embs = self.item_model.get_all_embeddings()  # [n_items + 1, factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_maxlen, factors]

        final_feat = session_embs[:, -1, :]  # only use last QKV classifier, a waste [batch_size, factors]

        item_embs = item_embs[item_indices]  # [n_items, factors]

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # [batch_size, n_items]

        return logits


class Trainer:
    """TODO"""

    def __init__(
        self,
        lr: float,
        batch_size: int,
        epochs: int,
        device: str,
        loss: str = "softmax",
    ):
        self.model: SASRec
        self.optimizer: torch.optim.Adam
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)
        self.loss_func = self._init_loss_func(loss)

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


class SASRecRecommender:
    """TODO"""

    def __init__(
        self,
        model: SASRec,
        item_id_map: IdMap,
        device: Union[torch.device, str] = "cuda:1",
    ):
        self.item_id_map = item_id_map
        self.model = model.eval()
        self.device = device

        self.model.to(self.device)

    # TODO what if candidate_items has elements not in item_features
    def recommend(
        self,
        user_item_interactions: pd.DataFrame,
        item_features: pd.Series,
        top_k: int,
        candidate_items: pd.Series,
        batch_size: int = 128,
    ) -> pd.DataFrame:
        """Accept 3 dataframes from DB and return dataset with recommends"""
        # TODO check that users with unsupported items in history have appropriate embeddings

        # filter out unsupported items
        supported_items = self.get_supported_items(item_features)
        candidate_items = self._filter_candidate_items(candidate_items, supported_items)

        # TODO candidate_items fix private method usage
        candidate_items_inds = torch.LongTensor(self.item_id_map.convert_to_internal(candidate_items)).to(self.device)

        # TODO here we can filter users completely.
        # We need to find out where to check if we can make recs.
        # If it is here, than just return error object with info about such users.
        user_item_interactions, item_features = self._filter_datasets(
            supported_items=supported_items,
            user_item_interactions=user_item_interactions,
            item_features=item_features,
        )

        user_item_interactions[Columns.Item] = self.item_id_map.convert_to_internal(
            user_item_interactions[Columns.Item]
        )

        model_x = SequenceDataset.from_interactions(user_item_interactions=user_item_interactions)
        recommend_dataloader = DataLoader(
            model_x,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn_recommend(batch, self.model.session_maxlen),
        )

        logits = []
        device = self.model.get_model_device()
        self.model.item_model.to_device(device)
        with torch.no_grad():
            for x_batch in tqdm.tqdm(recommend_dataloader):
                x_batch = x_batch.to(device)  # [batch_size, session_maxlen]
                logits_batch = (
                    self.model.predict(x_batch, candidate_items_inds).detach().cpu().numpy()
                )  # [batch_size, n_items]
                logits.append(logits_batch)

        logits_array = np.concatenate(logits, axis=0)  # [n_users, n_items]
        user_item_interactions[Columns.Item] = self.item_id_map.convert_to_external(
            user_item_interactions[Columns.Item]
        )
        users = user_item_interactions[Columns.User].drop_duplicates().sort_values()
        recs = self.collect_recs(
            user_item_interactions=user_item_interactions,
            candidate_items=candidate_items,
            users=users,
            logits=logits_array,
            top_k=top_k,
        )
        return recs

    # TODO move implementation to processor
    def get_supported_items(self, item_features: pd.Series) -> List[int]:
        """TODO"""
        supported_ids = set(self.item_id_map.get_external_sorted_by_internal())
        mask = item_features.isin(supported_ids)

        item_features = item_features[mask]

        return item_features.to_list()

    def _filter_candidate_items(self, candidate_items: Sequence[int], supported_items: Sequence[int]) -> List[int]:
        supported_candidate_items = list(set(candidate_items).intersection(supported_items))
        filtered_items_cnt = len(candidate_items) - len(supported_candidate_items)
        if filtered_items_cnt > 0:
            logger.warning("filtered out %s candidate items which are not supported by model", filtered_items_cnt)

        return supported_candidate_items

    def _filter_datasets(
        self,
        supported_items: Sequence[int],
        user_item_interactions: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        item_features = item_features[item_features.isin(supported_items)]
        user_item_interactions = user_item_interactions[user_item_interactions[Columns.Item].isin(supported_items)]

        return user_item_interactions, item_features

    def collect_recs(
        self,
        user_item_interactions: pd.DataFrame,
        candidate_items: List[int],
        users: List[str],
        logits: np.ndarray,  # [users, candidate_items]
        top_k: int,
    ) -> pd.DataFrame:
        """TODO"""
        user2hist = user_item_interactions.groupby(Columns.User)[Columns.Item].agg(set).to_dict()
        candidate_items_array = np.array(candidate_items)
        inds = np.argsort(-logits, axis=1)

        records = []
        for i, user in enumerate(tqdm.tqdm(users)):
            cur_hist = user2hist[user]
            cur_top_k = top_k + len(cur_hist)

            cur_rec = candidate_items_array[inds[i, :cur_top_k]]
            cur_scores = logits[i, inds[i, :cur_top_k]]
            mask = [rec not in cur_hist for rec in cur_rec]
            cur_recs = list(compress(cur_rec, mask))[:top_k]
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
