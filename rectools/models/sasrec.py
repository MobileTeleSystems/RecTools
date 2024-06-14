import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, Protocol, TypeVar
import pydantic
import os
import json
import logging
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import math
from tensorboardX import SummaryWriter
from datetime import date, timedelta
import enum
import bisect
from scipy.stats import norm
import tqdm
from itertools import compress
import random
from rectools.metrics.novelty import MeanInvUserFreq
from rectools.metrics.ranking import MAP
from rectools.metrics.serendipity import Serendipity
from collections import Counter
from rectools.columns import Columns
from rectools.metrics.distances import (Distances, ExternalIds,
                                        PairwiseDistanceCalculator)



logger = logging.getLogger(__name__)


class BaseConfig(pydantic.BaseModel):
    def save(self, filepath: str):
        filepath = self._preprocess_filepath(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        cfg_dict = self.dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f)

    @classmethod
    def load(cls, filepath: str):
        filepath = cls._preprocess_filepath(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        return cls(**cfg)

    @classmethod
    def _preprocess_filepath(cls, filepath: str):
        ending = ".json"

        if not filepath.endswith(ending):
            filepath = filepath + ending

        return filepath


class DatasetStats(BaseConfig):
    item_num: int
    user_num: int
    item_tags_num:int


@dataclass
class SequenceTaskData:
    users: List[str] # length is equal to len(sessions)
    items: List[str] # length is equal to len(item_model_input[0])
    sessions: List[List[str]]
    weights: List[List[float]]
    item_model_input: Tuple[List[str], List[List[str]]] # TODO maybe replace with named tuple


@dataclass
class SequenceTaskTarget:
    next_watch: List[List[str]]
    weights: List[List[float]]


@dataclass
class SASRecModelInput:
    """
        sessions (torch.LongTensor): [batch_size, maxlen] user history
        item_model_input (Tuple[torch.LongTensor, torch.LongTensor]): item features input to construct
    """
    sessions: torch.LongTensor 
    item_model_input:Tuple[torch.LongTensor, torch.LongTensor]
    items: List[str]
    users: List[str]


@dataclass
class SASRecTargetInput:
    next_watch: torch.LongTensor
    weights: Optional[torch.LongTensor]=None


class SequenceTaskConverterConfig(BaseConfig):
    min_score:Optional[float] = None


class SequenceTaskConverter:
    """Convert database data to use in particular model"""

    def __init__(self, config:Optional[SequenceTaskConverterConfig]=None):
        if config is None:
            config = SequenceTaskConverterConfig()

        self.config = config

    def train_transform(
        self,
        user_item_interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> Tuple[SequenceTaskData, SequenceTaskTarget]:
        """converts user-item interaction to sessions and extracts target.
        Prepares other features"""
        
        users, sessions, weights = self._interactions2sessions(user_item_interactions)
        sessions_x, sessions_y, x_weights, y_weights = self._sessions2xy(sessions, weights)

        items, item_model_input = self._prepare_items_feautes(item_features)

        return (
            SequenceTaskData(
                users=users, 
                items=items,
                sessions=sessions_x,
                weights=x_weights,
                item_model_input=item_model_input,
            ),
            SequenceTaskTarget(
                next_watch=sessions_y,
                weights=y_weights,
            )
        )
    
    # TODO What if sessions contains items not present in item features?
    # TODO What if user's consist only of items not known by the model?
    def inference_transform(
        self,
        user_item_interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> SequenceTaskData:
        """converts user-item interaction to sessions and prepares other features"""
        users, sessions, weights = self._interactions2sessions(user_item_interactions)

        items, item_model_input = self._prepare_items_feautes(item_features)


        return SequenceTaskData(
                users=users, 
                items=items,
                sessions=sessions,
                weights=weights,
                item_model_input=item_model_input,
            )
    
    def save(self, dirpath:str):
        self.config.save(os.path.join(dirpath, "task_converter.pkl"))
    
    @classmethod
    def load(cls, dirpath:str) -> "SequenceTaskConverter":
        config = SequenceTaskConverterConfig.load(os.path.join(dirpath, "task_converter.pkl"))
        
        return SequenceTaskConverter(config)

    def _interactions2sessions(self, user_item_interactions:pd.DataFrame) -> Tuple[List[str], List[List[str]], List[List[float]]]:
        if self.config.min_score is not None:
            logger.info("filter out interactions with score lower than %s", self.config.min_score)
            user_item_interactions = user_item_interactions[user_item_interactions["score"] >= self.config.min_score]

        sessions = (
            user_item_interactions.sort_values("first_intr_dt")
            .groupby("user_id")[["item_id", "score"]]
            .agg(list)
        )
        users, sessions, weights = (
            sessions.index.to_list(), 
            sessions["item_id"].to_list(), 
            sessions["score"].to_list(),
        )

        # TODO remove log here
        lens = [len(ses) for ses in sessions]

        logger.info(
            "sessions lens: 0.95q: %s; 0.5q: %s",
            np.quantile(lens, 0.95),
            np.quantile(lens, 0.5),
        )

        return users, sessions, weights
    
    def _sessions2xy(
        self, 
        sessions:List[List[str]], 
        weights:List[List[float]],
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

    def _prepare_items_feautes(self, item_features:pd.DataFrame) -> Tuple[List[str], Tuple[List[str], List[List[str]]]]:
        item_features = item_features.sort_values('item_id')

        items = item_features['item_id'].to_list()
        item_tags = item_features['tags_set'].to_list()
        
        return items, (items, item_tags)

class SequenceDataset(Dataset):
    def __init__(
        self, 
        x: SASRecModelInput, 
        batch_size:int,
        y:Optional[SASRecTargetInput]=None, 
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
                (
                self.x.sessions[start:end],
                self.x.item_model_input,
                )
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

    
class TrainPreprocessingConfig(BaseConfig):
    min_item_freq:int
    min_user_freq:int
    keep_tags_types: Optional[List[str]]=None


def preprocess_train_datasets(
    config:TrainPreprocessingConfig,
    user_item_interactions: pd.DataFrame,
    item_features: Optional[pd.DataFrame]=None,
    user_features: Optional[pd.DataFrame]=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess training data.
        - filter users and items by frequency
        - discard features of users and items which did not appear in interactions

    Args:
        config (TrainPreprocessingConfig): _description_
        user_item_interactions (pd.DataFrame): _description_
        item_features (Optional[pd.DataFrame], optional): _description_. Defaults to None.
        user_features (Optional[pd.DataFrame], optional): _description_. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (user_item_interactions, item_features, user_features)
    """
    # TODO remove users/items until both constraints are satisfied
    items = user_item_interactions['item_id'].value_counts()
    items = items[items >= config.min_item_freq]
    items = items.index.to_list()
    
    # TODO 
    user_item_interactions = user_item_interactions[user_item_interactions['item_id'].isin(items)]
    
    users = user_item_interactions['user_id'].value_counts()
    users = users[users >= config.min_user_freq]
    users = users.index.to_list()

    user_item_interactions = user_item_interactions[user_item_interactions['user_id'].isin(users)]
    # user_item_interactions = user_item_interactions[
    #     user_item_interactions['item_id'].isin(items) 
    #     & user_item_interactions['user_id'].isin(users)
    # ]

    # TODO recompute users/items as far as some of them may be dropped
    items = user_item_interactions['item_id'].drop_duplicates().to_list()
    users = user_item_interactions['user_id'].drop_duplicates().to_list()

    if item_features is not None:
        item_features = item_features[item_features['item_id'].isin(items)]
    if user_features is not None:
        user_features = user_features[user_features['user_id'].isin(users)]

    # keep only tags with particular type
    if config.keep_tags_types is not None:
        item_features = item_features.copy()
        item_features['tags_set'] = item_features['tags_set'].apply(
            lambda x: [tag for tag in x if any([tag.startswith(tag_type) 
                                                for tag_type in config.keep_tags_types])]
        )

    return (
        user_item_interactions, 
        item_features, 
        user_features,
    )


def test_processed_train_datasets(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
): 
    items = user_item_interactions['item_id'].drop_duplicates().to_list()
    users = user_item_interactions['user_id'].drop_duplicates().to_list()

    # same set of items in interactions and features
    if item_features is not None:
        assert set(items) == set(item_features['item_id'].drop_duplicates())

    # same set of users in interactions and features
    if user_features is not None:
        assert set(users) == set(user_features['user_id'].drop_duplicates())

    # at least 2 items per user's session. It is required for target building
    assert (user_item_interactions['user_id'].value_counts() > 1).all()


def save_pickle(obj, filepath: str):
    dirname = os.path.dirname(filepath)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
        
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    return obj

@dataclass
class SASRecProcessorConfig:
    session_maxlen:int
    enable_item_features: bool = False
    item_tags_maxlen: Optional[int] = None
    item_tags_min_frequency: int = 1
    
 
class Tokenizer:
    def __init__(self):
        self.vocab = None
        self.keys = None
        self.key2index = None

    def fit(self, tokens: Sequence[str]) -> None:
        vocab = set(tokens)
        self._init_mappers(vocab)

    def transform(self, tokens: Sequence[str]) -> np.ndarray:
        """Converts tokens to ids. Unknown tokens replaced with <PAD> (0 id)
        """
        if not self._is_built():
            raise Exception("call `fit` first")
        
        encoded_tokens = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
            if token in self.key2index:
                encoded_tokens[i] = self.key2index[token]

        return encoded_tokens
    
    def fit_transform(self, tokens: Sequence[str]) -> np.ndarray:
        self.fit(tokens)
        return self.transform(tokens)
    
    def inverse_transform(self, encoded_tokens: Sequence[int]) -> np.ndarray:
        tokens = []
        for enc_token in encoded_tokens:
            if enc_token not in self.key2index:
                raise Exception(f"id {enc_token} not in index")
            
            tokens.append(self.key2index[enc_token])

        tokens = np.array(tokens)

        return tokens

    def tokens(self) -> np.ndarray:
        return self.keys

    def save(self, f:str):
        save_pickle(self, f)
    
    @classmethod
    def load(cls, f:str) -> "Tokenizer":
        return load_pickle(f)
    
    def __len__(self):
        return len(self.keys)
    
    def _init_mappers(self, vocab:Sequence[str]):
        self.vocab = vocab
        self.keys = np.array(['<PAD>', *self.vocab])
        self.key2index = {e: i for i, e in enumerate(self.keys)}
    
    def _is_built(self) -> bool:
        return self.vocab is not None \
            and self.keys is not None \
            and self.key2index is not None


class TagsProcessor:
    def __init__(self, tokenizer:Tokenizer, maxlen:int):
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def fit(self, tokens: Sequence[Sequence[str]]) -> None:
        flat_tokens = [e for seq in tokens for e in seq]
        self.tokenizer.fit(flat_tokens)

    def transform(self, tokens: Sequence[Sequence[str]]) -> np.ndarray:
        encoded_tokens = np.stack(
            [self.pad_turnc(self.tokenizer.transform(seq)) for seq in tokens],
            axis=0,
        )

        return encoded_tokens

    def fit_transform(self, tokens: Sequence[Sequence[str]]) -> np.ndarray:
        self.fit(tokens)
        return self.transform(tokens)

    # def inverse_transform(
    #     self, encoded_tokens: Sequence[Sequence[int]]
    # ) -> List[List[str]]:
    #     pad_id = None
    #     if self.tokenizer.padding is not None:
    #         pad_id = self.tokenizer.padding["pad_id"]

    #     tokens = [
    #         [self.tokenizer.id_to_token(e) for e in s if e != pad_id]
    #         for s in encoded_tokens
    #     ]

    #     return tokens
    
    # TODO make normal tokenizer with ignoring unknown tags before converting to indexes
    def pad_turnc(self, x:np.ndarray):
        # remove zeros to avoid problem with unknown tags overflowing maxlen
        x = np.array([e for e in x if e != 0], dtype=x.dtype)
        
        if len(x) >= self.maxlen:
            # left trunc
            x = x[-self.maxlen:]

        else:
            # left padding
            x = np.concatenate(
                [
                    np.zeros(self.maxlen - len(x)),
                    x,
                ],
                axis=0,
            )

        return x
    
    def tokens(self) -> np.ndarray:
        return self.tokenizer.tokens()
    
    def __len__(self):
        return len(self.tokenizer)
    
    def save(self, f:str):
        save_pickle(self, f)
    
    @classmethod
    def load(cls, f:str) -> "TagsProcessor":
        return load_pickle(f)

    @classmethod
    def build(cls, maxlen: int):
        return cls(
            tokenizer=Tokenizer(),
            maxlen=maxlen,
        )
    

class SASRecProcessor:
    def __init__(self, 
        config: SASRecProcessorConfig,
        item_id_encoder: Optional[Tokenizer] = None,
        item_tags_encoder: Optional[TagsProcessor] = None,
    ):
        self.config = config
        self.item_id_encoder = item_id_encoder
        self.item_tags_encoder = item_tags_encoder
    
    def fit(self, x:SequenceTaskData) -> None:
        self.item_id_encoder.fit(x.item_model_input[0])
        
        if self.config.enable_item_features:
            self.item_tags_encoder.fit(x.item_model_input[1])

    def transform(self, x: SequenceTaskData) -> SASRecModelInput:
        """Returns data batch in right format for model. Possible very large batch containing all dataset
        
            Returns SASRecModelInput. sessions of size [batch_size, maxlen], 
                model input of size ([items_cnt], [items_cnt, item_tags_maxlen])
        """
        item_mapping = self._build_items_mapping(x.items)
        
        # TODO use weights in model
        sessions, _ = self._sessions_transform(x.sessions, x.weights, item_mapping)
        sessions = torch.LongTensor(sessions)

        item_model_input = self._build_item_model_input(x)

        return SASRecModelInput(
            sessions=sessions,
            item_model_input=item_model_input,
            items=x.items,
            users=x.users,
        )

    # def fit_transform(self, x: SequenceTaskData) -> SASRecModelInput:
    #     self.fit(x)
    #     return self.transform(x)

    def dataset_stats(self) -> DatasetStats:
        """Returns stats about mappers for model config"""

        return DatasetStats(
            item_num=len(self.item_id_encoder) if self.item_id_encoder is not None else -1,
            user_num=-1,
            item_tags_num=len(self.item_tags_encoder) if self.item_tags_encoder is not None else -1,
        )

    def transform_target(self, y:SequenceTaskTarget, model_input:SASRecModelInput) -> SASRecTargetInput:
        """replace target ids with internal integer id. Assumed that no issues with missing mappings are possible"""
        item_mapping = self._build_items_mapping(model_input.items)

        next_watch, weights = self._sessions_transform(y.next_watch, y.weights, item_mapping)
        next_watch = torch.LongTensor(next_watch)
        weights = torch.FloatTensor(weights)

        return SASRecTargetInput(
            next_watch=next_watch,
            weights=weights,
        )

    def save(self, dirpath:str):
        save_pickle(self, os.path.join(dirpath, "processor.pkl"))
    
    @classmethod
    def load(cls, dirpath:str) -> "SASRecProcessor":
        return load_pickle(os.path.join(dirpath, "processor.pkl"))


    def _sessions_transform(
            self, 
            sessions:List[List[str]], 
            weights:List[List[float]],  
            item_mapping:Dict[str,int],
        ) -> np.ndarray:
        x = np.zeros((len(sessions), self.config.session_maxlen))
        w = np.zeros((len(sessions), self.config.session_maxlen))
        # left padding, left truncation
        for i, (ses, ses_w)  in enumerate(zip(sessions, weights)):
            cur_ses = ses[-self.config.session_maxlen:]
            cur_w = ses_w[-self.config.session_maxlen:]

            cur_ses = [item_mapping[e] for e in cur_ses]
            x[i, -len(cur_ses):] = cur_ses
            w[i, -len(cur_ses):] = cur_w
        
        return x, w

    def _build_items_mapping(self, items: List[str]) -> Dict[str, int]:
        mp = {e: i+1 for i, e in enumerate(items)} # 0 is a PAD token
        return mp

    # TODO checking enable_item_features all the time looks weird
    def _build_item_model_input(self, x:SequenceTaskData):
        id_x = self.item_id_encoder.transform(x.item_model_input[0])
        id_x = np.concatenate([np.zeros_like(id_x[:1]), id_x], axis=0) # for PAD token

        if self.config.enable_item_features:
            tags_x = self.item_tags_encoder.transform(x.item_model_input[1])
            tags_x = np.concatenate([np.zeros_like(tags_x[:1]), tags_x], axis=0) # for PAD token

        if self.config.enable_item_features:
            return (
                torch.LongTensor(id_x),
                torch.LongTensor(tags_x),
            )

        return torch.LongTensor(id_x)

    @classmethod
    def build(cls, config: SASRecProcessorConfig):
        item_id_encoder = Tokenizer()

        item_tags_encoder = None
        if config.enable_item_features:
            item_tags_encoder = TagsProcessor.build(
                maxlen=config.item_tags_maxlen,
            )

        return cls(
            config=config,
            item_id_encoder=item_id_encoder,
            item_tags_encoder=item_tags_encoder,
        )



# TODO how to make config reusable for all item models?
class ItemModelConfig(BaseConfig):
    name:str
    hidden_units: int


# TODO make list of all models and list it during error
def build_item_model(cfg:ItemModelConfig, dataset_stats:DatasetStats):
    if cfg.name == "idemb":
        return ItemModelIdemb.from_config(cfg, dataset_stats)

    raise TypeError(f"model {cfg.name} not supported")


def masked_mean(x:torch.Tensor, mask: torch.Tensor, dim:int):
    divisors = mask.sum(dim=-1)
    divisors[divisors == 0] = 1 # to avoid NaN on paddings
    
    x_agg = torch.sum(x * mask.unsqueeze(-1) , dim=dim) / divisors.unsqueeze(-1)

    return x_agg


# TODO adpat to config creation
class ItemModel(nn.Module):
    def __init__(self, embedding_dim:int, item_num:int, item_tags_num:int):
        super().__init__()
        self.item_num = item_num
        self.item_tags_num = item_tags_num

        self.item_emb = nn.Embedding(
            num_embeddings=self.item_num,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.item_tags_emb = nn.Embedding(
            num_embeddings=self.item_tags_num,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        # TODO to config
        self.drop_layer = nn.Dropout(0.2)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """

        Args:
            x (Tuple[torch.Tensor, torch.Tensor]): (CF ids [batch], tags ids)

        Returns torch.Tensor: nn.Embedding like output
        """

        item_ids, item_tags_ids = x

        item_embs = self.item_emb(item_ids)  # [batch_size, maxlen, emb_dim]
        item_tags_embs = self.item_tags_emb(item_tags_ids)  # [batch_size, maxlen, tag_maxlen, emb_dim]
        # # TODO return back
        # all_ids = torch.concat([item_ids.unsqueeze(-1), item_tags_ids], dim=-1)
        # all_embs = torch.concat([item_embs.unsqueeze(-2), item_tags_embs], dim=-2)
        # # all_ids = item_tags_ids
        # # all_embs = item_tags_embs
        # # all_ids = item_ids.unsqueeze(-1)
        # # all_embs = item_embs.unsqueeze(-2)


        # mask = (all_ids != 0).to(all_embs.dtype)
        # divisors = mask.sum(dim=-1)
        # divisors[divisors == 0] = 1 # to avoid NaN on paddings
        
        # # it is hard to guarantee that pad embeddings equal to zero vector, so just mask it 
        # all_embs = torch.sum(all_embs * mask.unsqueeze(-1) , dim=-2) / divisors.unsqueeze(-1)

        # id
        item_ids_mask = (item_ids != 0).to(item_embs.dtype)
        item_embs = item_embs * item_ids_mask.unsqueeze(-1)
        # tags
        item_tags_mask = (item_tags_ids != 0).to(item_tags_embs.dtype)
        item_tags_agg = masked_mean(item_tags_embs, item_tags_mask, dim=-2)

        # dropout
        item_embs = self.drop_layer(item_embs)
        item_tags_agg = self.drop_layer(item_tags_agg)

        all_embs = item_embs + item_tags_agg

        return all_embs
    
    def get_device(self) -> torch.device:
        return self.item_emb.weight.device


class ItemModelIdemb(nn.Module):
    def __init__(self, embedding_dim:int, item_num:int):
        super().__init__()
        self.item_num = item_num

        self.item_emb = nn.Embedding(
            num_embeddings=self.item_num,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # TODO to config
        self.drop_layer = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        item_ids = x

        item_embs = self.item_emb(item_ids)
        item_embs = self.drop_layer(item_embs)

        return item_embs
    
    def get_device(self) -> torch.device:
        return self.item_emb.weight.device
    
    @classmethod
    def from_config(cls, cfg:ItemModelConfig, dataset_stats:DatasetStats):
        return cls(
            embedding_dim=cfg.hidden_units,
            item_num=dataset_stats.item_num,
        )



class SASRecConfig(BaseConfig):
    num_blocks: int
    hidden_units: int
    num_heads: int
    maxlen: int
    dropout_rate: float
    item_model: ItemModelConfig
    use_pos_emb: bool = True
    use_sm_head: bool = False


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    
    
def xavier_normal_init(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception as err:
            logger.info("undable to init param %s with xavier: %s", name, err)
            pass  # just ignore those failed init layers


def to_device(x, device:torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)

    if not isinstance(x, tuple) and not isinstance(x, list):
        raise Exception(f"expected Tuple, List or Tensor, found {type(x)}")
    
    return tuple(to_device(e, device) for e in x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        config: SASRecConfig,
    ):
        super().__init__()
        self.cfg = config

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(self.cfg.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.cfg.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                self.cfg.hidden_units, self.cfg.num_heads, self.cfg.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.cfg.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.cfg.hidden_units, self.cfg.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(
        self,
        seqs:torch.LongTensor, 
        attention_mask:torch.Tensor, 
        timeline_mask:torch.Tensor,
    ):
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        return seqs

# TODO merge common logic to some universal model
class SASRec(torch.nn.Module):
    def __init__(
        self,
        config: SASRecConfig,
        dataset_stats: DatasetStats,
    ):
        super().__init__()
        self.cfg = config
        self.dataset_stats = dataset_stats
        
        self.item_model = build_item_model(config.item_model, dataset_stats)
        # self.item_model = ItemModel(
        #     self.cfg.hidden_units,
        #     item_num=self.cfg.item_num,
        #     item_tags_num=self.cfg.item_tags_num,
        # )
        
        if self.cfg.use_pos_emb:
            self.pos_emb = torch.nn.Embedding(
                self.cfg.maxlen, self.cfg.hidden_units
            )
        self.emb_dropout = torch.nn.Dropout(p=self.cfg.dropout_rate)

        self.encoder = TransformerEncoder(config)
        self.last_layernorm = torch.nn.LayerNorm(self.cfg.hidden_units, eps=1e-8)

    def get_model_device(self) -> torch.device:
        return self.item_model.get_device()

    def log2feats(self, sessions:torch.LongTensor, item_embs:torch.Tensor) -> torch.Tensor:
        """Pass user history through item embeddings and transformer blocks.

        Returns:
            torch.Tensor. [batch_size, history_len, emdedding_dim]
        """

        seqs = item_embs[sessions]

        if self.cfg.use_pos_emb:
            positions = np.tile(
                np.array(range(sessions.shape[1])), [sessions.shape[0], 1]
            )
            seqs += self.pos_emb(
                torch.LongTensor(positions).to(self.get_model_device())
            )

        seqs = self.emb_dropout(seqs)

        timeline_mask = sessions == 0
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        # TODO do we need to mask padding embs attention?
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.get_model_device())
        )

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
        sessions, item_model_input = x
        # TODO merge item model with log2feats
        item_embs = self.item_model(item_model_input)

        log_feats = self.log2feats(sessions, item_embs)

        logits = log_feats @ item_embs.T

        return logits

    def predict(self, 
                x: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                item_indices:torch.LongTensor,
    ) -> torch.Tensor:
        # TODO fix docstring
        """ Inference model function

        Args:
            sessions (np.ndarray): [batch_size, maxlen] users' sessions
            item_indices (np.ndarray): [candidates_count] 1D array of candidate items

        Returns:
            torch.Tensor: [batch_size, candidates_count] logits for each user
        """

        sessions, item_model_input = x

        item_embs = self.item_model(item_model_input)
        log_feats = self.log2feats(sessions, item_embs)

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = item_embs[item_indices]  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # preds # (U, I)

    def save(self, dirpath:str):
        self.cfg.save(os.path.join(dirpath, "model_cfg.json"))
        self.dataset_stats.save(os.path.join(dirpath, "dataset_stats.json"))
        torch.save(self.state_dict(), os.path.join(dirpath, "model_state_dict.pt"))
    
    @classmethod
    def load(cls, dirpath:str) -> "SASRec":
        config = SASRecConfig.load(os.path.join(dirpath, "model_cfg.json"))
        dataset_stats = DatasetStats.load(os.path.join(dirpath, "dataset_stats.json"))
        state_dict = torch.load(os.path.join(dirpath, "model_state_dict.pt"))

        model = cls(config, dataset_stats)
        model.load_state_dict(state_dict)

        return model





class TrainConfig(BaseConfig):
    lr: float
    batch_size: int
    epochs: int
    l2_emb: float
    device: str
    negative_samples: int = 1  # 0 if no sampling, number of negatives othervise
    loss: str = "bce"
    

class Trainer:
    def __init__(self, 
        model: SASRec,
        config: TrainConfig, 
        model_dir:str,
    ):
        self.model = model
        self.config = config
        self.model_dir = model_dir
        self.writer = None
        self.tracker = None
        
        self.optimizer = None
        self.loss_func = None

        self.inited = False

    def init(self):
        self._init_optimizers()
        self._init_logger()
        self._init_loss_func()

    def fit(
        self,
        train_dataloader:DataLoader,
    ) -> SASRec:
        if not self.inited:
            self.init()

        self.model.to(self.config.device)
        xavier_normal_init(self.model)
        self.model.train()  # enable model training

        epoch_start_idx = 1

        # ce_criterion = torch.nn.CrossEntropyLoss()
        # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...

        iteration = 0
        try:
            for epoch in range(epoch_start_idx, self.config.epochs + 1):
                logger.info("training epoch %s", epoch)

                self.writer.add_scalar("epoch", epoch, iteration)

                for x, y, w in train_dataloader:
                    x = to_device(x, self.config.device)
                    y = to_device(y, self.config.device)
                    w = to_device(w, self.config.device)

                    self.train_step(x, y, w, iteration)

                    iteration += 1
            
        except KeyboardInterrupt:
            logger.info("training interrupted")

    def train_step(self, x, y, w, iteration):
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.loss_func(
            logits.transpose(1, 2), y
        )  # loss expects logits in form of [N, C, D1]
        loss = loss * w
        n = (loss > 0).to(loss.dtype)
        loss = torch.sum(loss) / torch.sum(n)
        # reg_loss = torch.zeros_like(loss)
        # for param in model.item_emb.parameters():
        #     reg_loss += self.config.l2_emb * torch.norm(param)

        joint_loss = loss # + reg_loss
        joint_loss.backward()
        self.optimizer.step()

        self.writer.add_scalar(
            "train_loss", loss.item(), iteration
        )  # expected 0.4~0.6 after init few epochs
        # self.writer.add_scalar("train_reg_loss", reg_loss.item(), iteration)
        self.writer.add_scalar("train_joint_loss", joint_loss.item(), iteration)

    def save_model(self):
        self.model.save(self.model_dir)

    def _init_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.98)
        )

    def _init_logger(self):
        logpath = os.path.join(self.model_dir, "artifacts")
        os.makedirs(logpath, exist_ok=True)
        self.writer = SummaryWriter(logpath)

    def _init_loss_func(self):
        if self.config.loss == "bce":
            self.loss_func = nn.BCEWithLogitsLoss()
        elif self.config.loss == "sm_ce":
            self.loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        else:
            raise Exception(f"loss {self.config.loss} is not supported")
        
        logger.info("used %s loss", self.config.loss)





class SingletonMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__call__(*args, **kwargs)

        return cls.instance


class ModelRegistryException(Exception):
    pass


def register_model(cls):
    MODEL_REGISTRY.register_model(cls)

    return cls

class ModelRegistry(metaclass=SingletonMeta):
    def __init__(self):
        self._name2cls = {}
        self._cls2name = {}
    
    def get_cls_by_name(self, name:str):
        if name not in self._name2cls:
            raise ModelRegistryException(f"model with name {name} not found")

        return self._name2cls[name]

    def get_name_by_cls(self, cls):
        if cls not in self._cls2name:
            raise ModelRegistryException(f"model with class {cls} not found")

        return self._cls2name[cls]
    
    def register_model(self, cls):
        name = cls.__name__

        if name in self._name2cls:
            # raise ModelRegistryException(f"model {cls} already registered")
            logger.warning("model %s already registered", cls)
        
        self._name2cls[name] = cls
        self._cls2name[cls] = name

MODEL_REGISTRY = ModelRegistry()



@register_model
class SASRecRecommeder:
    def __init__(
        self, 
        processor: SASRecProcessor, 
        model:SASRec, 
        task_converter:SequenceTaskConverter,
        device:Union[torch.device, str]="cpu",
    ):
        self.processor = processor
        self.model = model.eval()
        # TODO merge with processor
        self.task_converter = task_converter
        self.device = device

        self.model.to(self.device)

    # TODO what if candidate_items has elements not in item_features
    def recommend(
        self,
        user_item_interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        top_k: int,
        candidate_items: Sequence[str],
        batch_size:int=128,
    ) -> pd.DataFrame:
        """ Accepts 3 dataframes from DB and returns dataset with recommends
        """
        # TODO check that users with unsupported items in history have appropriate embeddings

        # filter out unsupported items
        supported_items = self.get_supported_items(item_features)
        candidate_items = self._filter_candidate_items(candidate_items, supported_items)
        
        # TODO here we can filter users completely. 
        # We need to find out where to check if we can make recs.
        # If it is here, than just return error object with info about such users.
        user_item_interactions, item_features =  self._filter_datasets(
            supported_items=supported_items,
            user_item_interactions=user_item_interactions,
            item_features=item_features,
        )

        # filter out unnecessary items to speed up calculation
        items = user_item_interactions['item_id'].drop_duplicates().to_list() + list(candidate_items)
        item_features = item_features[item_features['item_id'].isin(items)]

        x = self.task_converter.inference_transform(
            user_item_interactions=user_item_interactions,
            user_features=user_features,
            item_features=item_features,
        )

        x = self.processor.transform(x)

        model_x = SequenceDataset(x=x, batch_size=batch_size)
        dataloader = DataLoader(model_x, batch_size=None) # batching in dataset


        # TODO candidate_items fix private method usage
        item2index = self.processor._build_items_mapping(x.items)
        candidate_items_inds = [item2index[e] for e in candidate_items]
        candidate_items_inds = torch.LongTensor(candidate_items_inds).to(self.device)

        logits = []
        device = self.model.get_model_device()
        with torch.no_grad():
            for x_batch in tqdm.tqdm(dataloader):
                x_batch = to_device(x_batch, device)
                logits_batch = self.model.predict(x_batch, candidate_items_inds).detach().cpu().numpy()
                logits.append(logits_batch)

        logits = np.concatenate(logits, axis=0) 

        recs = collect_recs(
            user_item_interactions=user_item_interactions,
            candidate_items=candidate_items,
            users=x.users,
            logits=logits,
            top_k=top_k,
        )
        return recs

    # TODO move implementation to processor
    def get_supported_items(self, item_features:pd.DataFrame) -> List[str]:
        supported_ids = set(self.processor.item_id_encoder.tokens())
        mask = item_features['item_id'].isin(supported_ids)
        
        if self.processor.item_tags_encoder is not None:
            supported_tags = set(self.processor.item_tags_encoder.tokens())
            mask = mask | item_features['tags_set'].apply(lambda x: any([e in supported_tags for e in x]))
        
        item_features = item_features[mask]

        return item_features['item_id'].to_list()
    
    def _filter_candidate_items(self, candidate_items:Sequence[str], supported_items:Sequence[str]) -> List[str]:
        supported_candidate_items = list(set(candidate_items).intersection(supported_items))
        
        filtered_items_cnt = len(candidate_items) - len(supported_candidate_items)
        if filtered_items_cnt > 0:
            logger.warning("filtered out %s candidate items which are not supported by model", filtered_items_cnt)

        return supported_candidate_items
    
    def _filter_datasets(self, 
        supported_items:Sequence[str],
        user_item_interactions:pd.DataFrame,
        item_features:pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        item_features = item_features[item_features['item_id'].isin(supported_items)]
        user_item_interactions = user_item_interactions[user_item_interactions['item_id'].isin(supported_items)]

        return user_item_interactions, item_features

    @classmethod
    def load(cls, dirpath:str, **kwargs) -> "SASRecRecommeder":
        model = SASRec.load(dirpath)
        processor = SASRecProcessor.load(dirpath)
        task_converter = SequenceTaskConverter.load(dirpath)

        recommender = cls(
            model=model,
            processor=processor,
            task_converter=task_converter,
            **kwargs,
        )    

        return recommender
    
    def save(self, dirpath:str):
        self.model.save(dirpath)
        self.processor.save(dirpath)
        self.task_converter.save(dirpath)


def collect_recs(
    user_item_interactions:pd.DataFrame,
    candidate_items:List[str],
    users:List[str],
    logits: np.ndarray, # [users, candidate_items]
    top_k:int,
):
    user2hist = user_item_interactions.groupby("user_id")["item_id"].agg(set).to_dict()
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


        records.append((
            user,
            cur_recs,
            cur_scores,
        ))
    
    recs = pd.DataFrame(
        records, 
        columns=[
            'user_id',
            'item_id',
            'score',
        ]
    ).explode(["item_id", "score"])
    recs['rank'] = recs.groupby("user_id").cumcount() + 1

    return recs







# Prepare datasets

def unknown_items_filter(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
):
    """Removes interactions with unknown items"""

    interactions_items = set(user_item_interactions["item_id"].drop_duplicates())
    unknown_items = interactions_items.difference(item_features["item_id"])

    if len(unknown_items) > 0:
        logger.warning(
            "%s items have not passed unknown_items_filter",
            len(unknown_items),
        )
        user_item_interactions = user_item_interactions[
            ~user_item_interactions["item_id"].isin(unknown_items)
        ]

    return user_item_interactions, item_features, user_features

def too_high_multiwatch_rate_item_filter(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    multiwatch_th: float,
    max_multiwatch_ratio: float,
):
    """Removes interactions with items which have to high multiwatch rate"""

    data_ext = pd.merge(
        user_item_interactions[["user_id", "item_id", "score"]],
        item_features[["item_id", "duration_full"]],
        on=["item_id"],
        how="left",
    )

    assert not data_ext["duration_full"].isna().any()

    data_ext["multiwatch"] = (
        data_ext["score"] / data_ext["duration_full"] > multiwatch_th
    )

    item2multiwatch_rate = data_ext.groupby("item_id")["multiwatch"].mean()
    bad_items = set(
        item2multiwatch_rate[item2multiwatch_rate > max_multiwatch_ratio].index
    )

    if len(bad_items) > 0:
        logger.warning(
            "%s items have not passed too_high_multiwatch_rate_item_filter",
            len(bad_items),
        )
        user_item_interactions = user_item_interactions[
            ~user_item_interactions["item_id"].isin(bad_items)
        ]

    return user_item_interactions, item_features, user_features

def max_score_per_active_day_filter(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    value: float,
):
    """Removes users with too high WT per active day"""

    user_total_score = (
        user_item_interactions.groupby("user_id")["score"].sum().reset_index()
    )
    df = pd.merge(
        user_total_score[["user_id", "score"]],
        user_features[["user_id", "watch_days"]],
        on="user_id",
        how="left",
    ).set_index("user_id")

    assert (
        not df["watch_days"].isna().any()
    ), "assumed that all users in interactions have user features"

    df["score_per_active_day"] = df["score"] / df["watch_days"]
    valid_users = set(df[df["score_per_active_day"] < value].index)

    not_valid_users_cnt = df.shape[0] - len(valid_users)
    if not_valid_users_cnt > 0:
        logger.warning(
            "%s (%s) users have not passed max_score_per_active_day_filter filter",
            not_valid_users_cnt,
            round(not_valid_users_cnt / df.shape[0], 4),
        )
        user_item_interactions = user_item_interactions[
            user_item_interactions["user_id"].isin(valid_users)
        ]

    return user_item_interactions, item_features, user_features

def max_multiwatch_ratio_filter(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    multiwatch_th: float,
    max_multiwatch_ratio: float,
):
    """Removes users with too high ratio of multiwatch items in his history"""

    data_ext = pd.merge(
        user_item_interactions[["user_id", "item_id", "score"]],
        item_features[["item_id", "duration_full"]],
        on=["item_id"],
        how="left",
    )

    assert not data_ext["duration_full"].isna().any()

    data_ext["multiwatch"] = (
        data_ext["score"] / data_ext["duration_full"] > multiwatch_th
    )
    multiwatch_ratio = data_ext.groupby("user_id")["multiwatch"].mean()
    bad_users = set(multiwatch_ratio[multiwatch_ratio > max_multiwatch_ratio].index)
    if len(bad_users) > 0:
        logger.warning(
            "%s (%s) users have not passed max_multiwatch_ratio_filter",
            len(bad_users),
            round(len(bad_users) / multiwatch_ratio.shape[0], 4),
        )
        user_item_interactions = user_item_interactions[
            ~user_item_interactions["user_id"].isin(bad_users)
        ]

    return user_item_interactions, item_features, user_features


def filter_datasets(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_features: pd.DataFrame,
    train:bool,
    multiwatch_th:float=1.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # missing duration
    mask = item_features["duration_full"].isna()
    if any(mask):
        logging.warning(
            "found %s records without `duration_full`; dropping rows ...", sum(mask)
        )
        item_features = item_features[~mask]

    # drop items for which we do not have info for some reason
    user_item_interactions, item_features, user_features = unknown_items_filter(
        user_item_interactions=user_item_interactions,
        item_features=item_features,
        user_features=user_features,
    )

    # drop items which have high multiwatch ration
    # TODO this problem arises mainly due to episode transitions between glos.
    # golden record and prefiltration have to solve this problem without this hack.
    user_item_interactions, item_features, user_features = too_high_multiwatch_rate_item_filter(
        user_item_interactions=user_item_interactions,
        item_features=item_features,
        user_features=user_features,
        multiwatch_th=multiwatch_th,
        max_multiwatch_ratio=0.5, # found by looking at items's rations
    )

    # strange users
    users = user_item_interactions["user_id"].drop_duplicates()
    users = users[
        ~users.apply(lambda x: x.startswith("prod.") or x.startswith("xtv"))
    ].to_list()
    if len(users) > 0:
        logger.warning(
            "found %s strange users (for example: %s); dropping ...",
            len(users),
            users[:5],
        )
        user_item_interactions = user_item_interactions[
            ~user_item_interactions["user_id"].isin(users)
        ]

    # too much tvt per day
    if train:
        user_item_interactions, item_features, user_features = (
            max_score_per_active_day_filter(
                user_item_interactions=user_item_interactions,
                item_features=item_features,
                user_features=user_features,
                value=60 * 60 * 8,  # 8 hours per active day
            )
        )

    if train:
        user_item_interactions, item_features, user_features = max_multiwatch_ratio_filter(
            user_item_interactions=user_item_interactions,
            item_features=item_features,
            user_features=user_features,
            multiwatch_th=multiwatch_th,  # TODO find it statistically
            max_multiwatch_ratio=0.8,  # TODO experiment with it
        )

    # check item info for each interactions' item
    items = item_features["item_id"]
    missing_items = len(
        set(user_item_interactions["item_id"].drop_duplicates()).difference(items)
    )
    if missing_items > 0:
        logger.warning(
            "item features not found for %s items in interactions; they will be dropped",
            missing_items,
        )
        user_item_interactions = user_item_interactions[
            user_item_interactions["item_id"].isin(items)
        ]

    return user_item_interactions, item_features, user_features


def train_sasrec_script(
    train_ds_path: str,
    item_features_path: str,
    model_dir: str,
    processor_config: SASRecProcessorConfig,
    model_config: SASRecConfig,
    train_config: TrainConfig,
    train_preprocessing_config: TrainPreprocessingConfig,
    task_converter_config: Optional[SequenceTaskConverterConfig]=None,
):
    if task_converter_config is None:
        task_converter_config = SequenceTaskConverterConfig() # TODO do we need it?

    logger.info("loading datasets")
    train_x, train_y, items = load_pickle(train_ds_path)
    item_features = pd.read_parquet(item_features_path)

    logger.info("running training script")
    run_train_script(
        user_item_interactions=train_x,
        item_features=item_features,
        user_features=None,
        processor_config=processor_config,
        model_config=model_config,
        train_config=train_config,
        train_preprocessing_config=train_preprocessing_config,
        task_converter_config=task_converter_config,
        model_dir=model_dir,
    )


def run_train_script(
    user_item_interactions: pd.DataFrame,
    item_features: pd.DataFrame, # TODO use simple id features if absent
    user_features: pd.DataFrame, # TODO use simple id features if absent
    processor_config:SASRecProcessorConfig,
    model_config: SASRecConfig,
    train_config: TrainConfig,
    train_preprocessing_config:TrainPreprocessingConfig,
    task_converter_config:SequenceTaskConverterConfig,
    model_dir:str,
): 
    # logger.info("preprocessing dataset")
    # user_item_interactions, item_features, user_features = preprocess_train_datasets(
    #     config=train_preprocessing_config, 
    #     user_item_interactions=user_item_interactions,
    #     item_features=item_features,
    #     user_features=user_features,
    # )

    logger.info("testing dataset")
    test_processed_train_datasets(user_item_interactions, item_features, user_features)
    
    logger.info("converting datasets to task format")
    task_converter = SequenceTaskConverter(task_converter_config)
    train_x, train_y = task_converter.train_transform(
        user_item_interactions=user_item_interactions,
        user_features=user_features,
        item_features=item_features,
    )

    logger.info("building preprocessor")
    processor = SASRecProcessor.build(processor_config)
    processor.fit(train_x)
    train_x = processor.transform(train_x)
    train_y = processor.transform_target(train_y, train_x)

    logger.info("building train dataset")
    train_dataset = SequenceDataset(x=train_x, y=train_y, batch_size=train_config.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=None) # batching in dataset

    # init model
    logger.info("building model")
    dataset_stats = processor.dataset_stats()
    model = SASRec(model_config, dataset_stats)

    logger.info("building trainer")
    trainer = Trainer(
        model=model,
        config=train_config,
        model_dir=model_dir,
    )
    trainer.fit(train_dataloader)
    trainer.save_model()
    processor.save(model_dir)
    task_converter.save(model_dir)




