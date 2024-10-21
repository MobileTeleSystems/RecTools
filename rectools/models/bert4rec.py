import typing as tp
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from pytorch_lightning import Trainer
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset, Interactions
from rectools.dataset.features import SparseFeatures
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ErrorBehaviour, InternalRecoTriplet, ModelBase
from rectools.models.rank import Distance, ImplicitRanker
from rectools.models.sasrec import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    ItemNetConstructor,
    LearnableInversePositionalEncoding,
    PositionalEncodingBase,
    SequenceDataset,
    SessionEncoderDataPreparatorBase,
    SessionEncoderLightningModuleBase,
    TransformerLayersBase,
)
from rectools.types import InternalIdsArray

PADDING_VALUE = "PAD"
MASKING_VALUE = "MASK"


class BERT4RecDataPreparator(SessionEncoderDataPreparatorBase):
    """TODO"""

    def __init__(
        self,
        session_max_len: int,
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        mask_prob: float,
        item_extra_tokens: tp.Sequence[tp.Hashable] = (PADDING_VALUE, MASKING_VALUE),
        shuffle_train: bool = True,
    ) -> None:
        super().__init__()
        self.session_max_len = session_max_len
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions
        self.item_extra_tokens = item_extra_tokens
        self.mask_prob = mask_prob
        self.shuffle_train = shuffle_train
        # TODO: add SequenceDatasetType for fit and recommend

    def process_dataset_train(self, dataset: Dataset) -> Dataset:
        """TODO"""
        interactions = dataset.get_raw_interactions()

        # Filter interactions
        user_stats = interactions[Columns.User].value_counts()
        users = user_stats[user_stats >= self.train_min_user_interactions].index
        interactions = interactions[(interactions[Columns.User].isin(users))]
        interactions = interactions.sort_values(Columns.Datetime).groupby(Columns.User).tail(self.session_max_len)

        # Construct dataset
        # TODO: user features are dropped for now
        user_id_map = IdMap.from_values(interactions[Columns.User].values)
        item_id_map = IdMap.from_values(self.item_extra_tokens)
        item_id_map = item_id_map.add_ids(interactions[Columns.Item])

        # get item features
        item_features = None
        if dataset.item_features is not None:
            item_features = dataset.item_features
            # TODO: remove assumption on SparseFeatures and add Dense Features support
            if not isinstance(item_features, SparseFeatures):
                raise ValueError("`item_features` in `dataset` must be `SparseFeatures` instance.")

            internal_ids = dataset.item_id_map.convert_to_internal(
                item_id_map.get_external_sorted_by_internal()[self.n_item_extra_tokens :]
            )
            sorted_item_features = item_features.take(internal_ids)

            dtype = sorted_item_features.values.dtype
            n_features = sorted_item_features.values.shape[1]
            extra_token_feature_values = sparse.csr_matrix((self.n_item_extra_tokens, n_features), dtype=dtype)

            full_feature_values: sparse.scr_matrix = sparse.vstack(
                [extra_token_feature_values, sorted_item_features.values], format="csr"
            )

            item_features = SparseFeatures.from_iterables(values=full_feature_values, names=item_features.names)

        interactions = Interactions.from_raw(interactions, user_id_map, item_id_map)

        dataset = Dataset(user_id_map, item_id_map, interactions, item_features=item_features)

        self.item_id_map = dataset.item_id_map
        return dataset

    def _mask_session(self, ses: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_session = ses.copy()
        target = ses.copy()
        random_probs = np.random.rand(len(ses))
        for j in range(len(ses)):
            if random_probs[j] < self.mask_prob:
                random_probs[j] /= self.mask_prob
                if random_probs[j] < 0.8:
                    masked_session[j] = 1
                elif random_probs[j] < 0.9:
                    masked_session[j] = np.random.randint(low=2, high=self.item_id_map.size, size=1)[0]
            else:
                target[j] = 0
        return masked_session, target

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
        """TODO"""
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        for i, (ses, ses_weights) in enumerate(batch):
            masked_session, target = self._mask_session(ses)
            x[i, -len(ses) :] = masked_session  # ses: [session_len] -> x[i]: [session_max_len]
            y[i, -len(ses) :] = target  # ses: [session_len] -> y[i]: [session_max_len]
            yw[i, -len(ses) :] = ses_weights  # ses_weights: [session_len] -> yw[i]: [session_max_len]

        return torch.LongTensor(x), torch.LongTensor(y), torch.FloatTensor(yw)

    def get_dataloader_train(self, processed_dataset: Dataset) -> DataLoader:
        """TODO"""
        sequence_dataset = SequenceDataset.from_interactions(processed_dataset.interactions.df)
        train_dataloader = DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_train,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=self.shuffle_train,
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
            explanation = f"""{n_filtered} target users were considered cold because of missing known items"""
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
        interactions = dataset.get_raw_interactions()
        interactions = interactions[interactions[Columns.Item].isin(self.get_known_item_ids())]
        filtered_interactions = Interactions.from_raw(interactions, dataset.user_id_map, self.item_id_map)
        filtered_dataset = Dataset(dataset.user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> torch.LongTensor:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len))
        for i, (ses, _) in enumerate(batch):
            session = ses.copy()
            session = session + [1]
            x[i, -len(ses) :] = ses[-self.session_max_len :]
        return torch.LongTensor(x)

    def get_dataloader_recommend(self, dataset: Dataset) -> DataLoader:
        """TODO"""
        sequence_dataset = SequenceDataset.from_interactions(dataset.interactions.df)
        recommend_dataloader = DataLoader(
            sequence_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn_recommend,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )
        return recommend_dataloader


class PointWiseFeedForward(nn.Module):
    """TODO"""

    def __init__(self, n_factors: int, n_factors_ff: int, dropout_rate: float) -> None:
        """TODO"""
        super().__init__()
        self.ff_linear1 = nn.Linear(n_factors, n_factors_ff)
        self.ff_gelu = torch.nn.GELU()
        self.ff_dropout = torch.nn.Dropout(dropout_rate)
        self.ff_linear2 = nn.Linear(n_factors_ff, n_factors)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """TODO"""
        output = self.ff_gelu(self.ff_linear1(seqs))
        fin = self.ff_linear2(self.ff_dropout(output))
        return fin


class BERT4RecTransformerLayers(TransformerLayersBase):
    """TODO"""

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
            [nn.MultiheadAttention(n_factors, n_heads, dropout_rate, batch_first=True) for _ in range(n_blocks)]
        )
        self.layer_norm1 = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.dropout1 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        self.layer_norm2 = nn.ModuleList([nn.LayerNorm(n_factors) for _ in range(n_blocks)])
        self.feed_forward = nn.ModuleList(
            [PointWiseFeedForward(n_factors, n_factors * 4, dropout_rate) for _ in range(n_blocks)]
        )
        self.dropout2 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        # self.dropout3 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])

    def forward(self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """TODO"""
        for i in range(self.n_blocks):
            mha_input = self.layer_norm1[i](seqs)
            # mha_output, _ =
            # self.multi_head_attn[i](mha_input, mha_input, mha_input, attn_mask=attn_mask, need_weights=False)
            mha_output, _ = self.multi_head_attn[i](mha_input, mha_input, mha_input, need_weights=False)
            seqs = seqs + self.dropout1[i](mha_output)
            ff_input = self.layer_norm2[i](seqs)
            ff_output = self.feed_forward[i](ff_input)
            seqs = seqs + self.dropout2[i](ff_output)
            seqs = seqs * timeline_mask
            # seqs = self.dropout3[i](seqs)

        return seqs


# ####  --------------  Session Encoder  --------------  #### #


class TransformerBasedSessionEncoder(torch.nn.Module):
    """TODO"""

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        session_max_len: int,
        dropout_rate: float,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        transformer_layers_type: tp.Type[TransformerLayersBase] = BERT4RecTransformerLayers,
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
        self.n_factors = n_factors
        self.dropout_rate = dropout_rate
        self.n_heads = n_heads

        self.item_net_block_types = item_net_block_types

    def construct_item_net(self, dataset: Dataset) -> None:
        """TODO"""
        self.item_model = ItemNetConstructor.from_dataset(
            dataset, self.n_factors, self.dropout_rate, self.item_net_block_types
        )

    def encode_sessions(self, sessions: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pass user history through item embeddings and transformer blocks.

        Returns
        -------
            torch.Tensor. [batch_size, session_max_len, n_factors]

        """
        session_max_len = sessions.shape[1]
        attn_mask = None
        if self.use_causal_attn:
            attn_mask = ~torch.tril(
                torch.ones((session_max_len, session_max_len), dtype=torch.bool, device=sessions.device)
            )
        timeline_mask = sessions != 0
        attn_mask = ~timeline_mask.unsqueeze(1).repeat(self.n_heads, timeline_mask.squeeze(-1).shape[1], 1)
        timeline_mask = timeline_mask.unsqueeze(-1)
        seqs = item_embs[sessions]  # [batch_size, session_max_len, n_factors]
        seqs = self.pos_encoding(seqs, timeline_mask)
        seqs = self.emb_dropout(seqs)
        seqs = self.transformer_layers(seqs, timeline_mask, attn_mask)
        return seqs

    def forward(
        self,
        sessions: torch.Tensor,  # [batch_size, session_max_len]
    ) -> torch.Tensor:
        """TODO"""
        item_embs = self.item_model.get_all_embeddings()  # [n_items + 2, n_factors]
        session_embs = self.encode_sessions(sessions, item_embs)  # [batch_size, session_max_len, n_factors]
        logits = session_embs @ item_embs.T  # [batch_size, session_max_len, n_items + 2]
        return logits


class SessionEncoderLightningModule(SessionEncoderLightningModuleBase):
    """TODO"""

    def on_train_start(self) -> None:
        """TODO"""
        self._truncated_normal_init()

    def configure_optimizers(self) -> torch.optim.Adam:
        """TODO"""
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """TODO"""
        x, y, w = batch
        logits = self.forward(x)  # [batch_size, session_max_len, n_items + 2]
        if self.loss == "softmax":
            # We are using CrossEntropyLoss with a multi-dimensional case

            # Logits must be passed in form of [batch_size, n_items + 2, session_max_len],
            #  where n_items + 2 is number of classes

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
        raise ValueError(f"loss {loss} is not supported")

    def _truncated_normal_init(self) -> None:
        """TODO"""
        for _, param in self.torch_model.named_parameters():
            try:
                torch.nn.init.trunc_normal_(param.data)
            except ValueError:
                pass


class BERT4RecModel(ModelBase):
    """TODO"""

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        dropout_rate: float = 0.2,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        cpu_n_threads: int = 0,
        session_max_len: int = 32,
        batch_size: int = 128,
        loss: str = "softmax",
        lr: float = 0.01,
        dataloader_num_workers: int = 0,
        train_min_user_interaction: int = 2,
        mask_prob: float = 0.15,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = BERT4RecTransformerLayers,
        data_preparator_type: tp.Type[SessionEncoderDataPreparatorBase] = BERT4RecDataPreparator,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
        device: str = "cpu",  # TODO: remove
    ):
        super().__init__(verbose=verbose)
        self.data_preparator = data_preparator_type(
            session_max_len=session_max_len,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interaction,
            mask_prob=mask_prob,
        )
        self.torch_model: TransformerBasedSessionEncoder
        self._torch_model = TransformerBasedSessionEncoder(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            session_max_len=session_max_len,
            dropout_rate=dropout_rate,
            use_pos_emb=use_pos_emb,
            use_causal_attn=False,
            transformer_layers_type=transformer_layers_type,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
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
        self.lr = lr
        self.loss = loss
        self.n_threads = cpu_n_threads
        self.u2i_dist = Distance.DOT
        self.i2i_dist = Distance.COSINE
        self.device = torch.device(device)  # TODO: remove

    def _fit(
        self,
        dataset: Dataset,
    ) -> None:
        processed_dataset = self.data_preparator.process_dataset_train(dataset)
        train_dataloader = self.data_preparator.get_dataloader_train(processed_dataset)
        self.torch_model = deepcopy(self._torch_model)  # TODO: check that it works
        self.torch_model.construct_item_net(processed_dataset)

        lightning_model = self.lightning_module_type(self.torch_model, self.lr, self.loss)
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
        dataset: Dataset,  # [n_rec_users x n_items + 2]
        k: int,
        filter_viewed: bool,
        sorted_item_ids_to_recommend: tp.Optional[InternalIdsArray],  # model_internal
    ) -> InternalRecoTriplet:
        if sorted_item_ids_to_recommend is None:  # TODO: move to _get_sorted_item_ids_to_recommend
            sorted_item_ids_to_recommend = self.data_preparator.get_known_items_sorted_internal_ids()  # model internal

        self.torch_model = self.torch_model.eval()
        self.torch_model.to(self.device)

        # Dataset has already been filtered and adapted to known item_id_map
        recommend_dataloader = self.data_preparator.get_dataloader_recommend(dataset)

        session_embs = []
        item_embs = self.torch_model.item_model.get_all_embeddings()  # [n_items + 2, n_factors]
        with torch.no_grad():
            for x_batch in tqdm.tqdm(recommend_dataloader):  # TODO: from tqdm.auto import tqdm. Also check `verbose``
                x_batch = x_batch.to(self.device)  # [batch_size, session_max_len]
                encoded = self.torch_model.encode_sessions(x_batch, item_embs)[:, -1, :]  # [batch_size, n_factors]
                encoded = encoded.detach().cpu().numpy()
                session_embs.append(encoded)

        user_embs = np.concatenate(session_embs, axis=0)
        user_embs = user_embs[user_ids]
        item_embs_np = item_embs.detach().cpu().numpy()

        ranker = ImplicitRanker(
            self.u2i_dist,
            user_embs,  # [n_rec_users, n_factors]
            item_embs_np,  # [n_items + 2, n_factors]
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
            filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + 2]
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

        self.torch_model = self.torch_model.eval()
        item_embs = self.torch_model.item_model.get_all_embeddings().detach().cpu().numpy()  # [n_items + 2, n_factors]

        # TODO: i2i reco do not need filtering viewed. And user most of the times has GPU
        # Should we use torch dot and topk? Should be faster

        ranker = ImplicitRanker(
            self.i2i_dist,
            item_embs,  # [n_items + 2, n_factors]
            item_embs,  # [n_items + 2, n_factors]
        )
        return ranker.rank(
            subject_ids=target_ids,  # model internal
            k=k,
            filter_pairs_csr=None,
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model internal
            num_threads=0,
        )
