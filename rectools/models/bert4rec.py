import typing as tp
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import typing_extensions as tpe
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from torch import nn

from rectools.models.sasrec import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    LearnableInversePositionalEncoding,
    PointWiseFeedForward,
    PositionalEncodingBase,
    SessionEncoderDataPreparatorBase,
    SessionEncoderLightningModule,
    SessionEncoderLightningModuleBase,
    TransformerLayersBase,
    TransformerModelBase,
    TransformerModelConfig,
)

PADDING_VALUE = "PAD"
MASKING_VALUE = "MASK"


class BERT4RecDataPreparator(SessionEncoderDataPreparatorBase):
    """TODO"""

    def __init__(
        self,
        session_max_len: int,
        n_negatives: tp.Optional[int],
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        mask_prob: float,
        item_extra_tokens: tp.Sequence[tp.Hashable],
        shuffle_train: bool = True,
        get_val_mask_func: tp.Optional[tp.Callable] = None,
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            item_extra_tokens=item_extra_tokens,
            shuffle_train=shuffle_train,
            get_val_mask_func=get_val_mask_func,
        )
        self.mask_prob = mask_prob

    def _mask_session(self, ses: List[int]) -> Tuple[List[int], List[int]]:
        masked_session = ses.copy()
        target = ses.copy()
        random_probs = np.random.rand(len(ses))
        for j in range(len(ses)):
            if random_probs[j] < self.mask_prob:
                random_probs[j] /= self.mask_prob
                if random_probs[j] < 0.8:
                    masked_session[j] = self.extra_token_ids[MASKING_VALUE]
                elif random_probs[j] < 0.9:
                    masked_session[j] = np.random.randint(low=self.n_item_extra_tokens, high=self.item_id_map.size)
            else:
                target[j] = 0
        return masked_session, target

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        """TODO"""
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len + 1))
        y = np.zeros((batch_size, self.session_max_len + 1))
        yw = np.zeros((batch_size, self.session_max_len + 1))
        for i, (ses, ses_weights) in enumerate(batch):
            masked_session, target = self._mask_session(ses)
            x[i, -len(ses) :] = masked_session  # ses: [session_len] -> x[i]: [session_max_len + 1]
            y[i, -len(ses) :] = target  # ses: [session_len] -> y[i]: [session_max_len + 1]
            yw[i, -len(ses) :] = ses_weights  # ses_weights: [session_len] -> yw[i]: [session_max_len + 1]

        batch_dict = {"x": torch.LongTensor(x), "y": torch.LongTensor(y), "yw": torch.FloatTensor(yw)}
        # TODO: we are sampling negatives for paddings
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, self.session_max_len, self.n_negatives),
            )  # [batch_size, session_max_len, n_negatives]
            batch_dict["negatives"] = negatives
        return batch_dict

    def _collate_fn_recommend(self, batch: List[Tuple[List[int], List[float]]]) -> Dict[str, torch.Tensor]:
        """Right truncation, left padding to session_max_len"""
        x = np.zeros((len(batch), self.session_max_len + 1))
        for i, (ses, _) in enumerate(batch):
            session = ses.copy()
            session = session + [self.extra_token_ids[MASKING_VALUE]]
            x[i, -len(ses) - 1 :] = session[-self.session_max_len - 1 :]
        return {"x": torch.LongTensor(x)}


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
            [PointWiseFeedForward(n_factors, n_factors * 4, dropout_rate, torch.nn.GELU()) for _ in range(n_blocks)]
        )
        self.dropout2 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])
        self.dropout3 = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(n_blocks)])

    def forward(
        self, seqs: torch.Tensor, timeline_mask: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """TODO"""
        for i in range(self.n_blocks):
            mha_input = self.layer_norm1[i](seqs)
            mha_output, _ = self.multi_head_attn[i](
                mha_input,
                mha_input,
                mha_input,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            seqs = seqs + self.dropout1[i](mha_output)
            ff_input = self.layer_norm2[i](seqs)
            ff_output = self.feed_forward[i](ff_input)
            seqs = seqs + self.dropout2[i](ff_output)
            seqs = self.dropout3[i](seqs)
        # TODO: test with torch.nn.Linear and cross-entropy loss as in
        # https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/f66f2534ebfd937778c7174b5f9f216efdebe5de/models/bert.py#L11C1-L11C67
        return seqs


class BERT4RecModelConfig(TransformerModelConfig):
    """TODO."""

    mask_prob: float = 0.15


class BERT4RecModel(TransformerModelBase[BERT4RecModelConfig]):
    """TODO"""

    config_class = BERT4RecModelConfig

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        use_causal_attn: bool = True,
        use_key_padding_mask: bool = False,
        dropout_rate: float = 0.2,
        session_max_len: int = 32,
        dataloader_num_workers: int = 0,
        batch_size: int = 128,
        loss: str = "softmax",
        n_negatives: int = 1,
        gbce_t: float = 0.2,
        lr: float = 0.01,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        recommend_device: Union[str, Accelerator] = "auto",
        recommend_n_threads: int = 0,
        recommend_use_gpu_ranking: bool = True,
        train_min_user_interactions: int = 2,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = BERT4RecTransformerLayers,  # SASRec authors net
        data_preparator_type: tp.Type[BERT4RecDataPreparator] = BERT4RecDataPreparator,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
        top_k_saved_val_reco: tp.Optional[int] = None,
        get_val_mask_func: tp.Optional[tp.Callable] = None,
        mask_prob: float = 0.15,
    ):
        self.mask_prob = mask_prob
        super().__init__(
            transformer_layers_type=transformer_layers_type,
            data_preparator_type=data_preparator_type,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_factors=n_factors,
            use_pos_emb=use_pos_emb,
            use_causal_attn=use_causal_attn,
            use_key_padding_mask=use_key_padding_mask,
            dropout_rate=dropout_rate,
            session_max_len=session_max_len,
            loss=loss,
            gbce_t=gbce_t,
            lr=lr,
            epochs=epochs,
            verbose=verbose,
            deterministic=deterministic,
            recommend_device=recommend_device,
            recommend_n_threads=recommend_n_threads,
            recommend_use_gpu_ranking=recommend_use_gpu_ranking,
            trainer=trainer,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
            top_k_saved_val_reco=top_k_saved_val_reco,
            n_negatives=n_negatives,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            get_val_mask_func=get_val_mask_func,
        )

    def _init_data_preparator(self) -> None:
        self.data_preparator = self.data_preparator_type(
            session_max_len=self.session_max_len - 1,
            n_negatives=self.n_negatives if self.loss != "softmax" else None,
            batch_size=self.batch_size,
            dataloader_num_workers=self.dataloader_num_workers,
            train_min_user_interactions=self.train_min_user_interactions,
            get_val_mask_func=self.get_val_mask_func,
            item_extra_tokens=(PADDING_VALUE, MASKING_VALUE),
            mask_prob=self.mask_prob,
        )

    def _get_config(self) -> BERT4RecModelConfig:
        return BERT4RecModelConfig(
            cls=self.__class__,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            n_factors=self.n_factors,
            use_pos_emb=self.use_pos_emb,
            use_causal_attn=self.use_causal_attn,
            use_key_padding_mask=self.use_key_padding_mask,
            dropout_rate=self.dropout_rate,
            session_max_len=self.session_max_len,
            dataloader_num_workers=self.dataloader_num_workers,
            batch_size=self.batch_size,
            loss=self.loss,
            n_negatives=self.n_negatives,
            gbce_t=self.gbce_t,
            lr=self.lr,
            epochs=self.epochs,
            verbose=self.verbose,
            deterministic=self.deterministic,
            recommend_device=self.recommend_device,
            recommend_n_threads=self.recommend_n_threads,
            recommend_use_gpu_ranking=self.recommend_use_gpu_ranking,
            train_min_user_interactions=self.train_min_user_interactions,
            item_net_block_types=self.item_net_block_types,
            pos_encoding_type=self.pos_encoding_type,
            transformer_layers_type=self.transformer_layers_type,
            data_preparator_type=self.data_preparator_type,
            lightning_module_type=self.lightning_module_type,
            mask_prob=self.mask_prob,
        )

    @classmethod
    def _from_config(cls, config: BERT4RecModelConfig) -> tpe.Self:
        return cls(
            trainer=None,
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            n_factors=config.n_factors,
            use_pos_emb=config.use_pos_emb,
            use_causal_attn=config.use_causal_attn,
            use_key_padding_mask=config.use_key_padding_mask,
            dropout_rate=config.dropout_rate,
            session_max_len=config.session_max_len,
            dataloader_num_workers=config.dataloader_num_workers,
            batch_size=config.batch_size,
            loss=config.loss,
            n_negatives=config.n_negatives,
            gbce_t=config.gbce_t,
            lr=config.lr,
            epochs=config.epochs,
            verbose=config.verbose,
            deterministic=config.deterministic,
            recommend_device=config.recommend_device,
            recommend_n_threads=config.recommend_n_threads,
            recommend_use_gpu_ranking=config.recommend_use_gpu_ranking,
            train_min_user_interactions=config.train_min_user_interactions,
            item_net_block_types=config.item_net_block_types,
            pos_encoding_type=config.pos_encoding_type,
            transformer_layers_type=config.transformer_layers_type,
            data_preparator_type=config.data_preparator_type,
            lightning_module_type=config.lightning_module_type,
            mask_prob=config.mask_prob,
        )
