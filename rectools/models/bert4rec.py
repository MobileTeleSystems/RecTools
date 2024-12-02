import typing as tp
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import nn

from rectools.models.sasrec import (
    CatFeaturesItemNet,
    IdEmbeddingsItemNet,
    ItemNetBase,
    LearnableInversePositionalEncoding,
    LossCalculatorBase,
    PointWiseFeedForward,
    PositionalEncodingBase,
    SessionEncoderDataPreparatorBase,
    SessionEncoderLightningModule,
    SessionEncoderLightningModuleBase,
    TransformerLayersBase,
    TransformerModelBase,
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
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            item_extra_tokens=item_extra_tokens,
            shuffle_train=shuffle_train,
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


class BERT4RecModel(TransformerModelBase):
    """TODO"""

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        n_blocks: int = 1,
        n_heads: int = 1,
        n_factors: int = 128,
        use_pos_emb: bool = True,
        use_causal_attn: bool = False,
        use_key_padding_mask: bool = True,
        dropout_rate: float = 0.2,
        epochs: int = 3,
        verbose: int = 0,
        deterministic: bool = False,
        cpu_n_threads: int = 0,
        session_max_len: int = 32,
        n_negatives: int = 1,
        batch_size: int = 128,
        loss: tp.Union[tp.Literal["softmax", "BCE", "gBCE"], LossCalculatorBase] = "softmax",
        gbce_t: float = 0.2,
        lr: float = 0.01,
        dataloader_num_workers: int = 0,
        train_min_user_interaction: int = 2,
        mask_prob: float = 0.15,
        trainer: tp.Optional[Trainer] = None,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]] = (IdEmbeddingsItemNet, CatFeaturesItemNet),
        pos_encoding_type: tp.Type[PositionalEncodingBase] = LearnableInversePositionalEncoding,
        transformer_layers_type: tp.Type[TransformerLayersBase] = BERT4RecTransformerLayers,
        data_preparator_type: tp.Type[BERT4RecDataPreparator] = BERT4RecDataPreparator,
        lightning_module_type: tp.Type[SessionEncoderLightningModuleBase] = SessionEncoderLightningModule,
    ):
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
            epochs=epochs,
            verbose=verbose,
            deterministic=deterministic,
            cpu_n_threads=cpu_n_threads,
            loss=loss,
            gbce_t=gbce_t,
            lr=lr,
            session_max_len=session_max_len + 1,
            trainer=trainer,
            item_net_block_types=item_net_block_types,
            pos_encoding_type=pos_encoding_type,
            lightning_module_type=lightning_module_type,
        )
        self.data_preparator = data_preparator_type(
            session_max_len=session_max_len,
            n_negatives=n_negatives if loss != "softmax" else None,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interaction,
            item_extra_tokens=(PADDING_VALUE, MASKING_VALUE),
            mask_prob=mask_prob,
        )
