import typing as tp

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from rectools.semantic.tokenizer import SIDTokenizer


class TIGERDataset(TorchDataset):  # pylint: disable=too-many-instance-attributes
    """Dataset for the TIGER generative retrieval model.

    Accepts a pandas DataFrame of interactions, builds per-user item
    sequences, and tokenizes them into Semantic IDs.

    In train mode (``eval_mode=False``), each sample is a
    (history, next-item) pair:
      - ``input_ids``  : 1-D flattened encoder token sequence
      - ``dec_input``  : decoder input with BOS prepended
      - ``labels``     : raw SID codes of the target item

    In eval mode (``eval_mode=True``), each sample is:
      - ``input_ids``  : 1-D flattened encoder token sequence (all items except the last)
      - ``labels``     : raw item ID of the target (last item)

    Parameters
    ----------
    interactions : pd.DataFrame
        Interaction history with at least ``user_col`` and ``item_col`` columns.
    tokenizer : SIDTokenizer
        Trained tokenizer for converting item IDs to Semantic IDs.
    codebook_sizes : list of int
        Number of codes per RQ-VAE level.
    max_length : int
        Maximum number of history items per sequence.
    codeword_offset : int
        Offset added to codeword indices to reserve special tokens.
    bos_token_id : int
        Token ID used as beginning-of-sequence in the decoder.
    only_last : bool
        If True, only the last ``max_length + 1`` items per user are used.
        If False, all subsequences of length >= 2 are generated.
    eval_mode : bool
        If True, the dataset produces evaluation samples (raw item ID as label).
    max_users : int or None
        If set, subsample to at most this many users.
    user_col : str
        Name of the user ID column.
    item_col : str
        Name of the item ID column.
    timestamp_col : str
        Name of the timestamp column. If present in the DataFrame,
        interactions are sorted by (user_col, timestamp_col).
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        tokenizer: SIDTokenizer,
        codebook_sizes: tp.List[int],
        max_length: int = 20,
        codeword_offset: int = 2,
        bos_token_id: int = 1,
        only_last: bool = True,
        eval_mode: bool = False,
        max_users: tp.Optional[int] = None,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
    ) -> None:
        self.tokenizer = tokenizer
        self.codebook_sizes = codebook_sizes
        self.sid_len = len(codebook_sizes)
        self.max_length = max_length
        self.codeword_offset = codeword_offset
        self.bos_token_id = bos_token_id
        self.only_last = only_last
        self.eval_mode = eval_mode
        self.max_users = max_users
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col

        self.interactions = interactions

        self.offsets = np.cumsum([codeword_offset] + codebook_sizes)[:-1]

        df = interactions
        if timestamp_col in df.columns:
            df = df.sort_values([user_col, timestamp_col])

        user_sequences = df.groupby(user_col)[item_col].agg(list)

        self._create_sequences(user_sequences)

    def _create_sequences(self, user_sequences: pd.DataFrame) -> None:  # pylint: disable=too-many-branches
        user_ids = user_sequences.index.tolist()
        if self.max_users is not None and len(user_ids) > self.max_users:
            user_ids = np.random.choice(user_ids, size=self.max_users, replace=False).tolist()
            user_sequences = user_sequences.loc[user_ids]

        if self.eval_mode:
            self.sequences: tp.List[tp.List[tp.Tuple[int, ...]]] = []
            self.targets: tp.List[int] = []
            for seq in user_sequences:
                if len(seq) > self.max_length + 1:
                    tokenized = self.tokenizer.tokenize(seq[-self.max_length - 1 : -1])
                else:
                    tokenized = self.tokenizer.tokenize(seq[:-1])
                assert isinstance(tokenized, list)
                self.sequences.append(tokenized)
                self.targets.append(seq[-1])
        elif self.only_last:
            self.sequences = []
            for seq in user_sequences:
                seq = seq[-self.max_length - 1 :] if len(seq) > self.max_length + 1 else seq
                tokenized = self.tokenizer.tokenize(seq)
                assert isinstance(tokenized, list)
                self.sequences.append(tokenized)
        else:
            self.sequences = []
            for seq in tqdm(user_sequences, desc="Preparing the train data", ncols=120):
                if len(seq) < 2:
                    continue
                for t in range(1, len(seq)):
                    start = max(0, t - self.max_length)
                    tokenized = self.tokenizer.tokenize(seq[start : t + 1])
                    assert isinstance(tokenized, list)
                    self.sequences.append(tokenized)

    def __len__(self) -> int:
        return len(self.sequences)

    def _sid_to_tokens(self, sid: tp.Tuple[int, ...]) -> tp.List[int]:
        return [code + self.offsets[d] for d, code in enumerate(sid)]

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        if self.eval_mode:
            history_sids: tp.List[tp.Tuple[int, ...]] = self.sequences[idx]
            target = self.targets[idx]

            input_ids = np.array(
                [tok for sid in history_sids for tok in self._sid_to_tokens(sid)],
                dtype=np.int64,
            )
            return {"input_ids": input_ids, "labels": target}

        item_sequence: tp.List[tp.Tuple[int, ...]] = self.sequences[idx]
        history_sids = item_sequence[:-1]
        target_sid: tp.Tuple[int, ...] = item_sequence[-1]

        input_ids = np.array(
            [tok for sid in history_sids for tok in self._sid_to_tokens(sid)],
            dtype=np.int64,
        )

        target_tokens = np.array(self._sid_to_tokens(target_sid), dtype=np.int64)

        dec_input = np.concatenate([[self.bos_token_id], target_tokens[:-1]]).astype(np.int64)

        labels = np.array(target_sid, dtype=np.int64)

        return {"input_ids": input_ids, "dec_input": dec_input, "labels": labels}


class PaddingCollateFn:
    """Automatically right pad user interaction sequences and labels with specified padding values.

    Parameters
    ----------
    padding_value : int, optional
        Value to pad input sequences with, by default 0
    labels_padding_value : int, optional
        Value to pad labels with, by default -100
    """

    def __init__(self, padding_value: int = 0, labels_padding_value: int = -100) -> None:
        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value

    def __call__(self, batch: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, torch.Tensor]:
        """Apply padding to all fields in a batch."""
        collated = {}
        for key in batch[0]:
            if np.isscalar(batch[0][key]):
                collated[key] = torch.tensor([ex[key] for ex in batch])
                continue
            pad_val = self.labels_padding_value if key == "labels" else self.padding_value
            values = [torch.tensor(ex[key]) for ex in batch]
            collated[key] = pad_sequence(values, batch_first=True, padding_value=pad_val)
        return collated
