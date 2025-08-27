#  Copyright 2025 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import typing as tp
import warnings
from collections.abc import Hashable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from rectools import Columns, ExternalIds
from rectools.dataset import Dataset, Interactions
from rectools.dataset.features import DenseFeatures, Features, SparseFeatures
from rectools.dataset.identifiers import IdMap

from .constants import PADDING_VALUE
from .negative_sampler import TransformerNegativeSamplerBase

InitKwargs = tp.Dict[str, tp.Any]
# (user session, session weights, extra columns)
BatchElement = tp.Tuple[tp.List[int], tp.List[float], tp.Dict[str, tp.List[tp.Any]]]


class SequenceDataset(TorchDataset):
    """
    Dataset for sequential data.

    Parameters
    ----------
    sessions : List[List[int]]
        User sessions in the form of sequences of items ids.
    weights : List[List[float]]
        Weight of each interaction from the session.
    """

    def __init__(
        self,
        sessions: tp.List[tp.List[int]],
        weights: tp.List[tp.List[float]],
        extras: tp.Optional[tp.Dict[str, tp.List[tp.Any]]] = None,
    ):
        self.sessions = sessions
        self.weights = weights
        self.extras = extras

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int) -> BatchElement:
        session = self.sessions[index]  # [session_len]
        weights = self.weights[index]  # [session_len]
        extras = (
            {feature_name: features[index] for feature_name, features in self.extras.items()} if self.extras else {}
        )
        return session, weights, extras

    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        sort_users: bool = False,
    ) -> "SequenceDataset":
        """
        Group interactions by user.
        Construct SequenceDataset from grouped interactions.

        Parameters
        ----------
        interactions : pd.DataFrame
            User-item interactions.
        """
        cols_to_agg = [col for col in interactions.columns if col != Columns.User]
        sessions = (
            interactions.sort_values(Columns.Datetime, kind="stable")
            .groupby(Columns.User, sort=sort_users)[cols_to_agg]
            .agg(list)
        )
        sessions_items, weights = (
            sessions[Columns.Item].to_list(),
            sessions[Columns.Weight].to_list(),
        )
        extra_cols = [col for col in interactions.columns if col not in Columns.Interactions]
        extras = {col: sessions[col].to_list() for col in extra_cols} if len(extra_cols) > 0 else None
        return cls(sessions=sessions_items, weights=weights, extras=extras)


class TransformerDataPreparatorBase:  # pylint: disable=too-many-instance-attributes
    """
    Base class for data preparator. To change train/recommend dataset processing, train/recommend dataloaders inherit
    from this class and pass your custom data preparator to your model parameters.

    Parameters
    ----------
    session_max_len : int
        Maximum length of user sequence.
    batch_size : int
        How many samples per batch to load.
    dataloader_num_workers : int
        Number of loader worker processes.
    item_extra_tokens : Sequence(Hashable)
        Which element to use for sequence padding.
    shuffle_train : bool, default True
        If ``True``, reshuffles data at each epoch.
    train_min_user_interactions : int, default 2
        Minimum length of user sequence. Cannot be less than 2.
    get_val_mask_func : Callable, default None
        Function to get validation mask.
    n_negatives : optional(int), default ``None``
        Number of negatives for BCE, gBCE and sampled_softmax losses.
    negative_sampler: optional(TransformerNegativeSamplerBase), default ``None``
        Negative sampler.
    get_val_mask_func_kwargs: optional(InitKwargs), default ``None``
        Additional keyword arguments for the get_val_mask_func.
        Make sure all dict values have JSON serializable types.
    add_unix_ts: bool, default ``False``
        Add extra column ``unix_ts`` contains Column.Datetime converted to seconds
        from the beginning of the epoch
    extra_cols: optional(List[str]), default ``None``
        Extra columns to keep in train and recommend datasets.
    """

    # We sometimes need data preparators to add +1 to actual session_max_len
    # e.g. required by "Shifted Sequence" training objective (as in SASRecModel)
    train_session_max_len_addition: int = 0

    item_extra_tokens: tp.Sequence[Hashable] = (PADDING_VALUE,)

    def __init__(
        self,
        session_max_len: int,
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int = 2,
        get_val_mask_func: tp.Optional[tp.Callable] = None,
        shuffle_train: bool = True,
        n_negatives: tp.Optional[int] = None,
        negative_sampler: tp.Optional[TransformerNegativeSamplerBase] = None,
        get_val_mask_func_kwargs: tp.Optional[InitKwargs] = None,
        extra_cols: tp.Optional[tp.List[str]] = None,
        add_unix_ts: bool = False,
        **kwargs: tp.Any,
    ) -> None:
        self.item_id_map: IdMap
        self.extra_token_ids: tp.Dict
        self.train_dataset: Dataset
        self.val_interactions: tp.Optional[pd.DataFrame] = None
        self.session_max_len = session_max_len
        self.negative_sampler = negative_sampler
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_min_user_interactions = train_min_user_interactions
        self.shuffle_train = shuffle_train
        self.get_val_mask_func = get_val_mask_func
        self.get_val_mask_func_kwargs = get_val_mask_func_kwargs
        self.extra_cols = extra_cols
        self.add_unix_ts = add_unix_ts

    def get_known_items_sorted_internal_ids(self) -> np.ndarray:
        """Return internal item ids from processed dataset in sorted order."""
        return self.item_id_map.get_sorted_internal()[self.n_item_extra_tokens :]

    def get_known_item_ids(self) -> np.ndarray:
        """Return external item ids from processed dataset in sorted order."""
        return self.item_id_map.get_external_sorted_by_internal()[self.n_item_extra_tokens :]

    @staticmethod
    def _ensure_kwargs_dict(actual_kwargs: tp.Optional[InitKwargs]) -> InitKwargs:
        kwargs = {}
        if actual_kwargs is not None:
            kwargs = actual_kwargs
        return kwargs

    @property
    def n_item_extra_tokens(self) -> int:
        """Return number of padding elements"""
        return len(self.item_extra_tokens)

    @staticmethod
    def _process_features_for_id_map(
        raw_features: Features, raw_id_map: IdMap, id_map: IdMap, n_extra_tokens: int
    ) -> Features:
        raw_internal_ids = raw_id_map.convert_to_internal(id_map.get_external_sorted_by_internal()[n_extra_tokens:])
        sorted_features = raw_features.take(raw_internal_ids)
        n_features = sorted_features.values.shape[1]
        dtype = sorted_features.values.dtype

        if isinstance(raw_features, SparseFeatures):
            extra_token_feature_values = sparse.csr_matrix((n_extra_tokens, n_features), dtype=dtype)
            full_feature_values: sparse.scr_matrix = sparse.vstack(
                [extra_token_feature_values, sorted_features.values], format="csr"
            )
            return SparseFeatures.from_iterables(values=full_feature_values, names=raw_features.names)

        extra_token_feature_values = np.zeros((n_extra_tokens, n_features), dtype=dtype)
        full_feature_values = np.vstack([extra_token_feature_values, sorted_features.values])
        return DenseFeatures.from_iterables(values=full_feature_values, names=raw_features.names)

    def _filter_train_interactions(self, train_interactions: pd.DataFrame) -> pd.DataFrame:
        """Filter train interactions."""
        user_stats = train_interactions[Columns.User].value_counts()
        users = user_stats[user_stats >= self.train_min_user_interactions].index
        train_interactions = train_interactions[(train_interactions[Columns.User].isin(users))]
        train_interactions = (
            train_interactions.sort_values(Columns.Datetime, kind="stable")
            .groupby(Columns.User, sort=False)
            .tail(self.session_max_len + self.train_session_max_len_addition)
        )
        return train_interactions

    def _convert_to_unix_ts(self, datetime: pd.Series) -> pd.Series:
        return (datetime.values.astype("int64") / 10**9).astype("int64")

    def process_dataset_train(self, dataset: Dataset) -> None:
        """Process train dataset and save data."""
        extra_cols = False if self.extra_cols is None else self.extra_cols
        raw_interactions = dataset.get_raw_interactions(include_extra_cols=extra_cols)
        if self.add_unix_ts:
            raw_interactions["unix_ts"] = self._convert_to_unix_ts(raw_interactions[Columns.Datetime])

        # Exclude val interaction targets from train if needed
        interactions = raw_interactions
        if self.get_val_mask_func is not None:
            val_mask = self.get_val_mask_func(
                raw_interactions, **self._ensure_kwargs_dict(self.get_val_mask_func_kwargs)
            )
            interactions = raw_interactions[~val_mask]
            interactions.reset_index(drop=True, inplace=True)

        # Filter train interactions
        interactions = self._filter_train_interactions(interactions)

        # Prepare id maps
        user_id_map = IdMap.from_values(interactions[Columns.User].values)
        item_id_map = IdMap.from_values(self.item_extra_tokens)
        item_id_map = item_id_map.add_ids(interactions[Columns.Item])

        # Prepare item features
        item_features = None
        if dataset.item_features is not None:
            item_features = self._process_features_for_id_map(
                dataset.item_features, dataset.item_id_map, item_id_map, self.n_item_extra_tokens
            )

        # Prepare train dataset
        # User features are dropped for now because model doesn't support them
        final_interactions = Interactions.from_raw(
            interactions,
            user_id_map,
            item_id_map,
            keep_extra_cols=True,
        )
        self.train_dataset = Dataset(user_id_map, item_id_map, final_interactions, item_features=item_features)
        self.item_id_map = self.train_dataset.item_id_map
        self._init_extra_token_ids()

        # Define val interactions
        if self.get_val_mask_func is not None:
            val_targets = raw_interactions[val_mask]
            val_targets = val_targets[
                (val_targets[Columns.User].isin(user_id_map.external_ids))
                & (val_targets[Columns.Item].isin(item_id_map.external_ids))
            ]
            val_interactions = interactions[interactions[Columns.User].isin(val_targets[Columns.User].unique())].copy()
            val_interactions[Columns.Weight] = 0
            val_interactions = pd.concat([val_interactions, val_targets], axis=0)
            self.val_interactions = Interactions.from_raw(
                val_interactions, user_id_map, item_id_map, keep_extra_cols=True
            ).df

    def _init_extra_token_ids(self) -> None:
        extra_token_ids = self.item_id_map.convert_to_internal(self.item_extra_tokens)
        self.extra_token_ids = dict(zip(self.item_extra_tokens, extra_token_ids))

    def get_dataloader_train(self) -> DataLoader:
        """
        Construct train dataloader from processed dataset.

        Returns
        -------
        DataLoader
            Train dataloader.
        """
        sequence_dataset = SequenceDataset.from_interactions(self.train_dataset.interactions.df)
        train_dataloader = DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_train,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=self.shuffle_train,
        )
        return train_dataloader

    def get_dataloader_val(self) -> tp.Optional[DataLoader]:
        """
        Construct validation dataloader from processed dataset.

        Returns
        -------
        Optional(DataLoader)
            Validation dataloader.
        """
        if self.val_interactions is None:
            return None

        sequence_dataset = SequenceDataset.from_interactions(self.val_interactions)
        val_dataloader = DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_val,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )
        return val_dataloader

    def get_dataloader_recommend(self, dataset: Dataset, batch_size: int) -> DataLoader:
        """
        Construct recommend dataloader from processed dataset.

        Returns
        -------
        DataLoader
            Recommend dataloader.
        """
        # Recommend dataloader should return interactions sorted by user ids.
        # User ids here are internal user ids in dataset.interactions.df that was prepared for recommendations.
        # Sorting sessions by user ids will ensure that these ids will also be correct indexes in user embeddings matrix
        # that will be returned by the net.
        sequence_dataset = SequenceDataset.from_interactions(interactions=dataset.interactions.df, sort_users=True)
        recommend_dataloader = DataLoader(
            sequence_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_fn_recommend,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )
        return recommend_dataloader

    def transform_dataset_u2i(
        self,
        dataset: Dataset,
        users: ExternalIds,
        context: tp.Optional[pd.DataFrame] = None,
    ) -> Dataset:
        """
        Process dataset for u2i recommendations.
        Filter out interactions and adapt id maps.
        All users beyond target users for recommendations are dropped.
        All target users that do not have at least one known item in interactions are dropped.

        Parameters
        ----------
        dataset : Dataset
            RecTools dataset.
        users : ExternalIds
            Array of external user ids to recommend for.
        context : optional(pd.DataFrame), default  ``None``
            Optional DataFrame containing additional user context information (e.g., session features,
        demographics).

        Returns
        -------
        Dataset
            Processed RecTools dataset.
            Final dataset will consist only of model known items during fit and only of required
            (and supported) target users for recommendations.
            Final user_id_map is an enumerated list of supported (filtered) target users.
            Final item_id_map is model item_id_map constructed during training.
        """
        # Filter interactions in dataset internal ids
        required_cols = Columns.Interactions
        if self.extra_cols is not None:
            required_cols = required_cols + self.extra_cols
        interactions = dataset.interactions.df[required_cols]
        users_internal = dataset.user_id_map.convert_to_internal(users, strict=False)
        items_internal = dataset.item_id_map.convert_to_internal(self.get_known_item_ids(), strict=False)
        interactions = interactions[interactions[Columns.User].isin(users_internal)]
        interactions = interactions[interactions[Columns.Item].isin(items_internal)]

        # Convert to external ids
        interactions[Columns.Item] = dataset.item_id_map.convert_to_external(interactions[Columns.Item])
        interactions[Columns.User] = dataset.user_id_map.convert_to_external(interactions[Columns.User])

        # Prepare new user id mapping
        rec_user_id_map = IdMap.from_values(interactions[Columns.User])

        if context is not None:
            if not pd.Series(users).isin(context[Columns.User].unique()).all():
                raise ValueError("No context for some target users")
            if context.duplicated(subset=Columns.User).any():
                raise ValueError(
                    "Duplicated user entries found in context. Each user must have exactly one context row."
                )
            context[Columns.Item] = PADDING_VALUE  # External index pad element
            context = context[context[Columns.User].isin(interactions[Columns.User].unique())]
            interactions = pd.concat([interactions, context])
        if self.add_unix_ts:
            interactions["unix_ts"] = self._convert_to_unix_ts(interactions[Columns.Datetime])
        # Construct dataset
        # For now features are dropped because model doesn't support them on inference
        n_filtered = len(users) - rec_user_id_map.size
        if n_filtered > 0:
            explanation = f"""{n_filtered} target users were considered cold because of missing known items"""
            warnings.warn(explanation)
        filtered_interactions = Interactions.from_raw(
            interactions, rec_user_id_map, self.item_id_map, keep_extra_cols=True
        )
        filtered_dataset = Dataset(rec_user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def transform_dataset_i2i(self, dataset: Dataset) -> Dataset:
        """
        Process dataset for i2i recommendations.
        Filter out interactions and adapt id maps.

        Parameters
        ----------
        dataset: Dataset
            RecTools dataset.

        Returns
        -------
        Dataset
            Processed RecTools dataset.
            Final dataset will consist only of model known items during fit.
            Final user_id_map is the same as dataset original.
            Final item_id_map is model item_id_map constructed during training.
        """
        extra_cols = False if self.extra_cols is None else self.extra_cols
        interactions = dataset.get_raw_interactions(include_extra_cols=extra_cols)
        interactions = interactions[interactions[Columns.Item].isin(self.get_known_item_ids())]
        filtered_interactions = Interactions.from_raw(
            interactions, dataset.user_id_map, self.item_id_map, keep_extra_cols=True
        )
        filtered_dataset = Dataset(dataset.user_id_map, self.item_id_map, filtered_interactions)
        return filtered_dataset

    def _collate_fn_train(
        self,
        batch: tp.List[BatchElement],
    ) -> tp.Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _collate_fn_val(
        self,
        batch: tp.List[BatchElement],
    ) -> tp.Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def _collate_fn_recommend(
        self,
        batch: tp.List[BatchElement],
    ) -> tp.Dict[str, torch.Tensor]:
        raise NotImplementedError()
