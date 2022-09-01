#  Copyright 2022 MTS (Mobile Telesystems)
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

"""Distance metrics."""

import typing as tp
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import pandas as pd

from rectools import ExternalIds
from rectools.dataset import IdMap, SparseFeatures
from rectools.utils import fast_isin

Distances = tp.Union[tp.Sequence[float], np.ndarray]


class PairwiseDistanceCalculator(ABC):
    """Base pairwise distance calculator class"""

    def __getitem__(self, pair_index_sequences: tp.Tuple[ExternalIds, ExternalIds]) -> Distances:
        self._check_input_index_sequences(pair_index_sequences)
        return self._get_distances_for_item_pairs(pair_index_sequences[0], pair_index_sequences[1])

    @abstractmethod
    def _get_distances_for_item_pairs(self, items_0: ExternalIds, items_1: ExternalIds) -> Distances:
        pass

    def _check_input_index_sequences(self, pair_index_sequences: tp.Tuple[ExternalIds, ExternalIds]) -> None:
        if len(pair_index_sequences) != 2:
            raise IndexError("class returns distances only for an item PAIR index sequences")

        if not self._is_sequence(pair_index_sequences[0]) & self._is_sequence(pair_index_sequences[1]):
            raise TypeError("class returns distances for index SEQUENCES")

        if len(pair_index_sequences[0]) != len(pair_index_sequences[1]):
            raise IndexError("different lengths of index sequences")

    @staticmethod
    def _is_sequence(items: ExternalIds) -> bool:
        return bool(isinstance(items, np.ndarray) | (isinstance(items, Sequence) & ~isinstance(items, str)))


class PairwiseHammingDistanceCalculator(PairwiseDistanceCalculator):
    """Class for computing Hamming distance between a pair of items.

    Parameters
    ----------
    item_features_df: pandas dataframe with feature values and item ids as index
    """

    def __init__(self, item_features_df: pd.DataFrame) -> None:
        self.features_df = item_features_df.copy()

    def _get_distances_for_item_pairs(self, items_0: ExternalIds, items_1: ExternalIds) -> np.ndarray:
        features_0 = self.features_df.reindex(items_0).values
        features_1 = self.features_df.reindex(items_1).values

        mask_items_0_with_absent_feature_values = np.isnan(features_0).any(axis=1)
        mask_items_1_with_absent_feature_values = np.isnan(features_1).any(axis=1)

        if mask_items_0_with_absent_feature_values.any() | mask_items_1_with_absent_feature_values.any():
            warnings.warn(
                "Some items has absent feature values"
                " (NaN values in some columns of item_features_df or complete absence of corresponding rows)."
                " Corresponding pair distances are set to NaN."
            )

        result = np.sum(features_0 != features_1, axis=1).astype(np.float64)
        result[mask_items_0_with_absent_feature_values | mask_items_1_with_absent_feature_values] = np.nan
        return result


class SparsePairwiseHammingDistanceCalculator(PairwiseDistanceCalculator):
    """
    Class for computing Hamming distance between multiple pairs of elements
    represented in features matrix in sparse form.

    ATTENTION! An incorrect value may occur for float type matrix because nonsafe comparison isung (!=)

    Parameters
    ----------
    features : :class:`~rectools.dataset.SparseFeatures`
        Storage for sparse features with csr_matrix as field
        where the row index is associated with the identifier by id_map.
    id_map : :class:`~rectools.dataset.IdMap`
        Mapper which include mapping info between external and internal representations for
        all identificators for which you're planning search distances.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from rectools.dataset import IdMap, SparseFeatures
    >>> from rectools.metrics import SparsePairwiseHammingDistanceCalculator
    >>> features_matrix = csr_matrix(
    ...     [
    ...         [0, 0],
    ...         [0, 1],
    ...         [1, 1],
    ...     ])
    >>> features = SparseFeatures(values=features_matrix, names=["feature_1", "feature_2"])
    >>> mapper = IdMap.from_values(["i1", "i2", "i3", "i4", "i5"])
    >>> calculator = SparsePairwiseHammingDistanceCalculator(features, mapper)
    >>> calculator[
    ...    ["i1", "i1", "i1"],
    ...    ["i1", "i2", "i3"]
    ... ]
    array([0., 1., 2.], dtype=float32)
    """

    def __init__(self, features: SparseFeatures, id_map: IdMap) -> None:
        self.features = features.values.copy()
        self.mapper = deepcopy(id_map)

    def _get_distances_for_item_pairs(self, items_0: ExternalIds, items_1: ExternalIds) -> Distances:
        # Create accumulator for result
        result = np.empty(len(items_0), dtype=np.float32)
        # Find mask external ids that are not contained in the mapper
        existing_external_0 = fast_isin(np.asarray(items_0), self.mapper.external_ids)
        existing_external_1 = fast_isin(np.asarray(items_1), self.mapper.external_ids)
        existing_mask = np.logical_and(existing_external_0, existing_external_1)
        # Check absence items ids in mapper
        if not existing_mask.all():
            warnings.warn("Some items absent in mapper. Corresponding pair distances are set to NaN.")
            # Set nan to absent ids place
            result[~existing_mask] = np.nan
            # Get view for existing items
            external_indexes_0 = np.asarray(items_0)[existing_mask]
            external_indexes_1 = np.asarray(items_1)[existing_mask]
        else:
            external_indexes_0 = items_0
            external_indexes_1 = items_1
        # Get internal ids of existing external ids
        internal_indexes_0 = self.mapper.convert_to_internal(external_indexes_0)
        internal_indexes_1 = self.mapper.convert_to_internal(external_indexes_1)
        # Select the rows corresponding to the compared left and right elements
        try:
            features_0 = self.features[internal_indexes_0, :]
            features_1 = self.features[internal_indexes_1, :]
            # Set result values
            result[existing_mask] = np.ravel((features_0 != features_1).sum(axis=1))
        except IndexError:
            raise IndexError("Features matrix not contains row index, which associates with item identifier")
        # Find indexes of nan elements in left and right compared elements
        nan_indexes_0 = np.argwhere(np.isnan(features_0.data)).ravel()
        nan_indexes_1 = np.argwhere(np.isnan(features_1.data)).ravel()
        # Find rows indexes of nan elements in left and right compared elements
        indexes_items_0_with_absent_feature_values = np.searchsorted(features_0.indptr, nan_indexes_0, side="right") - 1
        indexes_items_1_with_absent_feature_values = np.searchsorted(features_1.indptr, nan_indexes_1, side="right") - 1
        if indexes_items_0_with_absent_feature_values.size + indexes_items_1_with_absent_feature_values.size > 0:
            warnings.warn(
                "Some items has absent feature values"
                " (NaN values in some columns of item_features_df or complete absence of corresponding rows)."
                " Corresponding pair distances are set to NaN."
            )
            # Set result values to nan if some features absence
            local_indexes = np.union1d(
                indexes_items_0_with_absent_feature_values, indexes_items_1_with_absent_feature_values
            )
            nan_indexes = np.argwhere(existing_mask).ravel()[local_indexes]
            result[nan_indexes] = np.nan
        return result
