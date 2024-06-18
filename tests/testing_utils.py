#  Copyright 2022-2024 MTS (Mobile Telesystems)
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

import numpy as np
import pandas as pd
from scipy import sparse

from rectools.dataset import DenseFeatures, Features, IdMap, Interactions, SparseFeatures


def assert_sparse_matrix_equal(actual: sparse.spmatrix, expected: sparse.spmatrix) -> None:
    assert isinstance(actual, type(expected))
    np.testing.assert_equal(actual.toarray(), expected.toarray())


def assert_id_map_equal(actual: IdMap, expected: IdMap) -> None:
    assert isinstance(actual, type(expected))
    pd.testing.assert_series_equal(actual.to_internal, expected.to_internal)


def assert_interactions_set_equal(actual: Interactions, expected: Interactions) -> None:
    assert isinstance(actual, type(expected))
    pd.testing.assert_frame_equal(actual.df, expected.df)


def assert_feature_set_equal(actual: tp.Optional[Features], expected: tp.Optional[Features]) -> None:
    if actual is None and expected is None:
        return

    assert isinstance(actual, type(expected))

    if isinstance(actual, DenseFeatures) and isinstance(expected, DenseFeatures):
        np.testing.assert_equal(actual.values, expected.values)
        assert actual.names == expected.names

    if isinstance(actual, SparseFeatures) and isinstance(expected, SparseFeatures):
        assert_sparse_matrix_equal(actual.values, expected.values)
        assert actual.names == expected.names
