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

# pylint: disable=attribute-defined-outside-init

import typing as tp

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rectools.dataset import DenseFeatures, IdMap, SparseFeatures
from rectools.dataset.features import DIRECT_FEATURE_VALUE
from tests.testing_utils import assert_sparse_matrix_equal


class TestDenseFeatures:
    def setup(self) -> None:
        self.values = np.array([[1, 10], [2, 20], [3, 30]])
        self.names = ("f1", "f2")

    def test_creation(self) -> None:
        features = DenseFeatures(self.values, self.names)
        np.testing.assert_equal(features.values, self.values)
        assert features.names == self.names

    @pytest.mark.parametrize("names", (("f1",), ("f1", "f2", "f3")))
    def test_shape_validation(self, names: tp.Tuple[str, ...]) -> None:
        with pytest.raises(ValueError):
            DenseFeatures(self.values, names)

    def test_from_dataframe_creation(self) -> None:
        df = pd.DataFrame(
            {
                "o": [10, 20, 30],
                "f1": [1, 2, 3],
                "f2": [10, 20, 30],
            },
        )
        id_map = IdMap.from_values([10, 20, 30])
        features = DenseFeatures.from_dataframe(df, id_col="o", id_map=id_map)
        np.testing.assert_equal(features.values, self.values)
        assert features.names == self.names

    def test_raises_when_in_dataframe_id_that_is_not_in_id_map(self) -> None:
        df = pd.DataFrame({"o": [10, 20, 30], "f1": [1, 2, 3]})
        id_map = IdMap.from_values([10, 30])
        with pytest.raises(ValueError):
            DenseFeatures.from_dataframe(df, id_col="o", id_map=id_map)

    def test_raises_when_in_id_map_id_that_is_not_in_dataframe(self) -> None:
        df = pd.DataFrame({"o": [10, 30], "f1": [1, 2]})
        id_map = IdMap.from_values([10, 20, 30])
        with pytest.raises(ValueError):
            DenseFeatures.from_dataframe(df, id_col="o", id_map=id_map)

    def test_get_dense(self) -> None:
        features = DenseFeatures(self.values, self.names)
        np.testing.assert_equal(features.get_dense(), self.values)

    def test_get_sparse(self) -> None:
        features = DenseFeatures(self.values, self.names)
        assert_sparse_matrix_equal(features.get_sparse(), sparse.csr_matrix(self.values))


class TestSparseFeatures:
    def setup(self) -> None:
        self.values = sparse.csr_matrix(
            [
                [3.2, 0, 1],
                [2.4, 2, 0],
                [0.0, 0, 1],
                [1.2, 3, 2],
            ],
        )
        self.names = (("f1", None), ("f2", 100), ("f2", 200))

    def test_creation(self) -> None:
        features = SparseFeatures(self.values, self.names)
        assert_sparse_matrix_equal(features.values, self.values)
        assert features.names == self.names

    def test_shape_validation(self) -> None:
        names = (("f1", None), ("f2", 100))
        with pytest.raises(ValueError):
            SparseFeatures(self.values, names)

    @pytest.mark.parametrize(
        "weights,expected_matrix",
        (
            (
                None,
                np.array(
                    [
                        [12, 4, 1, 0, 1],
                        [3, 0, 0, 2, 1],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                [1, 2, 1, 0.75, 1, 3, 1, 1, 0.5],
                np.array(
                    [
                        [19, 3, 1, 0, 1],
                        [3, 0, 0, 4, 0.5],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ),
    )
    def test_from_dataframe_creation(self, weights: tp.Optional[tp.List[float]], expected_matrix: np.ndarray) -> None:
        df = pd.DataFrame(
            [
                [10, "f1", 5],
                [10, "f1", 7],
                [20, "f1", 3],
                [10, "f2", 4],
                [10, "f3", "v1"],
                [20, "f4", 100],
                [10, "f4", 200],
                [20, "f4", 100],
                [20, "f4", 200],
            ],
            columns=["o", "f", "v"],
        )
        if weights is not None:
            df["w"] = weights
        id_map = IdMap.from_values([10, 20, 30])
        features = SparseFeatures.from_flatten(
            df,
            id_map=id_map,
            cat_features=["f3", "f4"],
            id_col="o",
            feature_col="f",
            value_col="v",
            weight_col="w",
        )
        expected_values = sparse.csr_matrix(expected_matrix)
        expected_names = (
            ("f1", DIRECT_FEATURE_VALUE),
            ("f2", DIRECT_FEATURE_VALUE),
            ("f3", "v1"),
            ("f4", 100),
            ("f4", 200),
        )
        assert_sparse_matrix_equal(features.values, expected_values)
        assert features.names == expected_names

    def test_from_dataframe_creation_only_direct(self) -> None:
        df = pd.DataFrame(
            [
                [10, "f1", 5],
                [10, "f1", 7],
                [20, "f1", 3],
                [10, "f2", 4],
            ],
            columns=["id", "feature", "value"],
        )
        id_map = IdMap.from_values([10, 20, 30])
        features = SparseFeatures.from_flatten(df, id_map=id_map, cat_features=["f3", "f4"])
        expected_values = sparse.csr_matrix([[12, 4], [3, 0], [0, 0]], dtype=float)
        assert_sparse_matrix_equal(features.values, expected_values)

    def test_from_dataframe_creation_only_categorical(self) -> None:
        df = pd.DataFrame(
            [
                [10, "f3", "v1"],
                [20, "f4", 100],
                [10, "f4", 200],
                [20, "f4", 100],
                [20, "f4", 200],
            ],
            columns=["id", "feature", "value"],
        )
        id_map = IdMap.from_values([10, 20, 30])
        features = SparseFeatures.from_flatten(df, id_map=id_map, cat_features=["f3", "f4"])
        expected_values = sparse.csr_matrix([[1, 0, 1], [0, 2, 1], [0, 0, 0]], dtype=float)
        assert_sparse_matrix_equal(features.values, expected_values)

    def test_get_dense(self) -> None:
        features = SparseFeatures(self.values, self.names)
        np.testing.assert_equal(features.get_dense(), self.values.toarray())

    def test_get_sparse(self) -> None:
        features = SparseFeatures(self.values, self.names)
        assert_sparse_matrix_equal(features.get_sparse(), self.values)

    @pytest.mark.parametrize("cat_features", ([], [1]))
    def test_raises_when_weight_not_numeric(self, cat_features: tp.List[tp.Any]) -> None:
        df = pd.DataFrame(
            [
                [1, 1, 1, 10],
                [1, 1, 2, "w"],
            ],
            columns=["id", "feature", "value", "weight"],
        )
        with pytest.raises(TypeError) as e:
            SparseFeatures.from_flatten(df, id_map=IdMap.from_values([1]), cat_features=cat_features)
        err_text = e.value.args[0]
        assert "weight" in err_text.lower()

    def test_raises_when_direct_feature_value_not_numeric(self) -> None:
        df = pd.DataFrame(
            [
                [1, 1, 1],
                [1, 2, "v"],
            ],
            columns=["id", "feature", "value"],
        )
        with pytest.raises(TypeError) as e:
            SparseFeatures.from_flatten(df, id_map=IdMap.from_values([1]))
        err_text = e.value.args[0]
        assert "value" in err_text.lower()
        assert "direct" in err_text.lower()

    def test_raises_when_unknown_ids(self) -> None:
        df = pd.DataFrame(
            [
                [1, 1, 10],
                [2, 1, 20],
            ],
            columns=["id", "feature", "value"],
        )
        id_map = IdMap.from_values([1])
        with pytest.raises(KeyError) as e:
            SparseFeatures.from_flatten(df, id_map=id_map)
        err_text = e.value.args[0]
        assert "id_map" in err_text.lower()

    @pytest.mark.parametrize("column", ("id", "feature", "value"))
    def test_raises_when_no_columns(self, column: str) -> None:
        df = pd.DataFrame(columns=["id", "feature", "value"])
        id_map = IdMap.from_values([])
        df.drop(columns=column, inplace=True)
        with pytest.raises(KeyError) as e:
            SparseFeatures.from_flatten(df, id_map=id_map)
        err_text = e.value.args[0]
        assert column in err_text.lower()
