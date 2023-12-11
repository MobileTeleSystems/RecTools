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

"""Structures to save explicit features."""

import typing as tp
import warnings

import attr
import numpy as np
import pandas as pd
from scipy import sparse

from rectools import InternalIds

from .identifiers import IdMap

DIRECT_FEATURE_VALUE = "__is_direct_feature"


class UnknownIdError(KeyError):
    """The error is raised when there are some ids in the dataframe that are not present in the id map"""


class AbsentIdError(ValueError):
    """The error is raised when there are some ids in the id map that are not present in the dataframe"""


@attr.s(slots=True, frozen=True)
class DenseFeatures:
    """
    Storage for dense features.

    Dense features are represented as a dense matrix,
    where rows correspond to objects, columns - to features.

    Usually you do not need to create this object directly,
    use `from_dataframe` class method instead.
    If you want to use custom logic,
    use `from_iterables` class method instead of direct creation.

    Parameters
    ----------
    values : np.ndarray
        Matrix of feature values in the classic format:
        rows - objects, columns - features.
    names : tuple(str)
        Names of features (number of names must be equal to the number of columns in values).
    """

    values: np.ndarray = attr.ib()
    names: tp.Tuple[str, ...] = attr.ib()

    @names.validator
    def _check_names_length(self, _: str, value: tp.Tuple[str]) -> None:
        if len(value) != self.values.shape[1]:
            raise ValueError(f"Number of features is {self.values.shape[1]}, but number of names is {len(value)}")

    @classmethod
    def from_iterables(cls, values: tp.Iterable[tp.Iterable[float]], names: tp.Iterable[str]) -> "DenseFeatures":
        """
        Create class instance from any iterables of feature values and names.

        Parameters
        ----------
        values : iterable(iterable(float))
            Feature values matrix.
            E.g. list of lists: [[1, 2, 3], [4, 5, 6]].
        names : iterable(str)
            Feature names.

        Returns
        -------
        DenseFeatures
        """
        return cls(
            values=np.asarray(values, dtype=np.float32),
            names=tuple(names),
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, id_map: IdMap, id_col: str = "id") -> "DenseFeatures":
        """
        Create DenseFeatures object from dataframe.

        Assume that feature values are values in dataframe, and feature names are column names.

        Parameters
        ----------
        df : pd.Dataframe
            Table in classic format: rows corresponds to objects, columns - to features.
            One special column `id_col` must contain object external ids.
        id_map : IdMap
            Mapping between external and internal ids.
            Sets of ids in `id_map` and in `df` must be equal.
        id_col : str, default ``id``
            Name of column containing object ids.

        Returns
        -------
        DenseFeatures
        """
        extern_ids = df[id_col]
        df_ids = set(df[id_col].values)
        if len(df_ids) != len(df):
            raise ValueError("Ids in dataframe must be unique")

        map_ids = set(id_map.external_ids)
        if df_ids - map_ids:
            raise UnknownIdError("All ids in `df` must be present in `id_map`")

        if map_ids - df_ids:
            raise AbsentIdError("In `df` must be present all ids from `id_map`")

        features = df.drop(columns=id_col)
        values = features.values
        names = features.columns

        inner_ids = extern_ids.map(id_map.to_internal)
        sorter = np.argsort(inner_ids)
        values = values[sorter]

        return cls.from_iterables(values, names)

    def get_dense(self) -> np.ndarray:
        """Return values in dense format."""
        return self.values

    def get_sparse(self) -> sparse.csr_matrix:
        """Return values in sparse format."""
        return sparse.csr_matrix(self.values)

    def take(self, ids: InternalIds) -> "DenseFeatures":
        """
        Take a subset of features for given subject (user or item) ids.

        Parameters
        ----------
        ids : array-like
            Array of internal ids to select features for.

        Returns
        -------
        DenseFeatures

        """
        return DenseFeatures(
            values=self.values[ids],
            names=self.names,
        )


SparseFeatureName = tp.Tuple[str, tp.Any]


@attr.s(slots=True, frozen=True)
class SparseFeatures:
    """
    Storage for sparse features.

    Sparse features are represented as CSR matrix,
    where rows correspond to objects, columns - to features.
    Assume that there are features of two types: direct and categorical.

    Each direct feature is represented in a single column with its real values.
    Direct features are numeric.
    E.g.
    +---+----+----+
    |   | f1 | f2 |
    +---+----+----+
    | 1 | 23 |  3 |
    +---+----+----+
    | 2 | 36 |  5 |
    +---+----+----+

    Categorical features are one-hot encoded
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html),
    values in matrix are counts in category.
    If you want to binarize a numeric feature,
    make it categorical with bin indices as categories.
    E.g.
    +---+------+------+------+
    |   | f1_a | f1_b | f2_1 |
    +---+------+------+------+
    | 1 |   0  |   2  |   1  |
    +---+------+------+------+
    | 2 |   1  |   1  |   0  |
    +---+------+------+------+

    Usually you do not need to create this object directly, use `from_flatten` class method instead.
    If you want to use custom logic, use `from_iterables` class method instead of direct creation.

    Parameters
    ----------
    values : csr_matrix
        CSR matrix containing OHE feature values.
    names : tuple(tuple(str, any))
        Tuple of feature names.
        Direct features are represented only by names, so for direct features use (``feature name``, `None`).
        For sparse features use (``feature name``, ``value``), as they are one-hot encoded.
        E.g. If you have direct feature `age` and cat. feature `sex`, names will be *((age, None), (sex, m), (sex, f))*.
        Number of names must be equal to the number of columns in values.
    """

    values: sparse.csr_matrix = attr.ib()
    names: tp.Tuple[SparseFeatureName, ...] = attr.ib()

    @names.validator
    def _check_names_length(self, _: str, value: tp.Tuple[str]) -> None:
        if len(value) != self.values.shape[1]:
            raise ValueError(f"Number of features is {self.values.shape[1]}, but number of names is {len(value)}")

    @classmethod
    def from_iterables(
        cls,
        values: sparse.csr_matrix,
        names: tp.Iterable[tp.Tuple[str, tp.Any]],
    ) -> "SparseFeatures":
        """
        Create class instance from sparse matrix and iterable feature names.

        Parameters
        ----------
        values : csr_matrix
            Feature values matrix.
        names : iterable((str, any))
            Feature names in same format as in constructor.

        Returns
        -------
        SparseFeatures
        """
        return cls(
            values=values.astype(np.float32),
            names=tuple(names),
        )

    @classmethod
    def from_flatten(
        cls,
        df: pd.DataFrame,
        id_map: IdMap,
        cat_features: tp.Iterable[tp.Any] = (),
        id_col: str = "id",
        feature_col: str = "feature",
        value_col: str = "value",
        weight_col: str = "weight",
    ) -> "SparseFeatures":
        """
        Construct `SparseFeatures` from flatten DataFrame.

        Flatten DataFrame has 3 obligatory columns: <id of object>, <feature name>, <feature value>,
        and <feature weight> as the optional fourth.
        If there is no <feature weight> column, all weights will be assumed to be equal to ``1``.

        Direct features converted to sparse matrix as is.
        Its values are multiplied by weights.
        Values for the same object and same feature are added up.
        E.g:
        +----+---------+-------+--------+
        | id | feature | value | weight |
        +----+---------+-------+--------+
        |  1 |  f1     | 10    | 1      |
        +----+---------+-------+--------+
        |  2 |  f1     | 20    | 1.5    |
        +----+---------+-------+--------+
        |  1 |  f1     | 15    | 1      |
        +----+---------+-------+--------+
        | 2  |  f2     | 3     | 1      |
        +----+---------+-------+--------+
        Out:
        +---+----+----+
        |   | f1 | f2 |
        +---+----+----+
        | 1 | 25 |    |
        +---+----+----+
        | 2 | 30 | 3  |
        +---+----+----+

        Categorical features are represented as horizontally stacked one-hot vectors.
        Duplicated values are counted.
        Final values (counts) are multiplied by weights.
        E.g:
        +----+---------+-------+--------+
        | id | feature | value | weight |
        +----+---------+-------+--------+
        | 1  | f1      | 10    | 1      |
        +----+---------+-------+--------+
        | 2  | f1      | 20    | 1.5    |
        +----+---------+-------+--------+
        | 1  | f1      | 10    | 1      |
        +----+---------+-------+--------+
        | 2  | f2      | 3     | 1      |
        +----+---------+-------+--------+

        Out:
        +---+--------+--------+-------+
        |   | f1__10 | f1__20 | f2__3 |
        +---+--------+--------+-------+
        | 1 | 2      |        |       |
        +---+--------+--------+-------+
        | 2 |        | 1.5    | 1     |
        +---+--------+--------+-------+

        Parameters
        ----------
        df : pd.DataFrame
            Flatten table with features with columns
            `id_col`, `feature_col`, `value_col`
            in format described above.
        id_map : IdMap
            Mapping between external and internal ids.
        cat_features : iterable(str), default ``()``
            List of categorical feature names.
        id_col : str, default ``id``
            Name of column with object ids.
        feature_col : str, default ``feature``
            Name of column with feature names.
        value_col : str, default ``value``
            Name of column with feature values.
        weight_col : str, default ``weight``
            Name of column with feature weight.
            If no such column provided, all weights will be equal to ``1``.

        Returns
        -------
        SparseFeatures
        """
        required_columns = {id_col, feature_col, value_col}
        actual_columns = set(df.columns)
        if not actual_columns >= required_columns:
            raise KeyError(f"Missed columns {required_columns - actual_columns}")

        try:
            ids = id_map.convert_to_internal(df[id_col])
        except KeyError:
            raise UnknownIdError("All ids in `df` must be present in `id_map`")
        try:
            weights = df[weight_col].values.astype(float) if weight_col in df else 1
        except ValueError:
            raise TypeError("Weights must be numeric")

        df = pd.DataFrame({"id": ids, "feature": df[feature_col], "value": df[value_col], "weight": weights})

        all_features = df["feature"].unique()
        direct_features = set(all_features) - set(cat_features)

        csr_direct, names_direct = cls._make_direct_features(df, direct_features, id_map.internal_ids.size)
        csrs = [csr_direct]
        names_all = names_direct

        for cat_feature in cat_features:
            csr, names = cls._make_cat_feature(df, cat_feature, id_map.internal_ids.size)
            csrs.append(csr)
            names_all.extend(names)

        csr = sparse.hstack(csrs, format="csr")
        csr.sum_duplicates()

        return cls.from_iterables(csr, names_all)

    @classmethod
    def _make_direct_features(
        cls,
        df: pd.DataFrame,
        features: tp.Collection[tp.Any],
        n_objects: int,
    ) -> tp.Tuple[sparse.csr_matrix, tp.List[SparseFeatureName]]:
        df = df.query("feature in @features")
        features_map = IdMap.from_values(df["feature"].unique())
        try:
            values = df["value"].values.astype(np.float32)
        except ValueError:
            raise TypeError("Values of direct features must be numeric")
        csr = sparse.csr_matrix(
            (
                values * df["weight"],
                (
                    df["id"],
                    df["feature"].map(features_map.to_internal).values,
                ),
            ),
            shape=(n_objects, len(features)),
        )
        names = [(feature, DIRECT_FEATURE_VALUE) for feature in features_map.get_external_sorted_by_internal()]
        return csr, names

    @classmethod
    def _make_cat_feature(
        cls,
        df: pd.DataFrame,
        feature: str,
        n_objects: int,
    ) -> tp.Tuple[sparse.csr_matrix, tp.List[SparseFeatureName]]:
        df = df.query("feature == @feature")
        unq_feature_values = df["value"].unique()
        n_unq_values = len(unq_feature_values)
        ids = np.arange(n_unq_values)
        value_map = pd.Series(ids, index=unq_feature_values)
        csr = sparse.csr_matrix(
            (np.ones(len(df)) * df["weight"], (df["id"], df["value"].map(value_map))),
            shape=(n_objects, n_unq_values),
        )
        names = [(feature, value) for value in value_map.index.values[np.argsort(value_map.values)]]
        return csr, names

    def get_dense(self) -> np.ndarray:
        """Return values in dense format."""
        warnings.warn("Converting sparse features to dense array may cause MemoryError")
        return self.values.toarray()

    def get_sparse(self) -> sparse.csr_matrix:
        """Return values in sparse format."""
        return self.values

    def take(self, ids: InternalIds) -> "SparseFeatures":
        """
        Take a subset of features for given subject (user or item) ids.

        Parameters
        ----------
        ids : array-like
            Array of internal ids to select features for.

        Returns
        -------
        SparseFeatures
        """
        return SparseFeatures(
            values=self.values[ids],
            names=self.names,
        )


Features = tp.Union[DenseFeatures, SparseFeatures]
