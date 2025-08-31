#  Copyright 2022-2025 MTS (Mobile Telesystems)
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

"""Dataset - all data container."""

import typing as tp
from collections.abc import Hashable

import attr
import numpy as np
import pandas as pd
import typing_extensions as tpe
from pydantic import PlainSerializer
from scipy import sparse

from rectools import Columns
from rectools.utils.config import BaseConfig

from .features import AbsentIdError, DenseFeatures, Features, SparseFeatureName, SparseFeatures
from .identifiers import IdMap
from .interactions import Interactions

AnyFeatureName = tp.Union[str, SparseFeatureName]


def _serialize_feature_name(spec: tp.Any) -> Hashable:
    type_error = TypeError(
        f"""
        Serialization for feature name '{spec}' is not supported.
        Please convert your feature names and category feature values to strings, numbers, booleans
        or their tuples.
        """
    )
    if isinstance(spec, (list, np.ndarray)):
        raise type_error
    if isinstance(spec, tuple):
        return tuple(_serialize_feature_name(item) for item in spec)
    if isinstance(spec, (int, float, str, bool)):
        return spec
    if hasattr(spec, "dtype") and (np.issubdtype(spec.dtype, np.number) or np.issubdtype(spec.dtype, np.bool_)):
        # numpy str is handled by isinstance(spec, str)
        return spec.item()
    raise type_error


FeatureName = tpe.Annotated[AnyFeatureName, PlainSerializer(_serialize_feature_name, when_used="json")]
DatasetSchemaDict = tp.Dict[str, tp.Any]


class BaseFeaturesSchema(BaseConfig):
    """Features schema."""

    names: tp.Tuple[FeatureName, ...]


class DenseFeaturesSchema(BaseFeaturesSchema):
    """Dense features schema."""

    kind: tp.Literal["dense"] = "dense"


class SparseFeaturesSchema(BaseFeaturesSchema):
    """Sparse features schema."""

    kind: tp.Literal["sparse"] = "sparse"
    cat_feature_indices: tp.List[int]
    cat_n_stored_values: int


FeaturesSchema = tp.Union[DenseFeaturesSchema, SparseFeaturesSchema]


class IdMapSchema(BaseConfig):
    """IdMap schema."""

    size: int
    dtype: str


class EntitySchema(BaseConfig):
    """Entity schema."""

    n_hot: int
    id_map: IdMapSchema
    features: tp.Optional[FeaturesSchema] = None


class DatasetSchema(BaseConfig):
    """Dataset schema."""

    n_interactions: int
    users: EntitySchema
    items: EntitySchema


@attr.s(slots=True, frozen=True)
class Dataset:
    """
    Container class for all data for a recommendation model.

    It stores data about internal-external id mapping,
    user-item interactions, user and item features
    in special `rectools` structures for convenient future usage.

    WARNING: It's highly not recommended to create `Dataset` object directly.
    Use `construct` class method instead.

    Parameters
    ----------
    user_id_map : IdMap
        User identifiers mapping.
    item_id_map : IdMap
        Item identifiers mapping.
    interactions : Interactions
        User-item interactions.
    user_features : DenseFeatures or SparseFeatures, optional
        User explicit features.
    item_features : DenseFeatures or SparseFeatures, optional
        Item explicit features.
    """

    user_id_map: IdMap = attr.ib()
    item_id_map: IdMap = attr.ib()
    interactions: Interactions = attr.ib()
    user_features: tp.Optional[Features] = attr.ib(default=None)
    item_features: tp.Optional[Features] = attr.ib(default=None)

    @staticmethod
    def _get_feature_schema(features: tp.Optional[Features]) -> tp.Optional[FeaturesSchema]:
        if features is None:
            return None
        if isinstance(features, SparseFeatures):
            return SparseFeaturesSchema(
                names=features.names,
                cat_feature_indices=features.cat_feature_indices.tolist(),
                cat_n_stored_values=features.get_cat_features().values.nnz,
            )
        return DenseFeaturesSchema(
            names=features.names,
        )

    @staticmethod
    def _get_id_map_schema(id_map: IdMap) -> IdMapSchema:
        return IdMapSchema(size=id_map.size, dtype=id_map.external_dtype.str)

    def get_schema(self) -> DatasetSchemaDict:
        """Get dataset schema in a dict form that contains all the information about the dataset and its statistics."""
        user_schema = EntitySchema(
            n_hot=self.n_hot_users,
            id_map=self._get_id_map_schema(self.user_id_map),
            features=self._get_feature_schema(self.user_features),
        )
        item_schema = EntitySchema(
            n_hot=self.n_hot_items,
            id_map=self._get_id_map_schema(self.item_id_map),
            features=self._get_feature_schema(self.item_features),
        )
        schema = DatasetSchema(
            n_interactions=self.interactions.df.shape[0],
            users=user_schema,
            items=item_schema,
        )
        return schema.model_dump(mode="json")

    @property
    def n_hot_users(self) -> int:
        """
        Return number of hot users in dataset.
        Users with internal ids from `0` to `n_hot_users - 1` are hot (they are present in interactions).
        Users with internal ids from `n_hot_users` to `dataset.user_id_map.size - 1` are warm
        (they aren't present in interactions, but they have features).
        """
        return self.interactions.df[Columns.User].max() + 1

    @property
    def n_hot_items(self) -> int:
        """
        Return number of hot items in dataset.
        Items with internal ids from `0` to `n_hot_items - 1` are hot (they are present in interactions).
        Items with internal ids from `n_hot_items` to `dataset.item_id_map.size - 1` are warm
        (they aren't present in interactions, but they have features).
        """
        return self.interactions.df[Columns.Item].max() + 1

    def get_hot_user_features(self) -> tp.Optional[Features]:
        """User features for hot users."""
        if self.user_features is None:
            return None
        return self.user_features.take(range(self.n_hot_users))

    def get_hot_item_features(self) -> tp.Optional[Features]:
        """Item features for hot items."""
        if self.item_features is None:
            return None
        return self.item_features.take(range(self.n_hot_items))

    @classmethod
    def construct(
        cls,
        interactions_df: pd.DataFrame,
        user_features_df: tp.Optional[pd.DataFrame] = None,
        cat_user_features: tp.Iterable[str] = (),
        make_dense_user_features: bool = False,
        item_features_df: tp.Optional[pd.DataFrame] = None,
        cat_item_features: tp.Iterable[str] = (),
        make_dense_item_features: bool = False,
        keep_extra_cols: bool = False,
    ) -> "Dataset":
        """Class method for convenient `Dataset` creation.

        Use it to create dataset from raw data.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Table where every row contains user-item interaction and columns are:
                - `Columns.User` - user id;
                - `Columns.Item` - item id;
                - `Columns.Weight` - weight of interaction, `float`,
                  use ``1`` if interactions have no weight;
                - `Columns.Datetime` - timestamp of interactions,
                  assign random value if you're not going to use it later.
        user_features_df, item_features_df : pd.DataFrame, optional
            User (item) explicit features table.
            It will be used to create `SparseFeatures` using `from_flatten` class method
            or `DenseFeatures` using `from_dataframe` class method
            depending on `make_dense_user_features` (`make_dense_item_features`) flag.
            See detailed info about the table structure in these methods description.
        cat_user_features, cat_item_features : tp.Iterable[str], default ``()``
            List of categorical user (item) feature names for
            `SparseFeatures.from_flatten` method.
            Used only if `make_dense_user_features` (`make_dense_item_features`)
            flag is ``False`` and `user_features_df` (`item_features_df`) is not ``None``.
        make_dense_user_features, make_dense_item_features : bool, default ``False``
            Create user (item) features as dense or sparse.
            Used only if `user_features_df` (`item_features_df`) is not ``None``.
            - if ``False``, `SparseFeatures.from_flatten` method will be used;
            - if ``True``,  `DenseFeatures.from_dataframe` method will be used.
        keep_extra_cols: bool, default ``False``
            Flag to keep all columns from interactions besides the default ones.

        Returns
        -------
        Dataset
            Container with all input data, converted to `rectools` structures.
        """
        for col in (Columns.User, Columns.Item):
            if col not in interactions_df:
                raise KeyError(f"Column '{col}' must be present in `interactions_df`")
        user_id_map = IdMap.from_values(interactions_df[Columns.User].values)
        item_id_map = IdMap.from_values(interactions_df[Columns.Item].values)
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map, keep_extra_cols)

        user_features, user_id_map = cls._make_features(
            user_features_df,
            cat_user_features,
            make_dense_user_features,
            user_id_map,
            Columns.User,
            "user",
        )
        item_features, item_id_map = cls._make_features(
            item_features_df,
            cat_item_features,
            make_dense_item_features,
            item_id_map,
            Columns.Item,
            "item",
        )
        return cls(user_id_map, item_id_map, interactions, user_features, item_features)

    @staticmethod
    def _make_features(
        df: tp.Optional[pd.DataFrame],
        cat_features: tp.Iterable[str],
        make_dense: bool,
        id_map: IdMap,
        possible_id_col: str,
        feature_type: str,
    ) -> tp.Tuple[tp.Optional[Features], IdMap]:
        if df is None:
            return None, id_map

        id_col = possible_id_col if possible_id_col in df else "id"
        id_map = id_map.add_ids(df[id_col].values, raise_if_already_present=False)

        if make_dense:
            try:
                return DenseFeatures.from_dataframe(df, id_map, id_col=id_col), id_map
            except AbsentIdError:
                raise ValueError(
                    f"An error has occurred while constructing {feature_type} features: "
                    "When using dense features all ids from interactions must be present in features table"
                )
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"An error has occurred while constructing {feature_type} features: {e!r}")

        try:
            return SparseFeatures.from_flatten(df, id_map, cat_features, id_col=id_col), id_map
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"An error has occurred while constructing {feature_type} features: {e!r}")

    def get_user_item_matrix(
        self,
        include_weights: bool = True,
        include_warm_users: bool = False,
        include_warm_items: bool = False,
        dtype: tp.Type = np.float32,
    ) -> sparse.csr_matrix:
        """
        Construct user-item CSR matrix based on `interactions` attribute.

        Return a resized user-item matrix.
        Resizing is done using `user_id_map` and `item_id_map`,
        hence if either a user or an item is not presented in interactions,
        but presented in id map, then it's going to be in the returned matrix.

        Parameters
        ----------
        include_weights : bool, default ``True``
             Whether include interaction weights in matrix or not.
             If False, all values in returned matrix will be equal to ``1``.
        include_warm : bool, default ``False``
            Whether to include warm users and items into the matrix or not.
            Rows and columns for warm users and items will be added to the end of matrix,
            they will contain only zeros.

        Returns
        -------
        csr_matrix
            Resized user-item CSR matrix
        """
        matrix = self.interactions.get_user_item_matrix(include_weights, dtype)
        n_rows = self.user_id_map.size if include_warm_users else matrix.shape[0]
        n_columns = self.item_id_map.size if include_warm_items else matrix.shape[1]
        matrix.resize(n_rows, n_columns)
        return matrix

    def get_raw_interactions(
        self,
        include_weight: bool = True,
        include_datetime: bool = True,
        include_extra_cols: tp.Union[bool, tp.List[str]] = True,
    ) -> pd.DataFrame:
        """
        Return interactions as a `pd.DataFrame` object with replacing internal user and item ids to external ones.

        Parameters
        ----------
        include_weight : bool, default ``True``
            Whether to include weight column into resulting table or not.
        include_datetime : bool, default ``True``
            Whether to include datetime column into resulting table or not.
        include_extra_cols: bool, default ``True``
            Whether to include extra columns into resulting table or not.

        Returns
        -------
        pd.DataFrame
        """
        return self.interactions.to_external(
            self.user_id_map, self.item_id_map, include_weight, include_datetime, include_extra_cols
        )

    def filter_interactions(
        self,
        row_indexes_to_keep: np.ndarray,
        keep_external_ids: bool = True,
        keep_features_for_removed_entities: bool = True,
    ) -> "Dataset":
        """
        Generate filtered dataset that contains only provided `row_indexes_to_keep` from original
        dataset interactions dataframe.
        Resulting dataset will get new id mapping for both users and items.

        Parameters
        ----------
        row_indexes_to_keep : np.ndarray
            Original dataset interactions df row indexes that are to be kept
        keep_external_ids : bool, default `True`
            Whether to keep external ids -> 2x internal ids mapping (default).
            Otherwise internal -> 2x internal ids mapping will be created.
        keep_features_for_removed_entities : bool, default `True`
            Whether to keep all features for users and items that are not hot any more.

        Returns
        -------
        Dataset
            Filtered dataset that has only selected interactions, new ids mapping and processed features.
        """
        interactions_df = self.interactions.df.iloc[row_indexes_to_keep]

        # 1x internal -> 2x internal
        user_id_map = IdMap.from_values(interactions_df[Columns.User].values)
        item_id_map = IdMap.from_values(interactions_df[Columns.Item].values)
        # We shouldn't drop extra columns if they are present
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map, keep_extra_cols=True)

        def _handle_features(
            features: tp.Optional[Features], target_id_map: IdMap, dataset_id_map: IdMap
        ) -> tp.Tuple[tp.Optional[Features], IdMap]:
            if features is None:
                return None, target_id_map

            if keep_features_for_removed_entities:
                all_features_ids = np.arange(len(features))
                target_id_map = target_id_map.add_ids(all_features_ids, raise_if_already_present=False)

            needed_ids = target_id_map.get_external_sorted_by_internal()
            features = features.take(needed_ids)
            return features, target_id_map

        user_features_new, user_id_map = _handle_features(self.user_features, user_id_map, self.user_id_map)
        item_features_new, item_id_map = _handle_features(self.item_features, item_id_map, self.item_id_map)

        if keep_external_ids:  # external -> 2x internal
            user_id_map = IdMap(self.user_id_map.convert_to_external(user_id_map.external_ids))
            item_id_map = IdMap(self.item_id_map.convert_to_external(item_id_map.external_ids))

        filtered_dataset = Dataset(
            user_id_map=user_id_map,
            item_id_map=item_id_map,
            interactions=interactions,
            user_features=user_features_new,
            item_features=item_features_new,
        )
        return filtered_dataset
