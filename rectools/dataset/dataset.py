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

"""Dataset - all data container."""

import typing as tp

import attr
import pandas as pd
from scipy import sparse

from rectools import Columns

from .features import AbsentIdError, DenseFeatures, Features, SparseFeatures, UnknownIdError
from .identifiers import IdMap
from .interactions import Interactions


@attr.s(slots=True, frozen=True)
class Dataset:
    """
    Container class for all data for a recommendation model.

    It stores data about internal-external id mapping,
    user-item interactions, user and item features
    in special `rectools` structures for convenient future usage.

    This is data class, so you can create it explicitly, but
    it's recommended to use `construct` method.

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
        interactions = Interactions.from_raw(interactions_df, user_id_map, item_id_map)
        user_features = cls._make_features(
            user_features_df,
            cat_user_features,
            make_dense_user_features,
            user_id_map,
            Columns.User,
            "user",
        )
        item_features = cls._make_features(
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
    ) -> tp.Optional[Features]:
        if df is None:
            return None

        id_col = possible_id_col if possible_id_col in df else "id"

        if make_dense:
            try:
                return DenseFeatures.from_dataframe(df, id_map, id_col=id_col)
            except UnknownIdError:
                raise ValueError(f"Some ids from {feature_type} features table not present in interactions")
            except AbsentIdError:
                raise ValueError(
                    f"An error has occurred while constructing {feature_type} features: "
                    "When using dense features all ids from interactions must present in features table"
                )
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"An error has occurred while constructing {feature_type} features: {e!r}")
        try:
            return SparseFeatures.from_flatten(df, id_map, cat_features, id_col=id_col)
        except UnknownIdError:
            raise ValueError(f"Some ids from {feature_type} features table not present in interactions")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"An error has occurred while constructing {feature_type} features: {e!r}")

    def get_user_item_matrix(self, include_weights: bool = True) -> sparse.csr_matrix:
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

        Returns
        -------
        csr_matrix
            Resized user-item CSR matrix
        """
        matrix = self.interactions.get_user_item_matrix(include_weights)
        matrix.resize(self.user_id_map.internal_ids.size, self.item_id_map.internal_ids.size)
        return matrix

    def get_raw_interactions(self, include_weight: bool = True, include_datetime: bool = True) -> pd.DataFrame:
        """
        Return iteractions as a `pd.DataFrame` object with replacing internal user and item ids to external ones.

        Parameters
        ----------
        include_weight : bool, default ``True``
            Whether to include weight column into resulting table or not.
        include_datetime : bool, default ``True``
            Whether to include datetime column into resulting table or not.

        Returns
        -------
        pd.DataFrame
        """
        return self.interactions.to_external(self.user_id_map, self.item_id_map, include_weight, include_datetime)
