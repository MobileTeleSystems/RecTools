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
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pytest_subtests import SubTests
from scipy import sparse

from rectools import Columns
from rectools.dataset import IdMap, Interactions
from tests.testing_utils import assert_sparse_matrix_equal


class TestInteractions:
    def setup(self) -> None:
        self.df = pd.DataFrame(
            {
                Columns.User: [1, 2, 1, 1],
                Columns.Item: [0, 1, 0, 1],
                Columns.Weight: [5, 7.0, 4, 1],
                Columns.Datetime: [datetime(2021, 9, 8)] * 4,
            }
        )

    def test_creation(self) -> None:
        interactions = Interactions(self.df)
        pd.testing.assert_frame_equal(interactions.df, self.df)

    def test_missing_columns_validation(self, subtests: SubTests) -> None:
        for col in self.df.columns:
            with subtests.test(f"drop {col} column"):
                with pytest.raises(KeyError):
                    Interactions(self.df.drop(columns=col))

    @pytest.mark.parametrize("column", (Columns.User, Columns.Item))
    def test_types_validation(self, column: str) -> None:
        with pytest.raises(TypeError):
            Interactions(self.df.astype({column: float}))

    @pytest.mark.parametrize("column", (Columns.User, Columns.Item))
    def test_positivity_validation(self, column: str) -> None:
        with pytest.raises(ValueError):
            self.df.at[0, column] = -1
            Interactions(self.df)

    def test_from_raw_creation(self) -> None:
        raw_df = pd.DataFrame(
            {
                Columns.User: ["u1", "u2", "u1", "u1"],
                Columns.Item: ["i1", "i2", "i1", "i2"],
                Columns.Weight: [5, 7, 4, 1],
                Columns.Datetime: ["2021-09-08"] * 4,
            }
        )
        user_id_map = IdMap(np.array(["u0", "u1", "u2"]))
        item_id_map = IdMap.from_values(["i1", "i2"])
        interactions = Interactions.from_raw(raw_df, user_id_map, item_id_map)
        pd.testing.assert_frame_equal(interactions.df, self.df)

    @pytest.mark.parametrize(
        "with_weights,expected_data",
        (
            (False, [1, 1, 1, 1]),
            (True, [5, 7, 4, 1]),
        ),
    )
    def test_getting_user_item_matrix(self, with_weights: bool, expected_data: tp.List[float]) -> None:
        interactions = Interactions(self.df)
        matrix = interactions.get_user_item_matrix(with_weights)
        expected = sparse.csr_matrix((expected_data, (self.df[Columns.User].values, self.df[Columns.Item].values)))
        assert_sparse_matrix_equal(matrix, expected)

    def test_raises_when_weight_not_numeric(self) -> None:
        df = self.df
        df.loc[1, Columns.Weight] = "w"
        with pytest.raises(TypeError) as e:
            Interactions.from_raw(df, IdMap.from_values(df[Columns.User]), IdMap.from_values(df[Columns.Item]))
        err_text = e.value.args[0]
        assert Columns.Weight in err_text.lower()

    def test_raises_when_datetime_type_incorrect(self) -> None:
        df = self.df
        df.loc[1, Columns.Datetime] = "dt"
        with pytest.raises(TypeError) as e:
            Interactions.from_raw(df, IdMap.from_values(df[Columns.User]), IdMap.from_values(df[Columns.Item]))
        err_text = e.value.args[0]
        assert Columns.Datetime in err_text.lower()
