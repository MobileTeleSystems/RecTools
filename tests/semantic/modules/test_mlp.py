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

import pytest
import torch
from pytorch_lightning import seed_everything

from rectools.semantic.modules.mlp import MLP


class TestMLP:
    def setup_method(self) -> None:
        seed_everything(42, workers=True)

    def test_output_shape(self) -> None:
        mlp = MLP(input_dim=16, hidden_dims=[8], out_dim=4)
        x = torch.randn(5, 16)
        out = mlp(x)
        assert out.shape == (5, 4)

    def test_no_hidden_dims(self) -> None:
        mlp = MLP(input_dim=16, hidden_dims=[], out_dim=4)
        x = torch.randn(5, 16)
        out = mlp(x)
        assert out.shape == (5, 4)

    def test_multiple_hidden_dims(self) -> None:
        mlp = MLP(input_dim=32, hidden_dims=[16, 8], out_dim=4)
        x = torch.randn(3, 32)
        out = mlp(x)
        assert out.shape == (3, 4)

    def test_with_dropout(self) -> None:
        mlp = MLP(input_dim=16, hidden_dims=[8], out_dim=4, dropout=0.5)
        x = torch.randn(5, 16)
        out = mlp(x)
        assert out.shape == (5, 4)

    def test_with_normalize(self) -> None:
        mlp = MLP(input_dim=16, hidden_dims=[8], out_dim=4, normalize=True)
        x = torch.randn(5, 16)
        out = mlp(x)
        assert out.shape == (5, 4)

    def test_invalid_input_dim_raises(self) -> None:
        mlp = MLP(input_dim=16, hidden_dims=[8], out_dim=4)
        with pytest.raises(AssertionError, match="Invalid input dim"):
            mlp(torch.randn(5, 32))
