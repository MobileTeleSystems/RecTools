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

import torch

from rectools.semantic.tokenizer.loss import CodebookLoss


class TestCodebookLoss:
    def test_zero_when_equal(self) -> None:
        loss_fn = CodebookLoss()
        x = torch.randn(4, 8)
        result = loss_fn(x, x.clone())
        assert result.item() == 0.0

    def test_positive_when_different(self) -> None:
        loss_fn = CodebookLoss()
        y_true = torch.randn(4, 8)
        y_pred = torch.randn(4, 8)
        result = loss_fn(y_true, y_pred)
        assert result.item() > 0.0

    def test_custom_coefficients(self) -> None:
        loss_default = CodebookLoss(mu=1.0, beta=0.25)
        loss_custom = CodebookLoss(mu=2.0, beta=0.5)
        y_true = torch.randn(4, 8)
        y_pred = torch.randn(4, 8)
        r_default = loss_default(y_true, y_pred)
        r_custom = loss_custom(y_true, y_pred)
        # Custom has doubled coefficients, so loss should roughly double
        assert r_custom.item() > r_default.item()

    def test_output_is_scalar(self) -> None:
        loss_fn = CodebookLoss()
        result = loss_fn(torch.randn(4, 8), torch.randn(4, 8))
        assert result.dim() == 0
