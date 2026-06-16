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

from rectools.semantic.tiger.loss import compute_tiger_loss


class TestComputeTigerLoss:
    def test_correct_predictions_low_loss(self) -> None:
        # Create logits that strongly predict the correct labels
        batch_size = 4
        codebook_size = 8
        sid_len = 3
        labels = torch.randint(0, codebook_size, (batch_size, sid_len))
        logits = []
        for d in range(sid_len):
            logit = torch.full((batch_size, codebook_size), -10.0)
            for i in range(batch_size):
                logit[i, labels[i, d]] = 10.0
            logits.append(logit)
        loss = compute_tiger_loss(logits, labels)
        assert loss.item() < 0.1

    def test_random_predictions_higher_loss(self) -> None:
        batch_size = 4
        codebook_size = 8
        sid_len = 3
        labels = torch.randint(0, codebook_size, (batch_size, sid_len))
        logits = [torch.randn(batch_size, codebook_size) for _ in range(sid_len)]
        loss = compute_tiger_loss(logits, labels)
        assert loss.item() > 0.1

    def test_output_is_scalar(self) -> None:
        logits = [torch.randn(2, 4), torch.randn(2, 4)]
        labels = torch.randint(0, 4, (2, 2))
        loss = compute_tiger_loss(logits, labels)
        assert loss.dim() == 0

    def test_ignores_minus_100_labels(self) -> None:
        logits = [torch.randn(2, 4)]
        labels = torch.tensor([[-100], [1]])
        loss = compute_tiger_loss(logits, labels)
        assert torch.isfinite(loss)
