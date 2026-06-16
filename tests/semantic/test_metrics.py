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

from rectools.semantic.metrics import coverage_k, gini_k


class TestCoverageK:
    def test_all_items_covered(self) -> None:
        all_topk = [[1, 2, 3], [4, 5]]
        result = coverage_k(all_topk, k=3, num_items=5)
        assert result == 1.0

    def test_partial_coverage(self) -> None:
        all_topk = [[1, 2], [1, 3]]
        result = coverage_k(all_topk, k=2, num_items=5)
        assert result == pytest.approx(3 / 5)

    def test_k_limits_items(self) -> None:
        all_topk = [[1, 2, 3, 4, 5]]
        result = coverage_k(all_topk, k=2, num_items=5)
        assert result == pytest.approx(2 / 5)

    def test_empty_lists(self) -> None:
        result = coverage_k([], k=5, num_items=10)
        assert result == 0.0

    def test_duplicate_items_across_users(self) -> None:
        all_topk = [[1, 2], [2, 3], [3, 1]]
        result = coverage_k(all_topk, k=2, num_items=3)
        assert result == 1.0


class TestGiniK:
    def test_uniform_distribution(self) -> None:
        all_topk = [[1], [2], [3], [4]]
        result = gini_k(all_topk, k=1)
        assert result == pytest.approx(0.0)

    def test_concentrated_distribution(self) -> None:
        all_topk = [[1], [1], [1], [1]]
        result = gini_k(all_topk, k=1)
        assert result == pytest.approx(0.0)

    def test_empty_input(self) -> None:
        result = gini_k([], k=5)
        assert result == 0.0

    def test_k_limits_items(self) -> None:
        all_topk = [[1, 99], [2, 99]]
        result_k1 = gini_k(all_topk, k=1)
        result_k2 = gini_k(all_topk, k=2)
        # With k=1 we only see items 1, 2 (uniform) -> gini=0
        assert result_k1 == pytest.approx(0.0)
        # With k=2, item 99 appears twice but 1 and 2 appear once each
        assert result_k2 > 0.0

    def test_skewed_distribution(self) -> None:
        # Item 1 recommended 3 times, item 2 once -> should be positive gini
        all_topk = [[1], [1], [1], [2]]
        result = gini_k(all_topk, k=1)
        assert result > 0.0
        assert result < 1.0
