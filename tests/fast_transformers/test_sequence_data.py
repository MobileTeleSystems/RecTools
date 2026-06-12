#  Copyright 2026 MTS (Mobile Telesystems)
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

"""Tests for vectorized sequence building and data utilities."""

import torch

from rectools.fast_transformers.preprocessing.sequence_data import (
    SequenceBatchDataset,
    align_embeddings,
    build_sequences,
)

DEVICE = "cpu"


class TestBuildSequences:
    """Tests for the build_sequences function."""

    def test_basic_two_users(self) -> None:
        """Two users with 3 interactions each, max_len=4."""
        user_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        item_ids = torch.tensor([10, 20, 30, 40, 50, 60])
        timestamps = torch.tensor([1, 2, 3, 4, 5, 6])

        x, y, _unique_items, result_users = build_sequences(
            user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device=DEVICE
        )

        assert x.shape == (2, 4)
        assert y.shape == (2, 4)

        # Items are mapped to internal 1-based IDs; 0 = padding
        # unique_items is sorted, so: [10, 20, 30, 40, 50, 60]
        # internal IDs: 10->1, 20->2, 30->3, 40->4, 50->5, 60->6

        # User 0: items [10, 20, 30] in order => internal [1, 2, 3]
        # x = [0, 1, 2] left-padded to len 4 => [0, 0, 1, 2]
        # y = [0, 2, 3] left-padded to len 4 => [0, 0, 2, 3]
        assert x[0].tolist() == [0, 0, 1, 2]
        assert y[0].tolist() == [0, 0, 2, 3]

        # User 1: items [40, 50, 60] in order => internal [4, 5, 6]
        # x = [0, 4, 5] => [0, 0, 4, 5]
        # y = [0, 5, 6] => [0, 0, 5, 6]
        assert x[1].tolist() == [0, 0, 4, 5]
        assert y[1].tolist() == [0, 0, 5, 6]

        assert result_users.tolist() == [0, 1]

    def test_unique_items_mapping(self) -> None:
        """unique_items should map internal_id - 1 => external_id."""
        user_ids = torch.tensor([0, 0, 0])
        item_ids = torch.tensor([100, 50, 200])
        timestamps = torch.tensor([1, 2, 3])

        _, _, unique_items, _ = build_sequences(
            user_ids, item_ids, timestamps, max_len=5, min_interactions=2, device=DEVICE
        )

        # torch.unique sorts, so unique_items = [50, 100, 200]
        assert unique_items.tolist() == [50, 100, 200]

    def test_min_interactions_filtering(self) -> None:
        """Users with fewer than min_interactions should be dropped."""
        user_ids = torch.tensor([0, 0, 0, 1, 2, 2])
        item_ids = torch.tensor([10, 20, 30, 40, 50, 60])
        timestamps = torch.tensor([1, 2, 3, 4, 5, 6])

        x, _y, _, result_users = build_sequences(
            user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device=DEVICE
        )

        # User 1 has only 1 interaction => dropped
        assert x.shape[0] == 2
        assert result_users.tolist() == [0, 2]

    def test_min_interactions_higher_threshold(self) -> None:
        """Higher min_interactions threshold filters more aggressively."""
        user_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        item_ids = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
        timestamps = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

        x, _y, _, result_users = build_sequences(
            user_ids, item_ids, timestamps, max_len=5, min_interactions=3, device=DEVICE
        )

        # User 0 has 3, User 1 has 2 (dropped), User 2 has 4
        assert x.shape[0] == 2
        assert result_users.tolist() == [0, 2]

    def test_all_users_filtered_out(self) -> None:
        """When all users have fewer than min_interactions, return empty tensors."""
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([10, 20, 30])
        timestamps = torch.tensor([1, 2, 3])

        x, y, _, result_users = build_sequences(
            user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device=DEVICE
        )

        assert x.shape == (0, 4)
        assert y.shape == (0, 4)
        assert len(result_users) == 0

    def test_max_len_truncation(self) -> None:
        """Sequences longer than max_len should be truncated, keeping the most recent items."""
        user_ids = torch.tensor([0, 0, 0, 0, 0])
        item_ids = torch.tensor([10, 20, 30, 40, 50])
        timestamps = torch.tensor([1, 2, 3, 4, 5])

        x, y, _, _ = build_sequences(user_ids, item_ids, timestamps, max_len=3, min_interactions=2, device=DEVICE)

        # 5 items total. capped_lens = min(5, 3+1) = 4, effective = 3
        # Sorted items: 10->1, 20->2, 30->3, 40->4, 50->5
        # last 4 items for x/y windowing: items at positions [1..4]
        # x takes [1,2,3] => internal [2,3,4]; y takes [2,3,4] => internal [3,4,5]
        assert x.shape == (1, 3)
        assert y.shape == (1, 3)
        assert x[0].tolist() == [2, 3, 4]
        assert y[0].tolist() == [3, 4, 5]

    def test_timestamp_ordering(self) -> None:
        """Items should be ordered by timestamp regardless of input order."""
        user_ids = torch.tensor([0, 0, 0])
        item_ids = torch.tensor([30, 10, 20])
        timestamps = torch.tensor([3, 1, 2])

        x, y, unique_items, _ = build_sequences(
            user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device=DEVICE
        )

        # unique_items (sorted by value): [10, 20, 30] => internal 1, 2, 3
        # By timestamp: 10(t=1), 20(t=2), 30(t=3) => internal [1, 2, 3]
        # x = [0, 0, 1, 2]
        # y = [0, 0, 2, 3]
        assert unique_items.tolist() == [10, 20, 30]
        assert x[0].tolist() == [0, 0, 1, 2]
        assert y[0].tolist() == [0, 0, 2, 3]

    def test_left_padding(self) -> None:
        """Sequences shorter than max_len should be left-padded with zeros."""
        user_ids = torch.tensor([0, 0])
        item_ids = torch.tensor([10, 20])
        timestamps = torch.tensor([1, 2])

        x, y, _, _ = build_sequences(user_ids, item_ids, timestamps, max_len=5, min_interactions=2, device=DEVICE)

        # 2 items => effective_len = 1 (capped_lens = 2, effective = 1)
        # x = [0, 0, 0, 0, 1], y = [0, 0, 0, 0, 2]
        assert x[0].tolist() == [0, 0, 0, 0, 1]
        assert y[0].tolist() == [0, 0, 0, 0, 2]

    def test_result_users_preserves_external_ids(self) -> None:
        """result_users should contain external user IDs, not internal indices."""
        user_ids = torch.tensor([100, 100, 100, 200, 200, 200])
        item_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        timestamps = torch.tensor([1, 2, 3, 4, 5, 6])

        _, _, _, result_users = build_sequences(
            user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device=DEVICE
        )

        assert result_users.tolist() == [100, 200]

    def test_shared_items_across_users(self) -> None:
        """Same items used by different users should share internal IDs."""
        user_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        item_ids = torch.tensor([10, 20, 30, 20, 30, 40])
        timestamps = torch.tensor([1, 2, 3, 4, 5, 6])

        x, y, unique_items, _ = build_sequences(
            user_ids, item_ids, timestamps, max_len=4, min_interactions=2, device=DEVICE
        )

        # unique_items: [10, 20, 30, 40] => internal 1, 2, 3, 4
        assert unique_items.tolist() == [10, 20, 30, 40]

        # User 0: 10(1), 20(2), 30(3) => x=[0, 1, 2], y=[0, 2, 3]
        assert x[0].tolist() == [0, 0, 1, 2]
        assert y[0].tolist() == [0, 0, 2, 3]

        # User 1: 20(2), 30(3), 40(4) => x=[0, 2, 3], y=[0, 3, 4]
        assert x[1].tolist() == [0, 0, 2, 3]
        assert y[1].tolist() == [0, 0, 3, 4]

    def test_output_device(self) -> None:
        """All output tensors should be on the specified device."""
        user_ids = torch.tensor([0, 0])
        item_ids = torch.tensor([1, 2])
        timestamps = torch.tensor([1, 2])

        x, y, unique_items, result_users = build_sequences(
            user_ids, item_ids, timestamps, max_len=3, min_interactions=2, device=DEVICE
        )

        assert x.device.type == DEVICE
        assert y.device.type == DEVICE
        assert unique_items.device.type == DEVICE
        assert result_users.device.type == DEVICE

    def test_output_dtypes(self) -> None:
        """Check x and y should be long tensors."""
        user_ids = torch.tensor([0, 0])
        item_ids = torch.tensor([1, 2])
        timestamps = torch.tensor([1, 2])

        x, y, _, _ = build_sequences(user_ids, item_ids, timestamps, max_len=3, min_interactions=2, device=DEVICE)

        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_exact_max_len_sequence(self) -> None:
        """Sequence with exactly max_len + 1 items should fill entire x and y."""
        user_ids = torch.tensor([0, 0, 0, 0])
        item_ids = torch.tensor([10, 20, 30, 40])
        timestamps = torch.tensor([1, 2, 3, 4])

        x, y, _, _ = build_sequences(user_ids, item_ids, timestamps, max_len=3, min_interactions=2, device=DEVICE)

        # 4 items, max_len=3 => capped_lens = min(4, 4) = 4, effective = 3
        # No padding needed
        assert 0 not in x[0].tolist()
        assert 0 not in y[0].tolist()

    def test_multiple_users_different_lengths(self) -> None:
        """Users with different sequence lengths should be properly handled."""
        user_ids = torch.tensor([0, 0, 1, 1, 1, 1])
        item_ids = torch.tensor([10, 20, 30, 40, 50, 60])
        timestamps = torch.tensor([1, 2, 3, 4, 5, 6])

        x, y, _unique_items, _ = build_sequences(
            user_ids, item_ids, timestamps, max_len=5, min_interactions=2, device=DEVICE
        )

        # unique_items: [10, 20, 30, 40, 50, 60] => internal 1..6
        # User 0: 2 items => effective=1
        # x[0] = [0, 0, 0, 0, 1], y[0] = [0, 0, 0, 0, 2]
        assert x[0].tolist() == [0, 0, 0, 0, 1]
        assert y[0].tolist() == [0, 0, 0, 0, 2]

        # User 1: 4 items => effective=3
        # x[1] = [0, 0, 3, 4, 5], y[1] = [0, 0, 4, 5, 6]
        assert x[1].tolist() == [0, 0, 3, 4, 5]
        assert y[1].tolist() == [0, 0, 4, 5, 6]


class TestAlignEmbeddings:
    """Tests for the align_embeddings function."""

    def test_2d_pretrained(self) -> None:
        """Align 2D pretrained embeddings to internal ID order."""
        pretrained = torch.tensor(
            [
                [1.0, 2.0],  # external item 0
                [3.0, 4.0],  # external item 1
                [5.0, 6.0],  # external item 2
                [7.0, 8.0],  # external item 3
            ]
        )
        # unique_items: external IDs that map to internal IDs 1, 2, 3
        unique_items = torch.tensor([2, 0, 3])
        n_items = 3

        aligned = align_embeddings(pretrained, unique_items, n_items)

        assert aligned.shape == (4, 2)  # n_items + 1
        # Row 0 (padding) should be zeros
        assert aligned[0].tolist() == [0.0, 0.0]
        # Internal ID 1 => external ID 2 => pretrained[2] = [5, 6]
        assert aligned[1].tolist() == [5.0, 6.0]
        # Internal ID 2 => external ID 0 => pretrained[0] = [1, 2]
        assert aligned[2].tolist() == [1.0, 2.0]
        # Internal ID 3 => external ID 3 => pretrained[3] = [7, 8]
        assert aligned[3].tolist() == [7.0, 8.0]

    def test_3d_pretrained(self) -> None:
        """Align 3D pretrained embeddings (multi-variant)."""
        pretrained = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # item 0, 2 variants
                [[5.0, 6.0], [7.0, 8.0]],  # item 1
            ]
        )
        unique_items = torch.tensor([1, 0])
        n_items = 2

        aligned = align_embeddings(pretrained, unique_items, n_items)

        assert aligned.shape == (3, 2, 2)  # (n_items+1, n_variants, dim)
        # Row 0 (padding) should be zeros
        torch.testing.assert_close(aligned[0], torch.zeros(2, 2))
        # Internal ID 1 => external ID 1
        torch.testing.assert_close(aligned[1], pretrained[1])
        # Internal ID 2 => external ID 0
        torch.testing.assert_close(aligned[2], pretrained[0])

    def test_padding_row_is_zero(self) -> None:
        """The first row (padding, internal ID 0) should always be zeros."""
        pretrained = torch.randn(10, 8)
        unique_items = torch.tensor([0, 1, 2])
        n_items = 3

        aligned = align_embeddings(pretrained, unique_items, n_items)

        torch.testing.assert_close(aligned[0], torch.zeros(8))

    def test_out_of_range_indices(self) -> None:
        """Items with external IDs outside pretrained range should get zero embeddings."""
        pretrained = torch.tensor(
            [
                [1.0, 2.0],  # external 0
                [3.0, 4.0],  # external 1
            ]
        )
        # External ID 5 is out of range (pretrained has only 2 rows)
        unique_items = torch.tensor([0, 5, 1])
        n_items = 3

        aligned = align_embeddings(pretrained, unique_items, n_items)

        assert aligned.shape == (4, 2)
        # Internal 1 => external 0 => valid
        assert aligned[1].tolist() == [1.0, 2.0]
        # Internal 2 => external 5 => out of range => zeros
        assert aligned[2].tolist() == [0.0, 0.0]
        # Internal 3 => external 1 => valid
        assert aligned[3].tolist() == [3.0, 4.0]

    def test_negative_indices_handled(self) -> None:
        """Negative external IDs should be treated as invalid and get zeros."""
        pretrained = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        unique_items = torch.tensor([-1, 0])
        n_items = 2

        aligned = align_embeddings(pretrained, unique_items, n_items)

        assert aligned.shape == (3, 2)
        # Internal 1 => external -1 => invalid => zeros
        assert aligned[1].tolist() == [0.0, 0.0]
        # Internal 2 => external 0 => valid
        assert aligned[2].tolist() == [1.0, 2.0]

    def test_output_shape_matches_n_items_plus_one(self) -> None:
        """Output shape should be (n_items + 1, D) regardless of unique_items length."""
        pretrained = torch.randn(20, 4)
        unique_items = torch.tensor([3, 7, 15])
        n_items = 3

        aligned = align_embeddings(pretrained, unique_items, n_items)

        assert aligned.shape == (4, 4)


class TestSequenceBatchDataset:
    """Tests for SequenceBatchDataset."""

    def test_length(self) -> None:
        x = torch.zeros(5, 3)
        y = torch.zeros(5, 3)
        ds = SequenceBatchDataset(x, y)
        assert len(ds) == 5

    def test_getitem_returns_dict(self) -> None:
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y = torch.tensor([[7, 8, 9], [10, 11, 12]])
        ds = SequenceBatchDataset(x, y)

        batch = ds[0]
        assert isinstance(batch, dict)
        assert "x" in batch
        assert "y" in batch
        assert batch["x"].tolist() == [1, 2, 3]
        assert batch["y"].tolist() == [7, 8, 9]

    def test_getitem_second_element(self) -> None:
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6], [7, 8]])
        ds = SequenceBatchDataset(x, y)

        batch = ds[1]
        assert batch["x"].tolist() == [3, 4]
        assert batch["y"].tolist() == [7, 8]

    def test_transform_applied(self) -> None:
        x = torch.tensor([[1, 2]])
        y = torch.tensor([[3, 4]])

        def double_x(batch: dict) -> dict:
            batch["x"] = batch["x"] * 2
            return batch

        ds = SequenceBatchDataset(x, y, transform=double_x)
        batch = ds[0]
        assert batch["x"].tolist() == [2, 4]
        assert batch["y"].tolist() == [3, 4]

    def test_no_transform(self) -> None:
        x = torch.tensor([[10, 20]])
        y = torch.tensor([[30, 40]])
        ds = SequenceBatchDataset(x, y, transform=None)

        batch = ds[0]
        assert batch["x"].tolist() == [10, 20]
        assert batch["y"].tolist() == [30, 40]
