import pandas as pd
import pytest

from rectools import Columns
from rectools.models.rerank import PerUserNegativeSampler


class TestPerUserNegativeSampler:
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        data = {
            Columns.User: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            Columns.Item: [101, 102, 103, 104, 201, 202, 203, 204, 301, 302, 303, 304],
            Columns.Score: [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6],
            Columns.Rank: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            Columns.Target: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
        return pd.DataFrame(data)

    @pytest.mark.parametrize("num_neg_samples", (1, 2))
    def test_sample_negatives(self, sample_data: pd.DataFrame, num_neg_samples: int) -> None:
        sampler = PerUserNegativeSampler(num_neg_samples=num_neg_samples, random_state=42)
        sampled_df = sampler.sample_negatives(sample_data)

        # Check if the resulting DataFrame has the correct columns
        assert set(sampled_df.columns) == set(sample_data.columns)

        # Check if the number of negatives per user is correct
        for user_id in sampled_df[Columns.User].unique():
            user_data = sampled_df[sampled_df[Columns.User] == user_id]
            num_negatives = len(user_data[user_data[Columns.Target] == 0])
            assert num_negatives == num_neg_samples

        # Check if positives were not changed
        pd.testing.assert_frame_equal(
            sampled_df[sampled_df[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
            sample_data[sample_data[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
        )

    def test_sample_negatives_with_insufficient_negatives(self, sample_data: pd.DataFrame) -> None:
        # Modify sample_data to have insufficient negatives for user 1
        sample_data.loc[sample_data[Columns.User] == 1, Columns.Target] = [1, 0, 1, 0]

        sampler = PerUserNegativeSampler(num_neg_samples=3, random_state=42)
        sampled_df = sampler.sample_negatives(sample_data)

        # Check if the resulting DataFrame has the correct columns
        assert set(sampled_df.columns) == set(sample_data.columns)

        # Check if the number of negatives per user is correct
        for user_id in sampled_df[Columns.User].unique():
            user_data = sampled_df[sampled_df[Columns.User] == user_id]
            num_negatives = len(user_data[user_data[Columns.Target] == 0])
            if user_id == 1:
                assert num_negatives == 2  # Only 2 negatives are available
            else:
                assert num_negatives == 3

        # Check if positives were not changed
        pd.testing.assert_frame_equal(
            sampled_df[sampled_df[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True),
            sample_data[sample_data[Columns.Target] == 1].sort_values(Columns.UserItem).reset_index(drop=True)
        )
