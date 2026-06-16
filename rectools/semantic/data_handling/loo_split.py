import typing as tp

import pandas as pd


def loo_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-one-out split on an interactions DataFrame.

    For each user, the last interaction goes to test, the second-to-last
    to validation, and all earlier interactions to train.  Users with
    fewer than 3 interactions are dropped.

    Parameters
    ----------
    interactions : pd.DataFrame
        Interaction history.
    user_col : str
        Name of the user ID column.
    timestamp_col : str
        Name of the timestamp column. If present, interactions are
        sorted by (user_col, timestamp_col) before splitting.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Train contains all but the last 2 interactions per user.
        Val contains all but the last interaction per user.
        Test contains all interactions.
    """
    df = interactions
    if timestamp_col in df.columns:
        df = df.sort_values([user_col, timestamp_col])

    grouped = df.groupby(user_col)
    valid_users = grouped.filter(lambda x: len(x) >= 3)
    grouped = valid_users.groupby(user_col)

    cumcounts = grouped.cumcount(ascending=False)
    train_df = valid_users[cumcounts >= 2].reset_index(drop=True)
    val_df = valid_users[cumcounts >= 1].reset_index(drop=True)
    test_df = valid_users.reset_index(drop=True)

    return train_df, val_df, test_df
