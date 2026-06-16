import pandas as pd


def k_core(
    interactions: pd.DataFrame,
    user_min_interactions: int = 5,
    item_min_interactions: int = 5,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Filter user and item interactions using iterative k-core algorithm.

    Removes users with fewer than ``user_min_interactions`` interactions
    and items with fewer than ``item_min_interactions`` interactions,
    repeating until no more removals are needed.

    Parameters
    ----------
    interactions : pd.DataFrame
        Interaction history. Must contain ``user_col`` and ``item_col`` columns.
    user_min_interactions : int
        Minimum number of interactions per user.
    item_min_interactions : int
        Minimum number of interactions per item.
    user_col : str
        Name of the user ID column.
    item_col : str
        Name of the item ID column.

    Returns
    -------
    pd.DataFrame
        Filtered interactions.
    """
    df = interactions
    while True:
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()
        bad_users = user_counts[user_counts < user_min_interactions].index
        bad_items = item_counts[item_counts < item_min_interactions].index

        if len(bad_users) == 0 and len(bad_items) == 0:
            break

        df = df[~df[user_col].isin(bad_users) & ~df[item_col].isin(bad_items)]
        df = df.reset_index(drop=True)

    return df
