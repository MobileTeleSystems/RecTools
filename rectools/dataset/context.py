import pandas as pd

from rectools import Columns


def get_context(df_to_context: pd.DataFrame) -> pd.DataFrame:
    """
    Extract initial interaction context for each user.

    For each user, finds the earliest index base on datetime and uses it to define
    the initial contextual data. If the item column is present, it is dropped from the result,
    as it's not part of the user context.

    Parameters
    ----------
    df_to_context : pd.DataFrame
        Input DataFrame containing user interactions with at least
        user ID and datetime columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per user, representing the earliest
        context data for that user.
    """
    earliest = df_to_context.groupby(Columns.User)[Columns.Datetime].idxmin()
    context = df_to_context.loc[earliest]
    if Columns.Item in context:
        context.drop(columns=[Columns.Item], inplace=True)
    return context
