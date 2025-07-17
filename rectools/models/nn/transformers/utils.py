import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns, ExternalIds


def leave_one_out_mask(
    interactions: pd.DataFrame, val_users: tp.Optional[tp.Union[ExternalIds, int]] = None
) -> np.ndarray:
    """
    Create a boolean mask for leave-one-out validation by selecting the last interaction per user.

    Identifies the most recent interaction for specified validation users based on timestamp ranking.
    Users can be filtered using `val_users` parameter which supports slicing or explicit user IDs.

    Parameters
    ----------
    interactions : pd.DataFrame
        User-item interaction data with at least two columns:
    val_users : Optional[Union[ExternalIds, int]]
        Validation user filter. Can be:
        - None: use all users
        - int: take first N users from unique user list
        - array-like: explicit list of user IDs to include

    Returns
    -------
    np.ndarray
        Boolean array where True indicates the interaction is the last one for its user
        in the validation set.
    """
    groups = interactions.groupby(Columns.User)
    time_order = groups[Columns.Datetime].rank(method="first", ascending=True).astype(int)
    n_interactions = groups.transform("size").astype(int)
    inv_ranks = n_interactions - time_order
    last_interact_mask = inv_ranks == 0
    users = interactions[Columns.User].unique()
    if isinstance(val_users, int):
        val_users = users[:val_users]
    elif val_users is None:
        val_users = users

    mask = (interactions[Columns.User].isin(val_users)) & last_interact_mask
    return mask.values
