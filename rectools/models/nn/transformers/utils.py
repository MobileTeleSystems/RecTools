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

import typing as tp

import numpy as np
import pandas as pd

from rectools import Columns, ExternalIds


def leave_one_out_mask(interactions: pd.DataFrame, val_users: tp.Union[ExternalIds, int, None] = None) -> np.ndarray:
    """
    Create a boolean mask for leave-one-out validation by selecting the last interaction per user.

    Identifies the most recent interaction for specified validation users based on timestamp ranking.
    Users can be filtered using `val_users` parameter which supports slicing or explicit user IDs.

    Parameters
    ----------
    interactions : pd.DataFrame
        User-item interactions data with at least three columns:
        Columns.User, Columns.Item and Columns.Datetime
    val_users : Optional[Union[ExternalIds, int]], default ``None``
        Validation user filter. Can be:
        - None: use all users
        - int:  randomly sample N users from unique user list without replacement
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
    if isinstance(val_users, int):
        users = interactions[Columns.User].unique()
        val_users = np.random.choice(users, size=val_users, replace=False)
    elif val_users is None:
        return last_interact_mask.values

    mask = interactions[Columns.User].isin(val_users) & last_interact_mask
    return mask.values
