#  Copyright 2022 MTS (Mobile Telesystems)
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

import numpy as np
from scipy import sparse

from rectools.utils import isin_2d_int


def get_not_seen_mask(
    train_users: np.ndarray,
    train_items: np.ndarray,
    test_users: np.ndarray,
    test_items: np.ndarray,
) -> np.ndarray:
    """
    Return mask for test interactions that is not in train interactions.

    Parameters
    ----------
    train_users : np.ndarray
        Integer array of users in train interactions (it's not a unique users!).
    train_items : np.ndarray
        Integer array of items in train interactions. Has same length as `train_users`.
    test_users : np.ndarray
        Integer array of users in test interactions (it's not a unique users!).
    test_items : np.ndarray
        Integer array of items in test interactions. Has same length as `test_users`.

    Returns
    -------
    np.ndarray
        Boolean mask of same length as `test_users` (`test_items`).
        ``True`` means interaction not present in train.
    """
    if train_users.size != train_items.size:
        raise ValueError("Lengths of `train_users` and `train_items` must be the same")
    if test_users.size != test_items.size:
        raise ValueError("Lengths of `test_users` and `test_items` must be the same")

    if train_users.size == 0:
        return np.ones(test_users.size, dtype=bool)
    if test_users.size == 0:
        return np.array([], dtype=bool)

    n_users = max(train_users.max(), test_users.max()) + 1
    n_items = max(train_items.max(), test_items.max()) + 1

    cls = sparse.csr_matrix if n_users < n_items else sparse.csc_matrix

    def make_matrix(users: np.ndarray, items: np.ndarray) -> sparse.spmatrix:
        return cls((np.ones(len(users), dtype=bool), (users, items)), shape=(n_users, n_items))

    train_mat = make_matrix(train_users, train_items)
    test_mat = make_matrix(test_users, test_items)

    already_seen_coo = test_mat.multiply(train_mat).tocoo(copy=False)
    del train_mat, test_mat
    already_seen_arr = np.vstack((already_seen_coo.row, already_seen_coo.col)).T.astype(test_users.dtype)
    del already_seen_coo

    test_ui = np.vstack((test_users, test_items)).T
    not_seen_mask = isin_2d_int(test_ui, already_seen_arr, invert=True)
    return not_seen_mask
