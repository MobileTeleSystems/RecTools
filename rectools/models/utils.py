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

"""Useful functions."""

import typing as tp

import numpy as np
from scipy import sparse

from rectools.utils import fast_isin_for_sorted_test_elements


def get_viewed_item_ids(user_items: sparse.csr_matrix, user_id: int) -> np.ndarray:
    """
    Return indices of items that user has interacted with.

    Parameters
    ----------
    user_items : csr_matrix
        Matrix of interactions.
    user_id : int
        Internal user id.

    Returns
    -------
    np.ndarray
        Internal item indices that user has interacted with.
    """
    return user_items.indices[user_items.indptr[user_id] : user_items.indptr[user_id + 1]]


def recommend_from_scores(
    scores: np.ndarray,
    k: int,
    sorted_blacklist: tp.Optional[np.ndarray] = None,
    sorted_whitelist: tp.Optional[np.ndarray] = None,
    ascending: bool = False,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Prepare top-k recommendations for a user.

    Recommendations are sorted by item scores for this particular user.
    Recommendations can be filtered according to whitelist and blacklist.

    If `I` - set of all items, `B` - set of blacklist items, `W` - set of whitelist items, then:
        - if `W` is ``None``, then for recommendations will be used `I - B` set of items
        - if `W` is not ``None``, then for recommendations will be used `W - B` set of items

    Parameters
    ----------
    scores : np.ndarray
        Array of floats. Scores of relevance of all items for this user. Shape ``(n_items,)``.
    k : int
        Desired number of final recommendations.
        If, after applying white- and blacklist, number of available items `n_available` is less than `k`,
        then `n_available` items will be returned without warning.
    sorted_blacklist : np.ndarray, optional, default ``None``
        Array of unique ints. Sorted inner item ids to exclude from recommendations.
    sorted_whitelist : np.ndarray, optional, default ``None``
        Array of unique ints. Sorted inner item ids to use in recommendations.
    ascending : bool, default False
        If False, sorting by descending of score, use when score are metric of similarity.
        If True, sorting by ascending of score, use when score are distance.

    Returns
    -------
    np.ndarray
        Array of recommended items, sorted by score descending.
    """
    if k <= 0:
        raise ValueError("`k` must be positive")

    items_to_recommend = None

    if sorted_blacklist is not None:
        if sorted_whitelist is None:
            sorted_whitelist = np.arange(scores.size)
        items_to_recommend = sorted_whitelist[~fast_isin_for_sorted_test_elements(sorted_whitelist, sorted_blacklist)]
    elif sorted_whitelist is not None:
        items_to_recommend = sorted_whitelist

    if items_to_recommend is not None:
        scores = scores[items_to_recommend]

    if ascending:
        scores = -scores

    n_reco = min(k, scores.size)
    unsorted_reco_positions = scores.argpartition(-n_reco)[-n_reco:]
    unsorted_reco_scores = scores[unsorted_reco_positions]
    sorted_reco_positions = unsorted_reco_positions[unsorted_reco_scores.argsort()[::-1]]

    if items_to_recommend is not None:
        reco_ids = items_to_recommend[sorted_reco_positions]
    else:
        reco_ids = sorted_reco_positions
    reco_scores = scores[sorted_reco_positions]

    if ascending:
        reco_scores = -reco_scores

    return reco_ids, reco_scores
