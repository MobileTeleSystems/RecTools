"""Module providing DPP algorithm for recos"""
import typing as tp

import numpy as np
import pandas as pd

from rectools import InternalIds


def get_dpp_recos_for_single_user(
        feature_vectors: np.ndarray[np.dtype[float], np.dtype[float]],
        model_output: tp.Tuple[InternalIds, InternalIds, float],
        feed_size: int,
        epsilon: float = 1e-10,
) -> tp.Sequence:
    """
    Rearranges recommendation system output to increase its diversity

    Parameters
    ----------
    feature_vectors :
        Two-dimensional np.ndarray of ints
        sparse or dense representation of feature vectors
    model_output : np.ndarray with columns "user_id", "item_id" (InternalIds both), "Score"
        user-to-item recommendation from any RecTools model
    feed_size : int, self.Dataset.shape[0] as Default
        Size of items to be recommended for user
    epsilon: float=1E-10
        tolerance threshold for greedy algorithm.

    Returns
    -------
    np.ndarray - 1-dimensional array
    rearranged recommendations for single user

    """
    kernel_matrix = build_kernel(feature_vectors, model_output)

    item_size = kernel_matrix.shape[0]
    cis = np.zeros((feed_size, item_size))
    di2s = np.copy(np.diag(kernel_matrix))

    selected_items = []
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)

    while len(selected_items) < feed_size and di2s[selected_item] > epsilon:
        window_size = len(selected_items) - 1
        ci_optimal = cis[:window_size, selected_item]
        di_optimal = np.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:window_size, :])) / di_optimal
        cis[window_size, :] = np.copy(eis)
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] > epsilon:
            selected_items.append(selected_item)
    return selected_items


def get_dpp_recos_for_several_users(
        feature_vectors: np.ndarray[(float, float)],
        model_output: tp.Tuple[InternalIds, InternalIds, float],
        user_group: InternalIds,
        feed_size: int,
) -> pd.DataFrame:
    """
    Rearranges recommendation system output for group of users.

    Parameters
    ----------
    feature_vectors :
        Two-dimensional np.ndarray of ints
        sparse representation of feature_vectors for each item. WARNING! Strings should be sorted in
        ascending order by Internal item Ids
    model_output : pd.DataFrame with columns "user_id", "item_id" (InternalIds both), "Score"
        user-to-item recommendation from any RecTools model
    user_group
        user ids in Internal notation
    feed_size :
        int, self.Dataset.shape[0] as Default
        Size of items to be recommended for every user

    Returns
    -------
    tp.Tuple[tp.Sequence, tp.Sequence] (pd.DataFrame)
    rearranged recommendations for single user, 1st column is User_id, 2nd column is Item_id

    """
    model_output = pd.DataFrame(model_output)

    final_user_ids = []
    final_recos = []
    for user in user_group:
        personal_recos = model_output.query["user_id = user"]
        sorted_recos = get_dpp_recos_for_single_user(
            feature_vectors, personal_recos, feed_size
        )
        sorted_recos = np.ndarray(sorted_recos)
        final_user_ids.extend([user] * sorted_recos.shape[0])
        final_recos.append(sorted_recos)

    all_dpp_ids = np.concatenate(final_user_ids)
    dpp_recos = pd.DataFrame({"user_id": final_user_ids, "item_id": all_dpp_ids})
    return dpp_recos


def build_kernel(
        feature_vectors: np.ndarray[(float, float)],
        model_output: tp.Tuple[InternalIds, InternalIds, float],
) -> np.ndarray:
    """
    Builds kernel matrix for dpp algorithm.

    Parameters
    ----------
    feature_vectors:
        Two-dimensional np.ndarray of ints
        sparse representation of feature_vectors for each item. WARNING! strings should be sorted
        in ascending order by item InternaId
    model_output: np.ndarray with columns "user_id", "item_id" (InternalIds both), "Score"
        user-to-item recommendation from any RecTools model

    Returns
    -----------
    two-dimensional np.ndarray of ints
    """
    model_output = pd.DataFrame(model_output)
    relevant_item_ids = model_output.item_id.unique()  # collect unique InternalIdx ids
    feature_vectors = feature_vectors[relevant_item_ids, :]

    scores = np.ndarray(model_output.Score)
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    similarities = np.dot(feature_vectors, feature_vectors.T)
    kernel_matrix = (
            scores.reshape((feature_vectors.shape[0], 1))
            * similarities
            * scores.reshape((1, feature_vectors.shape[0]))
    )
    return kernel_matrix
l
