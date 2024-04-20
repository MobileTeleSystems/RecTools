import typing as tp

import attr
import numpy as np
import pandas as pd

from rectools import Columns

from .base import Catalog, MetricAtK


@attr.s
class LAUC(MetricAtK):
    """
    ROC-curve is a graph showing the performance of a classification model at all classification thresholds
    AUC is an area under the roc-curve.
    See more: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    LAUC is limited AUC. It is limited to first k recommendations and then interpolates to (1, 1).
    See more: https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf

    Parameters
    ----------
    k : int
        Number of items in top of recommendations list that will be used to calculate metric.

    """

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> float:
        """
        AUC is an area under the roc-curve.
        LAUC is limited AUC. See more: https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf
        This method calculates auc for all users
        Note that for users without interactions calc_for_user is NaN

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions, catalog)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> pd.Series:
        """
        Calculate LAUC for every user using trapezoidal rule
        Note that for users without interactions calc_per_user is NaN

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        MetricAtK._check(reco, interactions=interactions)
        if interactions.empty:
            return pd.Series(index=pd.Series(name=Columns.User, dtype=int), dtype=np.float64)
        reco_k_first_ranks = reco[reco[Columns.Rank] <= self.k]

        # list of users, who interacted during interactions time
        interacted_users = interactions[Columns.User].unique()
        reco_for_interacted_users = reco_k_first_ranks[reco_k_first_ranks[Columns.User].isin(interacted_users)]

        auc_per_user = self._calc_auc_for_interacted(reco_for_interacted_users, interactions, len(catalog))
        return auc_per_user

    def _calc_auc_for_interacted(
        self, reco_for_interacted_users: pd.DataFrame, interactions: pd.DataFrame, catalog_len: int
    ) -> pd.Series:
        """Calculate auc for users, who interacted with items during interactions time"""
        # make interactions, which were predicted right
        reco_true_interactions = reco_for_interacted_users.merge(
            interactions, left_on=[Columns.User, Columns.Item], right_on=[Columns.User, Columns.Item], how="inner"
        )

        # make pd.Series with index: user_id, value: np.array of ranks of reco_true_interactions
        users_true_interactions = (
            reco_true_interactions[[Columns.User, Columns.Rank]].groupby(Columns.User)[Columns.Rank].apply(np.array)
        )

        auc, fn_all = self._calc_auc_for_true_interacted(users_true_interactions, interactions, catalog_len)
        auc_series = pd.Series(index=users_true_interactions.index, data=auc)
        fn_for_users_with_all_fps = fn_all.drop(labels=users_true_interactions.index)
        # For users with no true positives auc is the square of the last interpolated triangle
        auc_for_users_with_all_fps = (1 - self.k / (catalog_len - fn_for_users_with_all_fps)) / 2
        auc_with_users_with_all_fps = pd.concat([auc_series, auc_for_users_with_all_fps])
        return auc_with_users_with_all_fps

    def _calc_auc_for_true_interacted(
        self, users_true_interactions: pd.DataFrame, interactions: pd.DataFrame, catalog_len: int
    ) -> tp.Tuple[np.ndarray, pd.Series]:
        # can't stack because sizes are different
        users_tps_series = users_true_interactions.apply(lambda x: put_copy(np.zeros(self.k + 1), x, 1))
        # can stack now
        users_tps = np.stack(users_tps_series.values)
        users_tps = np.cumsum(users_tps, axis=1)
        users_fps = np.arange(self.k + 1) - users_tps

        _tp = users_tps[:, -1]
        _fp = users_fps[:, -1]

        interactions_count_series = (
            interactions.groupby(Columns.User)[Columns.Item].count().rename("interactions_count")
        )
        tp_series = pd.Series(index=users_true_interactions.index, data=_tp, name="tps")

        # fn_all counts fn for users with zero true interactions, these users don't appear in reco_true_interactions
        fn_all = pd.concat([interactions_count_series, tp_series], axis=1)
        fn_all["tps"] = fn_all["tps"].fillna(0)
        fn_all = fn_all["interactions_count"] - fn_all["tps"]

        # these users are dropped here
        _fn = fn_all.drop(labels=(set(fn_all.index) - set(users_true_interactions.index))).to_numpy()
        _tn = catalog_len - _fn - self.k

        users_tpr = users_tps / (_tp + _fn).reshape(-1, 1)
        users_fpr = users_fps / (_fp + _tn).reshape(-1, 1)
        # interpolating to (1, 1)
        users_tpr = np.hstack((users_tpr, np.ones((users_tpr.shape[0], 1))))
        users_fpr = np.hstack((users_fpr, np.ones((users_fpr.shape[0], 1))))

        auc = np.trapz(users_tpr, users_fpr)
        return auc, fn_all


# https://stackoverflow.com/questions/36985659/numpy-replace-values-and-return-new-array
def put_copy(arr: np.ndarray, ind: np.ndarray, v: tp.Union[int, np.ndarray]) -> np.ndarray:
    """np.put with copy"""
    arr_copy = arr.copy()
    np.put(arr_copy, ind, v)
    return arr_copy
