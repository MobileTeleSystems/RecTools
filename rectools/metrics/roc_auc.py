import typing as tp

import attr
import numpy as np
import pandas as pd

from rectools import Columns, ExternalId

from .base import Catalog, MetricAtK

TP = "__TP"
FP = "__FP"
FN = "__FN"
TN = "__TN"
LIKED = "__LIKED"


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

    def calc_user_tpr_fpr(
        self, user_reco: pd.DataFrame, user_interactions: pd.DataFrame, catalog: Catalog
    ) -> tp.Tuple[np.array, np.array]:
        """
        Calculate tpr and fpr for L AUC, make_confusions is not used for optimization
            y-axis: ``tpr`` equals to ``tp / (tp + fn)``
            x-axis: ``fpr`` equals to ``fp / (fp + tn)``

        Parameters
        ----------
        user_reco: recommendations for user: pd.DataFrame with columns: [user_id, item_id, rank],
            but user_id is considered the same
        user_interactions: interactions for user: pd.DataFrame with columns: [user_id, item_id],
            but user_id is considered the same
        catalog: catalog
        ----------

        Can make plot using:

        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, c="orange")
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="dashed", c="green")
        plt.title("ROC curve")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.01, 1.01))
        plt.show()

        """
        tpr = np.zeros(self.k + 2)
        fpr = np.zeros(self.k + 2)
        tpr[self.k + 1] = 1.0
        fpr[self.k + 1] = 1.0

        _tp = 0
        _fp = 0
        _fn = len(user_interactions[Columns.Item].unique())
        _tn = len(catalog) - _fn

        if user_reco.shape[0] == 0:
            raise ValueError("user_reco is empty")
        if user_interactions.shape[0] == 0:
            raise ValueError("user_interactions is empty")

        if self.k > user_reco.shape[0]:
            raise ValueError(
                f"user {user_reco.iloc[0][Columns.User]} has {user_reco.shape[0]} rows, which is less than k: {self.k}"
            )
        for roc_k in range(self.k):
            if not user_interactions.loc[user_interactions[Columns.Item] == user_reco.iloc[roc_k][Columns.Item]].empty:
                _tp += 1
                _fn -= 1
            else:
                _fp += 1
                _tn -= 1
            tpr[roc_k + 1] = _tp / (_tp + _fn)
            fpr[roc_k + 1] = _fp / (_fp + _tn)

        return tpr, fpr

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> float:
        """
        AUC is an area under the roc-curve.
        LAUC is limited AUC. See more: https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf
        This method calculates auc for all users
        Note that for users without interactions calc_for_user is NaN

        """
        per_user = self.calc_per_user(reco, interactions, catalog)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> pd.Series:
        """
        Calculate LAUC for user every user
        Note that for users without interactions calc_for_user is NaN

        """
        MetricAtK._check(reco, interactions=interactions)
        users = np.array(reco[Columns.User].unique())
        per_user = np.zeros_like(users, dtype=float)
        for index, user_id in enumerate(reco[Columns.User].unique()):
            per_user[index] = self.calc_for_user(user_id, reco, interactions, catalog)
            # print(per_user[index])
        return pd.Series(data=per_user, index=users)

    def calc_for_user(
        self, user_id: ExternalId, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog
    ) -> float:
        """
        Calculate LAUC for user using numpy.trapz
        Note that for users without interactions calc_for_user is NaN

        """
        MetricAtK._check(reco, interactions=interactions)
        if user_id not in reco[Columns.User]:
            raise ValueError(f"Expected user_id from recommendations, got {user_id}")
        user_interactions = interactions.loc[interactions[Columns.User] == user_id]
        if user_interactions.empty:
            return np.nan
        else:
            user_reco = reco.loc[reco[Columns.User] == user_id]
        tpr, fpr = self.calc_user_tpr_fpr(user_reco, user_interactions, catalog)
        return np.trapz(tpr, fpr)
