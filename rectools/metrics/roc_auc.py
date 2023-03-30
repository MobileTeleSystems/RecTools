import typing as tp

import attr
import matplotlib.pyplot as plt
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
class AUC(MetricAtK):
    """
    ROC-curve is a graph showing the performance of a classification model at all classification thresholds
    AUC is an area under the roc-curve
    See more: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
        and: https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf

    Parameters
    ----------
    k : int
        Number of items in top of recommendations list that will be used to calculate metric.
    """

    def plot_roc_for_user(
        self, user_id: ExternalId, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog
    ) -> None:
        """
        ROC-curve is a graph showing the performance of a classification model at all classification thresholds
        This method plot roc-curve for thresholds from 0 to k recommended items with linear continuation to (1, 1)
        See more: https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf

        y-axis: ``tpr`` equals to ``tp / (tp + fn)``
        x-axis: ``fpr`` equals to ``fp / (fp + tn)``
        where
        - ``tp`` is the number of relevant recommendations
          among the first ``k`` items in recommendation list;
        - ``tn`` is the number of items with which user has not interacted (bought, liked) with
          (in period after recommendations were given) and we do not recommend to him
          (in the top ``k`` items of recommendation list);
        - ``fp`` - number of non-relevant recommendations among the first `k` items of recommendation list;
        - ``fn`` - number of items the user has interacted with but that weren't recommended (in top-`k`)

        """
        self._auc_check(reco, interactions, user_id, catalog)
        user_reco = reco.loc[reco[Columns.User] == user_id]
        user_interactions = interactions.loc[interactions[Columns.User] == user_id]
        if user_interactions.empty:
            raise ValueError(f"user {user_id} has no interactions. Can't build roc-curve")

        tpr, fpr = self._calc_tpr_fpr(user_reco, user_interactions, catalog)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, c="orange")
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="dashed", c="green")
        plt.title("ROC curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.01, 1.01))
        plt.show()

    def _calc_tpr_fpr(
        self, user_reco: pd.DataFrame, user_interactions: pd.DataFrame, catalog: Catalog
    ) -> tp.Tuple[np.array, np.array]:
        """
        Calculate tpr and fpr for roc-curve
        make_confusions is not used for optimization. Data is updating instead of recalculating

        """
        tpr = np.zeros(self.k + 2)
        fpr = np.zeros(self.k + 2)
        tpr[self.k + 1] = 1.0
        fpr[self.k + 1] = 1.0

        _tp = 0
        _fp = 0
        _fn = len(user_interactions[Columns.Item].unique())
        _tn = len(catalog) - _fn
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

    @staticmethod
    def _auc_check(self, reco: pd.DataFrame, interactions: pd.DataFrame, user_id: ExternalId, catalog: Catalog) -> None:
        MetricAtK._check(reco, interactions=interactions)
        if user_id not in reco[Columns.User]:
            raise ValueError(f"Expected user_id from recommendations, got {user_id}")

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> float:
        """
        AUC is an area under the roc-curve.
        This method calculates auc for all users
        Note that for users without interactions calc_for_user is NaN

        """
        per_user = self.calc_per_user(reco, interactions, catalog)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> pd.Series:
        """
        Calculate AUC for user every user
        Note that for users without interactions calc_for_user is NaN

        """
        MetricAtK._check(reco, interactions=interactions)
        users = np.array(reco[Columns.User].unique())
        per_user = np.zeros_like(users, dtype=float)
        for index, user_id in enumerate(reco[Columns.User].unique()):
            per_user[index] = self.calc_for_user(user_id, reco, interactions, catalog)

        return pd.Series(data=per_user, index=users)

    def calc_for_user(
        self, user_id: ExternalId, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog
    ) -> float:
        """
        Calculate AUC for user
        AUC equals to:
         sum{by positives}sum{by negatives} Indicator(positive scored higher) / num(positives) * num(negatives)
        where
        - positives: objects that user has interacted with
        - negatives: objects that user hasn't interacted with
        See more: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

        Note that is scores are equal Indicator(...) == 0.5
        Note that for users without interactions calc_for_user is NaN

        """
        user_reco = reco.loc[reco[Columns.User] == user_id]
        user_interactions = interactions.loc[interactions[Columns.User] == user_id]
        if user_interactions.empty:
            return np.nan
        positives = user_interactions[Columns.Item]
        negatives = list(set(catalog) - set(positives))
        auc_numerator = 0
        for positive_item in positives:
            for negative_item in negatives:
                auc_numerator += self._auc_positive_scored_higher(positive_item, negative_item, user_reco)

        return auc_numerator / (len(positives) * len(negatives))

    @staticmethod
    def _auc_positive_scored_higher(
        positive_item: ExternalId, negative_item: ExternalId, user_reco: pd.DataFrame
    ) -> float:
        """
        Indicator(positive scored higher)
        Note that is scores are equal Indicator(...) == 0.5

        """
        pos_score = (
            float(user_reco.loc[user_reco[Columns.Item] == positive_item][Columns.Score])
            if not user_reco.loc[user_reco[Columns.Item] == positive_item].empty
            else 0.0
        )
        neg_score = (
            float(user_reco.loc[user_reco[Columns.Item] == negative_item][Columns.Score])
            if not user_reco.loc[user_reco[Columns.Item] == negative_item].empty
            else 0.0
        )
        if pos_score > neg_score:
            return 1.0
        if pos_score == neg_score:
            return 0.5
        return 0.0
