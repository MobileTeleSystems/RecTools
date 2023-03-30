import attr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from rectools import ExternalId, ExternalIds, Columns
from .base import MetricAtK, Catalog
from .classification import make_confusions


TP = "__TP"
FP = "__FP"
FN = "__FN"
TN = "__TN"
LIKED = "__LIKED"


@attr.s
class AUC(MetricAtK):

    def plot_roc_for_user(self, user_id: ExternalId, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog):
        self._auc_check(reco, interactions, user_id, catalog)
        user_reco = reco.loc[reco[Columns.User] == user_id]
        user_interactions = interactions.loc[interactions[Columns.User] == user_id]
        if user_interactions.empty:
            raise ValueError(f"user {user_id} has no interactions. Can't build roc-curve")
        tpr = np.zeros(self.k + 1)
        fpr = np.zeros(self.k + 1)
        tpr[self.k] = 1.0
        fpr[self.k] = 1.0

        for roc_k in range(self.k):
            confusion_df = self.make_extended_confusions(user_reco, user_interactions, roc_k, catalog)
            tpr[roc_k] = confusion_df[TP] / (confusion_df[TP] + confusion_df[FN])
            fpr[roc_k] = confusion_df[FP] / (confusion_df[FP] + confusion_df[TN])

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, c='orange')
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='dashed', c='green')
        plt.title('ROC curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim((-0.01, 1.01))
        plt.ylim((-0.01, 1.01))
        plt.show()


    @staticmethod
    def make_extended_confusions(reco: pd.DataFrame, interactions: pd.DataFrame, k: int, catalog: Catalog):
        confusion_df = make_confusions(reco, interactions, k)
        if TN not in confusion_df:
            confusion_df[TN] = len(catalog) - k - confusion_df[FN]
        return confusion_df

    def _auc_check(self, reco: pd.DataFrame, interactions: pd.DataFrame, user_id: ExternalId, catalog: Catalog):
        MetricAtK._check(reco, interactions=interactions)
        if user_id not in reco[Columns.User]:
            raise ValueError(f"Expected user_id from recommendations, got {user_id}")

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> float:
        per_user = self.calc_per_user(reco, interactions, catalog)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> pd.Series:
        MetricAtK._check(reco, interactions=interactions)
        users = np.array(reco[Columns.User].unique())
        per_user = np.zeros_like(users, dtype=float)
        for index, user_id in enumerate(reco[Columns.User].unique()):
            per_user[index] = self.calc_for_user(user_id, reco, interactions, catalog)

        return pd.Series(data=per_user, index=users)

    def calc_for_user(self, user_id: ExternalId, reco: pd.DataFrame, interactions: pd.DataFrame, catalog: Catalog) -> float:
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
    def _auc_positive_scored_higher(positive_item: ExternalId, negative_item: ExternalId, user_reco: pd.DataFrame):
        pos_score = float(user_reco.loc[user_reco[Columns.Item] == positive_item][Columns.Score]) if not user_reco.loc[user_reco[Columns.Item] == positive_item].empty else 0.0
        neg_score = float(user_reco.loc[user_reco[Columns.Item] == negative_item][Columns.Score]) if not user_reco.loc[
            user_reco[Columns.Item] == negative_item].empty else 0.0
        if pos_score > neg_score:
            return 1.0
        if pos_score == neg_score:
            return 0.5
        return 0.0

