import attr
import pandas as pd

from rectools import Columns

from .base import Catalog, MetricAtK


@attr.s
class ItemCoverage(MetricAtK):
    """
    Item space coverage (also referred as catalog coverage) is a metric that shows
    what part of the items is covered by first k recommendations
    ItemCoverage = #recommended_items / num_items

    Parameters
    ----------
    k : int
        Number of items in top of recommendations list that will be used to calculate metric.

    """

    def calc(self, reco: pd.DataFrame, catalog: Catalog) -> float:
        """
        Calculate item space coverage for all users

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        float
            Value of metric.
        """
        reco_k_first_ranks = reco[reco[Columns.Rank] <= self.k]
        return len(reco_k_first_ranks[Columns.Item].unique()) / len(catalog)

    def calc_per_user(self, reco: pd.DataFrame, catalog: Catalog) -> pd.Series:
        """
        Calculate item space coverage per user

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        catalog : collection
            Collection of unique item ids that could be used for recommendations.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        reco_k_first_ranks = reco[reco[Columns.Rank] <= self.k]
        return reco_k_first_ranks.groupby(Columns.User)[Columns.Item].nunique() / len(catalog)


@attr.s
class NumRetrieved(MetricAtK):
    """
    Number of recommendations retrieved is a metric that shows
    how much items retrieved by first k recommendations (less or equal k)
    See more: https://elliot.readthedocs.io/en/latest/guide/metrics/coverage.html

    Parameters
    ----------
    k : int
        Number of items in top of recommendations list that will be used to calculate metric.

    """

    def calc(self, reco: pd.DataFrame) -> float:
        """
        Calculate average num retrieved for all users.
        If num retrieved equals k, it means that k items were recommended to every user

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame) -> pd.Series:
        """
        Calculate num retrieved per user.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        reco_k_first_ranks = reco[reco[Columns.Rank] <= self.k]
        return reco_k_first_ranks.groupby(Columns.User)[Columns.Item].count()
