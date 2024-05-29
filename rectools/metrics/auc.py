#  Copyright 2024 MTS (Mobile Telesystems)
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

"""ROC AUC based ranking recommendations metrics."""
import typing as tp
from enum import Enum

import attr
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK, outer_merge_reco
from rectools.utils import select_by_type


class InsufficientHandling(str, Enum):
    """Strategy for handling insufficient reommendations cases"""

    SKIP = "skip"
    EXCLUDE = "exclude"
    RAISE = "raise"


@attr.s
class AUCFitted:
    """
    Container with meta data got from `_AUCMetric.fit` method.

    Parameters
    ----------
    outer_merged_enriched : pd.DataFrame
        Recommendations outer merged with test interactions. Table has Columns.User, Columns.Item,
        Columns.Rank. Precomputed columns include "__test_positive", "__tp", "__fp", "__fp_cumsum",
        "__test_pos_cumcum". All ranks for all users are present with no skipping.
        Null ranks are specified for test interactions that were not predicted in recommendations.
    num_pos : pd.Series
        Number of positive items for each user in test insteractions.
    num_fp_insufficient : pd.Series
        Number of false positive items for each user in `outer_merged_enriched` that had at least
        one false negative. This users will be checked for insufficient cases processing.

    """

    outer_merged_enriched: pd.DataFrame = attr.ib()
    num_pos: pd.Series = attr.ib()
    num_fp_insufficient: pd.Series = attr.ib()


class _AUCMetric(MetricAtK):
    """
    ROC AUC based metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    insufficient_handling : {"skip", "raise", "exclude"}, default `"skip"`
        Method of handling users with insufficient recommendations for metric calculation.
        ROC AUC based metrics with `k` parameter often need more then `k` recommendations
        for each user. This happens because this metrics calculate ROC AUC
        score for specific number of user false positives and ranked test positives that is derived
        from provided `k` parameter but is not equal to it.
        The following methods are available:
        - `skip` - don'c check for insufficient recommendations lists, handle all of insufficient
        cases as if algorithms are not able to retrieve users unpredicted test positives on any k
        level. This will understate the metric value;
        - `exclude` - exclude all users with insufficient recommendations lists from metrics
        computation;
        - `raise` - raise error if there are any users with insufficient recommendations lists. Use
        this option very carefully because some of the algorithms are unable to provide full required
        lists because of their inference logic. So can get errors even if you requested enough
        recommendations in `recommend` method. For example, ItemKNN generates recommendations only
        until the model has non-zero scores for the item in item-item similarity matrix. So with
        small `K` for neighbours in ItemKNN and big `K` for `recommend` and AUC based metric you
        will still get an error when `insufficient_handling` is set to `raise`.
    """

    def __init__(self, k: int, insufficient_handling: tp.Optional[str] = InsufficientHandling.SKIP) -> None:
        super().__init__(k=k)
        try:
            self.insufficient_handling = InsufficientHandling(insufficient_handling)
        except ValueError:
            possible_values = {item.value for item in InsufficientHandling.__members__.values()}
            raise ValueError(
                f"`insufficient_handling` must be one of the {possible_values}. Got {insufficient_handling}."
            )

    @classmethod
    def fit(cls, reco: pd.DataFrame, interactions: pd.DataFrame, k_max: int) -> AUCFitted:
        """
        Prepare intermediate data for effective calculation.

        You can use this method to prepare some intermediate data
        for later calculation. It can optimize calculations if
        you want calculate different AUC based metrics with different `k` parameter.
        """
        cls._check(reco, interactions=interactions)

        outer_merged = outer_merge_reco(reco, interactions)
        outer_merged["__tp"] = (~outer_merged[Columns.Rank].isna()) & (outer_merged["__test_positive"])
        outer_merged["__fp"] = (~outer_merged[Columns.Rank].isna()) & (~outer_merged["__test_positive"])
        outer_merged["__fp_cumsum"] = outer_merged.groupby(Columns.User)["__fp"].cumsum()
        outer_merged["__test_pos_cumsum"] = outer_merged.groupby(Columns.User)["__test_positive"].cumsum()

        num_pos = outer_merged.groupby(Columns.User)["__test_pos_cumsum"].max()

        # Every user with FP count more then k_max has sufficient recommendations for partial AUC based metrics
        # We calculate and keep number of false positives for all other users
        users_num_fp = outer_merged.groupby(Columns.User)["__fp_cumsum"].max()
        num_fp_insufficient = users_num_fp[users_num_fp < k_max]
        users_with_fn = outer_merged[outer_merged[Columns.Rank].isna()][Columns.User].unique()
        num_fp_insufficient = num_fp_insufficient[num_fp_insufficient.index.isin(users_with_fn)]

        return AUCFitted(outer_merged, num_pos, num_fp_insufficient)

    def _get_sufficient_reco_explananation(self) -> str:
        raise NotImplementedError()

    def _handle_insufficient_cases(
        self, outer_merged: pd.DataFrame, num_pos: pd.Series, num_fp_insufficient: pd.Series, metric_name: str
    ) -> pd.Series:
        if self.insufficient_handling == InsufficientHandling.SKIP:
            return outer_merged, num_pos

        insufficient_users = num_fp_insufficient[num_fp_insufficient < self.k].index.values
        if not insufficient_users.any():
            return outer_merged, num_pos

        if self.insufficient_handling == InsufficientHandling.EXCLUDE:
            outer_merged_suf = outer_merged[~outer_merged[Columns.User].isin(insufficient_users)]
            num_pos_suf = num_pos[~num_pos.index.isin(insufficient_users)]
            return outer_merged_suf, num_pos_suf

        raise ValueError(
            f"""
            {metric_name}@{self.k} metric requires at least {self.k} negatives in
            recommendations for each user. Or all items from user test interactions ranked in
            recommendations - meaning that all other recommended items will be negatives.
            There are {len(insufficient_users)} users with less then required negatives.
            For correct {metric_name} computation please provide each user with sufficient number
            of recommended items. {self._get_sufficient_reco_explananation()}
            You can disable this error by specifying `insufficient_handling`="{InsufficientHandling.SKIP}" or
            by excluding all users with insuffissient recommendations from metric computation
            with specifying `insufficient_handling` = "{InsufficientHandling.EXCLUDE}".
            """
        )

    def _calc_roc_auc(self, cropped_outer_merged: pd.DataFrame, num_pos: pd.Series) -> pd.Series:
        """
        Calculate ROC AUC given that all data has already been prepared, merged, enriched and cropped following
        metric specific logic.
        """
        cropped = cropped_outer_merged.copy()
        cropped["__auc_numenator_gain"] = (self.k - cropped["__fp_cumsum"]) * cropped["__tp"]
        auc_numenator = cropped.groupby(Columns.User)["__auc_numenator_gain"].sum()
        auc_denominator = num_pos * self.k
        auc = (auc_numenator / (auc_denominator)).fillna(0)
        return auc

    def calc(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> float:
        """
        Calculate metric value.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user(reco, interactions)
        return per_user.mean()

    def calc_per_user(self, reco: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
        """
        Calculate metric values for all users.

        Parameters
        ----------
        reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
        interactions : pd.DataFrame
            Interactions table with columns `Columns.User`, `Columns.Item`.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        self._check(reco, interactions=interactions)
        fitted = self.fit(reco, interactions, k_max=self.k)
        return self.calc_per_user_from_fitted(fitted)

    def calc_from_fitted(self, fitted: AUCFitted) -> float:
        """
        Calculate metric value from fitted data.

        Parameters
        ----------
        fitted : AUCFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_fitted(fitted)
        return per_user.mean()

    def calc_per_user_from_fitted(self, fitted: AUCFitted) -> pd.Series:
        """
        Calculate metric values for all users from from fitted data.

        Parameters
        ----------
        fitted : AUCFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        raise NotImplementedError()


class PAUC(_AUCMetric):
    r"""
    Partial AUC at k (pAUC@k).
    pAUC@k measures ROC AUC score for ranking of the top-k irrelevant items and all relevant items
    for each user. Averaged between users. For one user the formula is:

    .. math::
        pAUC@k = \frac{1}{kn_+}\sum_{{x_i}\in S^+}\sum_{{x_j}\in S^-}\mathbb{1}[s(x_i)\geq s(x_j)]

    where
        - :math:`k` is the number of user top scored negatives for metric computation
        - :math:`s` is a scoring function which provides scores to rank items for user
        - :math:`\mathbb{1}` is the indicator function
        - :math:`n_+` is the number of all user test positives
        - :math:`S^+` is the set of all positives for user
        - :math:`S^-` is the set of top :math:`k` negatives for user acquired by :math:`s`
        - :math:`x_i` and `x_j` are user positives and negatives for metric computation

    Analysed in ["Rich-Item Recommendations for Rich-Users: Exploiting Dynamic and Static Side
    Information"](https://arxiv.org/abs/2001.10495), analysed in ["Optimization and Analysis of the
    pAp@k Metric for Recommender Systems"](https://proceedings.mlr.press/v119/hiranandani20a.html)


    Parameters
    ----------
    k : int
        Number of top irrelevant items for user to be taken for ROC AUC computation. This does not
        equal `k` for classic `@k` metrics.
    insufficient_handling : {"skip", "raise", "exclude"}, default `"skip"`
        Method of handling users with insufficient recommendation lists for metric calculation.
        pAUC@k needs more then `k` recommendations for each user. This happens because this metris
        calculate ROC AUC score for specific number of user false positives and ranked test
        positives that is derived from provided `k` parameter but is not equal to it.
        It fill be enough to have :math:`n^+` (number of user positives) + `k` recommended items for
        each user.
        The following methods are available:
        - `skip` - don'c check for insufficient recommendations lists, handle all of insufficient
        cases as if algorithms are not able to retrieve users unpredicted test positives on any k
        level. This will understate the metric value if recommendation lists are not sufficient;
        - `exclude` - exclude all users with insufficient recommendations lists from metrics
        computation;
        - `raise` - raise error if there are any users with insufficient recommendations lists. Use
        this option very carefully because some of the algorithms are unable to provide full required
        lists because of their inference logic. So can get errors even if you requested enough
        recommendations in `recommend` method. For example, ItemKNN generates recommendations only
        until the model has non-zero scores for the item in item-item similarity matrix. So with
        small `K` for neighbours in ItemKNN and big `K` for `recommend` and AUC based metric you
        will still get an error when `insufficient_handling` is set to `raise`.
        
        Examples
        --------
        >>> reco = pd.DataFrame(
        ...     {
        ...         Columns.User: [1, 1, 2, 2, 2, 3, 3],
        ...         Columns.Item: [1, 2, 3, 1, 2, 3, 2],
        ...         Columns.Rank: [1, 2, 1, 2, 3, 1, 2],
        ...     }
        ... )
        >>> interactions = pd.DataFrame(
        ...     {
        ...         Columns.User: [1, 1, 2, 2, 3, 3],
        ...         Columns.Item: [1, 2, 1, 3, 1, 2],
        ...     }
        ... )
        >>> PAUC(k=1).calc_per_user(reco, interactions).values
        array([1., 1., 0.])
        >>> PAUC(k=3).calc_per_user(reco, interactions).values
        array([1., 1. , 0.33333333])
        >>> PAUC(k=3, insufficient_handling="exclude").calc_per_user(reco, interactions).values
        array([[1., 1.])
    """

    def _get_sufficient_reco_explananation(self) -> str:
        return f"""
            It fill be enough to have `n_user_positives` + `PAUC_k` ({self.k}) recommended items for
            each user. For simplification it will be enough to have max(`n_user_positives`) +
            `PAUC_k` ({self.k}) recommended items for all users if max(`n_user_positives`) is
            not too high.
            """

    def calc_per_user_from_fitted(self, fitted: AUCFitted) -> pd.Series:
        """
        Calculate metric values for all users from from fitted data.

        Parameters
        ----------
        fitted : AUCFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        outer_merged = fitted.outer_merged_enriched

        # Keep k first false positives for roc auc computation, keep all predicted test positives
        cropped = outer_merged[(outer_merged["__fp_cumsum"] < self.k) & (~outer_merged[Columns.Rank].isna())]

        cropped_suf, num_pos_suf = self._handle_insufficient_cases(
            outer_merged=cropped,
            num_pos=fitted.num_pos,
            num_fp_insufficient=fitted.num_fp_insufficient,
            metric_name="PAUC",
        )
        return self._calc_roc_auc(cropped_suf, num_pos_suf)


class PAP(_AUCMetric):
    r"""
    `partial-AUC + precision@k` (pAp@k) joint classification and ranking metric.
    pAp@k measures AUC between the top-k irrelevant items and top-β relevant items, where β is the
    minimum of k and the number of relevant items. The metric behaves like prec@k when the number of
    relevant items are larger than k and like pAUC otherwise.  Averaged between users. For one user
    the formula is:

    .. math::
        pAp@k = \frac{1}{k\beta}\sum_{{x_i}\in S^+}\sum_{{x_j}\in S^-}\mathbb{1}[s(x_i)\geq s(x_j)]

    where
        - :math:`k` is the number of top scored negatives and top border for number of user top
    scored positives for metric computation
        - :math:`s` is a scoring function which provides scores to rank items for user
        - :math:`\mathbb{1}` is the indicator function
        - :math:`\beta` is the minimum between `k` and number of user test positives
        - :math:`S^+` is the set of top :math:`\beta` positives for user acquired by :math:`s`
        - :math:`S^-` is the set of top :math:`k` negatives for user acquired by :math:`s`
        - :math:`x_i` and `x_j` are user positives and negatives for metric computation

    Introduced in ["Rich-Item Recommendations for Rich-Users: Exploiting Dynamic and Static Side
    Information"](https://arxiv.org/abs/2001.10495), analysed in ["Optimization and Analysis of the
    pAp@k Metric for Recommender Systems"](https://proceedings.mlr.press/v119/hiranandani20a.html)


    Parameters
    ----------
    k : int
        Number of top irrelevant items for user to be taken for ROC AUC computation. This does not
        equal `k` for classic `@k` metrics.
    insufficient_handling : {"skip", "raise", "exclude"}, default `"skip"`
        Method of handling users with insufficient recommendation lists for metric calculation.
        pAp@k needs more then `k` recommendations for each user. This happens because this metris
        calculate ROC AUC score for specific number of user false positives and ranked test
        positives that is derived from provided `k` parameter but is not equal to it.
        It fill be enough to have `k` * 2 recommended items for each user.
        The following methods are available:
        - `skip` - don'c check for insufficient recommendations lists, handle all of insufficient
        cases as if algorithms are not able to retrieve users unpredicted test positives on any k
        level. This will understate the metric value if recommendation lists are not sufficient;
        - `exclude` - exclude all users with insufficient recommendations lists from metrics
        computation;
        - `raise` - raise error if there are any users with insufficient recommendations lists. Use
        this option very carefully because some of the algorithms are unable to provide full required
        lists because of their inference logic. So can get errors even if you requested enough
        recommendations in `recommend` method. For example, ItemKNN generates recommendations only
        until the model has non-zero scores for the item in item-item similarity matrix. So with
        small `K` for neighbours in ItemKNN and big `K` for `recommend` and AUC based metric you
        will still get an error when `insufficient_handling` is set to `raise`.
        
        
        Examples
        --------
        >>> reco = pd.DataFrame(
        ...     {
        ...         Columns.User: [1, 1, 2, 2, 2, 3, 3],
        ...         Columns.Item: [1, 2, 3, 1, 2, 3, 2],
        ...         Columns.Rank: [1, 2, 1, 2, 3, 1, 2],
        ...     }
        ... )
        >>> interactions = pd.DataFrame(
        ...     {
        ...         Columns.User: [1, 1, 2, 2, 3, 3],
        ...         Columns.Item: [1, 2, 1, 3, 1, 2],
        ...     }
        ... )
        >>> PAP(k=1).calc_per_user(reco, interactions).values
        array([1., 1., 0.])
        >>> PAP(k=3).calc_per_user(reco, interactions).values
        array([1., 1. , 0.33333333])
        >>> PAP(k=3, insufficient_handling="exclude").calc_per_user(reco, interactions).values
        array([[1., 1.])
    """

    def _get_sufficient_reco_explananation(self) -> str:
        return f"""
            It fill be enough to have min(`n_user_positives`, `PAP_k` ({self.k}))  + `PAP_k`
            ({self.k}) recommended items for each user.
            For simplification it will be enough to have `PAP_k` ({self.k})) * 2 recommended items
            for all users.
            """

    def calc_per_user_from_fitted(self, fitted: AUCFitted) -> pd.Series:
        """
        Calculate metric values for all users from outer merged recommendations.

        Parameters
        ----------
        fitted : AUCFitted
            Meta data that got from `.fit` method.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        outer_merged = fitted.outer_merged_enriched

        # Keep k first false positives and k first predicted test positives for roc auc computation
        cropped = outer_merged[
            (outer_merged["__test_pos_cumsum"] <= self.k)
            & (outer_merged["__fp_cumsum"] < self.k)
            & (~outer_merged[Columns.Rank].isna())
        ]

        cropped_suf, num_pos_suf = self._handle_insufficient_cases(
            outer_merged=cropped,
            num_pos=fitted.num_pos.clip(upper=self.k),
            num_fp_insufficient=fitted.num_fp_insufficient,
            metric_name="PAP",
        )
        return self._calc_roc_auc(cropped_suf, num_pos_suf)


AucMetric = tp.Union[PAUC, PAP]


def calc_auc_metrics(
    metrics: tp.Dict[str, AucMetric],
    reco: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tp.Dict[str, float]:
    """
    Calculate any ROC AUC based ranking metric.

    Works with pre-prepared data.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> (AucMetric))
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    reco : pd.DataFrame
            Recommendations table with columns `Columns.User`, `Columns.Item`, `Columns.Rank`.
    interactions : pd.DataFrame, optional
        Interactions table with columns `Columns.User`, `Columns.Item`.
        Obligatory only for some types of metrics.

    Returns
    -------
    dict(str->float)
        Dictionary where keys are the same with keys in `metrics`
        and values are metric calculation results.
    """
    results = {}

    auc_metrics: tp.Dict[str, AucMetric] = select_by_type(metrics, AucMetric)
    if auc_metrics:
        k_max = max(metric.k for metric in metrics.values())
        fitted = _AUCMetric.fit(reco, interactions, k_max)
        for name, metric in auc_metrics.items():
            results[name] = metric.calc_from_fitted(fitted)

    return results
