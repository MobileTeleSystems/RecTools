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

"""AUC like recommendations metrics."""
import typing as tp

import attr
import pandas as pd

from rectools import Columns
from rectools.metrics.base import MetricAtK, outer_merge_reco
from rectools.utils import select_by_type


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


@attr.s
class _AUCMetric(MetricAtK):
    """
    Partial AUC metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    """

    insufficient_cases: str = attr.ib(default="exclude")

    @classmethod
    def fit(cls, reco: pd.DataFrame, interactions: pd.DataFrame, k_max: int) -> AUCFitted:
        """
        Prepare intermediate data for effective calculation.

        You can use this method to prepare some intermediate data
        for later calculation. It can optimize calculations if
        you want calculate partial AUC metric values for different `k`.
        """
        cls._check(reco, interactions=interactions)

        outer_merged = outer_merge_reco(reco, interactions)
        outer_merged["__tp"] = (~outer_merged[Columns.Rank].isna()) & (outer_merged["__test_positive"])
        outer_merged["__fp"] = (~outer_merged[Columns.Rank].isna()) & (~outer_merged["__test_positive"])
        outer_merged["__fp_cumsum"] = outer_merged.groupby(Columns.User)["__fp"].cumsum()
        outer_merged["__test_pos_cumsum"] = outer_merged.groupby(Columns.User)["__test_positive"].cumsum()

        num_pos = outer_merged.groupby(Columns.User)["__test_pos_cumsum"].max()

        # Every user with FP count more then k_max has sufficient recommendations for pauc metrics
        # We calculate and keep number of false positives for all other users
        users_num_fp = outer_merged.groupby(Columns.User)["__fp_cumsum"].max()
        num_fp_insufficient = users_num_fp[users_num_fp < k_max]
        users_with_fn = outer_merged[outer_merged[Columns.Rank].isna()][Columns.User].unique()
        num_fp_insufficient = num_fp_insufficient[num_fp_insufficient.index.isin(users_with_fn)]

        return AUCFitted(outer_merged, num_pos, num_fp_insufficient)
    
    def _process_insufficient_cases(
        self, outer_merged: pd.DataFrame, num_pos: pd.Series, num_fp_insufficient: pd.Series, metric_name: str
    ) -> pd.Series:
        if self.insufficient_cases == "don't check":
            return outer_merged, num_pos
        
        insufficient_users = num_fp_insufficient[num_fp_insufficient < self.k].index.values
        if not insufficient_users.any():
            return outer_merged, num_pos

        if self.insufficient_cases == "exclude":
            outer_merged_suf = outer_merged[~outer_merged[Columns.User].isin(insufficient_users)].copy()  # remove copy
            num_pos_suf = num_pos[~num_pos.index.isin(insufficient_users)].copy()  # remove copy
            return outer_merged_suf, num_pos_suf
        
        raise ValueError(
            f"""
            {metric_name}@{self.k} metric requires at least {self.k} negatives in 
            recommendations for each user. Or all items from user test interactions ranked in
            recommendations - meaning that all other recommended items will be negatives.
            There are {len(insufficient_users)} users with less then required negatives.
            For correct {metric_name} computation please provide each user with sufficient number
            of recommended items. It fill be enough to have `n_user_positives` + `{metric_name}_k`
            recommended items for each user.
            You can disable this error by specifying `insufficient_cases` = "don't check" or
            by dropping all users with insuffissient recommendations from metric computation
            with specifying `insufficient_cases` = "drop".
            """
        )
        
    def _calc_roc_auc(self, outer_merged: pd.DataFrame, num_pos: pd.Series) -> pd.Series:
        """
        Calculate partial roc auc given that all data has already been prepared and cropped following
        required metric logic.
        """
        cropped = outer_merged.copy()  # TODO: remove copy
        cropped["__auc_numenator_gain"] = (self.k - cropped["__fp_cumsum"]) * cropped["__tp"]
        pauc_numenator = cropped.groupby(Columns.User)["__auc_numenator_gain"].sum()
        pauc_denominator = num_pos * self.k
        pauc = (pauc_numenator / (pauc_denominator)).fillna(0)
        return pauc

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
        Calculate metric value from merged recommendations.

        Parameters
        ----------
        fitted : AUCFitted
            ...

        Returns
        -------
        float
            Value of metric (average between users).
        """
        per_user = self.calc_per_user_from_fitted(fitted)
        return per_user.mean()

    def calc_per_user_from_fitted(self, fitted: AUCFitted) -> float:
        """
        Calculate metric value from merged recommendations.

        Parameters
        ----------
        fitted : AUCFitted
            ...

        Returns
        -------
        float
            Value of metric (average between users).
        """
        raise NotImplementedError()


@attr.s
class PAUC(_AUCMetric):
    r"""
    Partial AUC at k (pAUC@k).
    Write all info here
    """

    def calc_per_user_from_fitted(self, fitted: AUCFitted) -> pd.Series:
        """
        Calculate metric values for all users from outer merged recommendations.

        Parameters
        ----------
        outer_merged : pd.DataFrame
            Result of merging recommendations and interactions tables with `outer` logic and full ranks provided.
            Can be obtained using `outer_merge_reco` function.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        # Keep k first false positives for roc auc computation
        outer_merged = fitted.outer_merged_enriched
        cropped = outer_merged[(outer_merged["__fp_cumsum"] < self.k) & (~outer_merged[Columns.Rank].isna())].copy()
        outer_merged, num_pos = self._process_insufficient_cases(
            outer_merged=cropped,
            num_pos=fitted.num_pos,
            num_fp_insufficient=fitted.num_fp_insufficient,
            metric_name="PAUC",
        )
        return self._calc_roc_auc(outer_merged, num_pos)


@attr.s
class PAP(_AUCMetric):
    r"""
    Partial AUC ... at k (pAp@k).
    Write all info here
    """

    def calc_per_user_from_fitted(self, fitted: AUCFitted) -> pd.Series:
        """
        Calculate metric values for all users from outer merged recommendations.

        Parameters
        ----------
        outer_merged : pd.DataFrame
            Result of merging recommendations and interactions tables with `outer` logic and full ranks provided.
            Can be obtained using `outer_merge_reco` function.

        Returns
        -------
        pd.Series
            Values of metric (index - user id, values - metric value for every user).
        """
        # Keep k first false positives and k first true positives for roc auc computation
        outer_merged = fitted.outer_merged_enriched
        cropped = outer_merged[
            (outer_merged["__test_pos_cumsum"] <= self.k)
            & (outer_merged["__fp_cumsum"] < self.k)
            & (~outer_merged[Columns.Rank].isna())
        ].copy()
        
        outer_merged, num_pos = self._process_insufficient_cases(
            outer_merged=cropped,
            num_pos=fitted.num_pos.clip(upper=self.k),
            num_fp_insufficient=fitted.num_fp_insufficient,
            metric_name="PAP",
        )
        return self._calc_roc_auc(outer_merged, num_pos)


AucMetric = tp.Union[PAUC, PAP]


def calc_auc_metrics(
    metrics: tp.Dict[str, AucMetric],
    reco: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tp.Dict[str, float]:
    """
    Calculate any AUC-like metrics (PAUC for now).

    Works with pre-prepared data.

    Warning: It is not recommended to use this function directly.
    Use `calc_metrics` instead.

    Parameters
    ----------
    metrics : dict(str -> (AucMetric))
        Dict of metric objects to calculate,
        where key is metric name and value is metric object.
    outer_merged : pd.DataFrame
        Result of merging recommendations and interactions tables with `outer` logic and full ranks provided.
        Can be obtained using `outer_merge_reco` function.

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
