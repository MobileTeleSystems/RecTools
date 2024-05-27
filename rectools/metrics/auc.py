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
class AUCFitted:  # TODO: docrstring
    """
    Container with meta data got from `_AUCMetric.fit` method.

    Parameters
    ----------
    precision_at_k : csr_matrix
        CSR matrix where rows corresponds to users,
        rows corresponds all possible k from 0 to `k_max`,
        and values are weighted precisions for relevant recommended items.
    users : np.ndarray
        Array of user ids.
    n_relevant_items : np.ndarray
        Tally of relevant items for each user.
        Users are in the same order as in `precision_at_k` matrix.
    """

    outer_merged_enriched: pd.DataFrame = attr.ib()
    num_pos: pd.Series = attr.ib()
    users_num_fp_insuf: pd.Series = attr.ib()

@attr.s
class _AUCMetric(MetricAtK):
    """
    AUC like metric base class.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    k : int
        Number of items at the top of recommendations list that will be used to calculate metric.
    """
    insufficient_cases: str = attr.ib(default="exclude")
    
    @classmethod
    def fit(self, reco: pd.DataFrame, interactions: pd.DataFrame, k_max: int) -> AUCFitted:
        self._check(reco, interactions=interactions)
        
        outer_merged = outer_merge_reco(reco, interactions)  # crop works for isna logic?
        
        # TODO: crop to max before cumsum
        outer_merged["__tp"] = (~outer_merged[Columns.Rank].isna()) & (outer_merged["__test_positive"])
        outer_merged["__fp"] = (~outer_merged[Columns.Rank].isna()) & (~outer_merged["__test_positive"])
        outer_merged["__fp_cumsum"] = outer_merged.groupby(Columns.User)["__fp"].cumsum()
        outer_merged["__test_pos_cumsum"] = outer_merged.groupby(Columns.User)["__test_positive"].cumsum()
        
        num_pos = outer_merged.groupby(Columns.User)["__test_pos_cumsum"].max()
        
        # Every user with FP count more then k_max has sufficient recommendations for pauc metrics
        # We calculate and keep number of false positives for all other users
        users_num_fp = outer_merged.groupby(Columns.User)["__fp_cumsum"].max()
        users_num_fp_insuf = users_num_fp[users_num_fp < k_max]
        users_with_fn = outer_merged[outer_merged[Columns.Rank].isna()][Columns.User].unique()
        users_num_fp_insuf = users_num_fp_insuf[users_num_fp_insuf.index.isin(users_with_fn)]
        
        return AUCFitted(outer_merged, num_pos, users_num_fp_insuf)
    
    def _calc_partial_roc_auc(self, outer_merged: pd.DataFrame, num_pos: pd.Series, users_num_fp_insuf: pd.Series, metric_name: str) -> pd.Series:
        if self.insufficient_cases in ["exclude", "raise"]:
            insufficient_users = users_num_fp_insuf[users_num_fp_insuf < self.k].index.values

            if insufficient_users.any():
                if self.insufficient_cases == "exclude":
                    outer_merged = outer_merged[~outer_merged[Columns.User].isin(insufficient_users)]
                    num_pos = num_pos[~num_pos.index.isin(insufficient_users)]
                else:
                    raise ValueError(
                        f"""
                        {metric_name}@{self.k} metric requires at least {self.k} negatives in reco for each user.
                        Or all items from user interactions ranked in reco meaning that all other reco will be
                        negatives.
                        There are {len(insufficient_users)} users with less negatives.
                        For correct {metric_name} computation please provide each user with sufficient number
                        of recommended items. It fill be enough to have `n_user_positives` + `{metric_name}_k`
                        recommended items for each user.
                        You can disable this error by specifying `insufficient_cases` = "don't check" or
                        by dropping all users with insuffissient recommendations from metric computation
                        with `insufficient_cases` = "drop"
                        """
                    )

        cropped = outer_merged[(outer_merged["__fp_cumsum"] < self.k) & (~outer_merged[Columns.Rank].isna())].copy()
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
        return self._calc_partial_roc_auc(
            outer_merged=fitted.outer_merged_enriched,
            num_pos=fitted.num_pos,
            users_num_fp_insuf=fitted.users_num_fp_insuf,
            metric_name="PAUC"
        )


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
        return self._calc_partial_roc_auc(
            outer_merged=fitted.outer_merged_enriched[fitted.outer_merged_enriched["__test_pos_cumsum"] <= self.k].copy(),
            num_pos=fitted.num_pos.clip(upper=self.k),
            users_num_fp_insuf=fitted.users_num_fp_insuf,
            metric_name="PAP"
        )
        

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
    metrics : dict(str -> (pAUC))
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
