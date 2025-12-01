import typing as tp
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
import typing_extensions as tpe

from rectools import Columns
from rectools.dataset import Dataset
from rectools.dataset.identifiers import ExternalIds
from rectools.exceptions import NotFittedForStageError
from rectools.model_selection import Splitter
from rectools.models.base import ErrorBehaviour, ModelBase


@tp.runtime_checkable
class ClassifierBase(tp.Protocol):
    """
    A protocol that defines the interface for a classifier model. Classes implementing this protocol
    should provide methods for fitting the model and predicting class probabilities.

    Methods
    -------
    fit
        Fit the classifier to the training data.
    predict_proba
        Predict class probabilities for the given input data. The implementation should return
        an array where each element is a probability distribution over the classes.
    """

    def fit(self, *args: tp.Any, **kwargs: tp.Any) -> tpe.Self:
        """
        Fit the classifier to the training data.

        Parameters
        ----------
        *args : any
            Positional arguments for fitting the classifier.
        **kwargs : any
            Keyword arguments for fitting the classifier.

        Returns
        -------
        tpe.Self
            The fitted classifier instance.
        """

    def predict_proba(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        """
        Predict class probabilities for the given input data.

        Parameters
        ----------
        *args : any
            Positional arguments for predicting probabilities.
        **kwargs : any
            Keyword arguments for predicting probabilities.

        Returns
        -------
        np.ndarray
            An array of predicted probabilities, where each element is a distribution over the classes.
        """


@tp.runtime_checkable
class RankerBase(tp.Protocol):
    """
    A protocol that defines the interface for a ranker model. Classes implementing this protocol
    should provide methods for fitting the model and predicting scores for ranking.

    Methods
    -------
    fit
        Fit the ranker to the training data.
    predict
        Predict scores for the given input data. The implementation should return an array of
        scores that can be used for ranking items.
    """

    def fit(self, *args: tp.Any, **kwargs: tp.Any) -> tpe.Self:
        """
        Fit the ranker to the training data.

        Parameters
        ----------
        *args : any
            Positional arguments for fitting the ranker.
        **kwargs : any
            Keyword arguments for fitting the ranker.

        Returns
        -------
        tpe.Self
            The fitted ranker instance.
        """

    def predict(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        """
        Predict scores for the given input data.

        Parameters
        ----------
        *args : any
            Positional arguments for predicting scores.
        **kwargs : any
            Keyword arguments for predicting scores.

        Returns
        -------
        np.ndarray
            An array of predicted scores, which can be used for ranking items.
        """


class Reranker:
    """
    A class used to re-rank candidates from first stage using ranking model.
    The model can be either a classifier or a ranker.
    """

    def __init__(
        self,
        model: tp.Union[ClassifierBase, RankerBase],
        fit_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ):
        """
        Initialize the Reranker with `model` and `fit_kwargs`.

        Parameters
        ----------
        model : ClassifierBase | RankerBase
            Ranking model. It must implement `fit` and `predict` or `predict_proba`.
        fit_kwargs : dict(str -> any), optional, default ``None``
            Additional keyword arguments to pass to the model's fit method.
        """
        self.model = model
        self.fit_kwargs = fit_kwargs

    def prepare_fit_kwargs(self, candidates_with_target: pd.DataFrame) -> tp.Dict[str, tp.Any]:
        """
        Prepare the keyword arguments for fitting the model, based on the provided candidates with targets.

        Parameters
        ----------
        candidates_with_target : pd.DataFrame
            A DataFrame containing the features and target labels for the candidates.

        Returns
        -------
        dict(str -> any)
            A dictionary containing the features (`X`) and target labels (`y`) for fitting the model.
        """
        candidates_with_target = candidates_with_target.drop(columns=Columns.UserItem)

        fit_kwargs = {
            "X": candidates_with_target.drop(columns=Columns.Target),
            "y": candidates_with_target[Columns.Target],
        }

        if self.fit_kwargs is not None:
            fit_kwargs.update(self.fit_kwargs)

        return fit_kwargs

    def fit(self, candidates_with_target: pd.DataFrame) -> None:
        """
        Fit the model using the provided candidates with target labels.

        Parameters
        ----------
        candidates_with_target : pd.DataFrame
            A DataFrame containing the features and target labels for the candidates.
        """
        fit_kwargs = self.prepare_fit_kwargs(candidates_with_target)
        self.model.fit(**fit_kwargs)

    def predict_scores(self, candidates: pd.DataFrame) -> pd.Series:
        """
        Predict scores for the provided candidates using the fitted model.

        Parameters
        ----------
        candidates : pd.DataFrame
            A DataFrame containing the features for the candidates.

        Returns
        -------
        pd.Series
            A series containing the predicted scores for each candidate. If the model is a classifier, the scores
            represent probabilities for the positive class.
        """
        x_full = candidates.drop(columns=Columns.UserItem)

        if isinstance(self.model, ClassifierBase):
            return self.model.predict_proba(x_full)[:, 1]

        return self.model.predict(x_full)

    @classmethod
    def recommend(cls, scored_pairs: pd.DataFrame, k: int, add_rank_col: bool = True) -> pd.DataFrame:
        """
        Generate top-k recommendations for each user based on the provided scores.

        Parameters
        ----------
        scored_pairs : pd.DataFrame
            A DataFrame containing user-item pairs with associated scores.
            The DataFrame must have columns `Columns.User` and `Columns.Score`.
        k : int
            The number of top items to recommend for each user.
        add_rank_col : bool, default ``True``
            Whether to add a rank column to the resulting DataFrame, indicating the rank
            of each item within the user's recommendations.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the top-k recommended items for each user. If `add_rank_col` is True, the DataFrame
            will include an additional column `Columns.Score` for the rank of each item.
        """
        # TODO: optimize computations
        # Discussion here: https://github.com/MobileTeleSystems/RecTools/pull/209
        # Branch here: https://github.com/blondered/RecTools/tree/feature/polars
        reco = (
            scored_pairs.groupby(Columns.User, sort=False)
            .apply(lambda x: x.sort_values([Columns.Score], ascending=False).head(k))
            .reset_index(drop=True)
        )

        if add_rank_col:
            reco[Columns.Rank] = reco.groupby(Columns.User, sort=False).cumcount() + 1

        return reco


class CandidateFeatureCollector:
    """
    Base class for collecting features for candidates user-item pairs. Useful for creating train with features for
    CandidateRankingModel.
    Using this in CandidateRankingModel will result in not adding any features at all.
    Inherit from this class and rewrite private methods to grab features from dataset and external sources
    """

    # TODO: this class can be used in pipelines directly. It will keep scores and ranks and add nothing
    # TODO: create an inherited class that will get all features from dataset?

    def _get_user_features(
        self, users: ExternalIds, dataset: Dataset, fold_info: tp.Optional[tp.Dict[str, tp.Any]]
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=[Columns.User])

    def _get_item_features(
        self, items: ExternalIds, dataset: Dataset, fold_info: tp.Optional[tp.Dict[str, tp.Any]]
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=[Columns.Item])

    def _get_user_item_features(
        self, useritem: pd.DataFrame, dataset: Dataset, fold_info: tp.Optional[tp.Dict[str, tp.Any]]
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=Columns.UserItem)

    def collect_features(
        self, useritem: pd.DataFrame, dataset: Dataset, fold_info: tp.Optional[tp.Dict[str, tp.Any]]
    ) -> pd.DataFrame:
        """
        Collect features for users-item pairs from any desired sources.

        Parameters
        ----------
        useritem : pd.DataFrame
            Candidates with score/rank features from first stage. Ids are either external or 1x internal
        dataset : Dataset
            Dataset will have either external -> 2x internal id maps to internal -> 2x internal.
        fold_info : dict(str -> any), optional, default ``None``
            Fold info from splitter can be used for adding time-based features.

        Returns
        -------
        pd.DataFrame
            `useritem` dataframe enriched with features for users, items and useritem pairs.
        """
        user_features = self._get_user_features(useritem[Columns.User].unique(), dataset, fold_info)
        item_features = self._get_item_features(useritem[Columns.Item].unique(), dataset, fold_info)
        useritem_features = self._get_user_item_features(useritem, dataset, fold_info)

        res = (
            useritem.merge(user_features, on=Columns.User, how="left")
            .merge(item_features, on=Columns.Item, how="left")
            .merge(useritem_features, on=Columns.UserItem, how="left")
        )
        return res


class NegativeSamplerBase:
    """A base class for negative sampling."""

    def sample_negatives(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Sample negative examples from the given training data.

        Parameters
        ----------
        train : pd.DataFrame
            A DataFrame containing the training data from which negative examples will be sampled.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sampled negative examples.
        """
        raise NotImplementedError()


class PerUserNegativeSampler(NegativeSamplerBase):
    """
    A negative sampler that samples a specified number of negative examples per user from the training data.
    This class implements a per-user negative sampling strategy, where a fixed number of negative examples are
    randomly selected for each user.
    """

    def __init__(
        self,
        n_negatives: int = 3,
        random_state: tp.Optional[int] = None,
    ):
        """
        Initialize the PerUserNegativeSampler with `n_negatives` and `random_state`.

        Parameters
        ----------
        n_negatives : int, default ``3``
            The number of negative examples to sample for each user.
        random_state : int, optional, default ``None``
            An optional random seed for reproducibility of the sampling process.
        """
        self.n_negatives = n_negatives
        self.random_state = random_state

    def sample_negatives(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Sample negative examples from the given training data for each user.

        Parameters
        ----------
        train : pd.DataFrame
            A DataFrame containing the training data with user-item interactions.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sampled training data, which includes the specified number of negative
            examples per user along with all positive examples. The resulting DataFrame is shuffled.
        """
        # train: user_id, item_id, scores, ranks, target(1/0)

        # TODO: refactor for faster computations: avoid shuffle and apply
        # https://github.com/MobileTeleSystems/RecTools/pull/209#discussion_r1842977064

        negative_mask = train[Columns.Target] == 0
        pos = train[~negative_mask]
        neg = train[negative_mask]

        # Some users might not have enough negatives for sampling
        num_negatives = neg.groupby([Columns.User])[Columns.Item].count()
        sampling_mask = train[Columns.User].isin(num_negatives[num_negatives > self.n_negatives].index)

        neg_for_sample = train[sampling_mask & negative_mask]
        neg = neg_for_sample.groupby([Columns.User], sort=False).apply(
            pd.DataFrame.sample,
            n=self.n_negatives,
            replace=False,
            random_state=self.random_state,
        )
        neg = pd.concat([neg, train[(~sampling_mask) & negative_mask]], axis=0)
        sampled_train = pd.concat([neg, pos], ignore_index=True).sample(frac=1, random_state=self.random_state)

        return sampled_train


class CandidateGenerator:
    """
    A class responsible for generating recommendation candidates using a specified model. The generator
    can be configured to retain or discard ranks and scores, and it supports both training and recommendation
    modes.
    """

    def __init__(
        self,
        model: ModelBase,
        num_candidates: int,
        keep_ranks: bool,
        keep_scores: bool,
        scores_fillna_value: tp.Optional[float] = None,
        ranks_fillna_value: tp.Optional[float] = None,
    ):
        """
        Initialize the CandidateGenerator with model, num_candidates, keep_ranks, keep_scores,
        scores_fillna_value and ranks_fillna_value.

        Parameters
        ----------
        model : ModelBase
            The model used for generating recommendation candidates.
        num_candidates : int
            The number of candidates to generate for each user.
        keep_ranks : bool
            Whether to include rank information in the generated candidates.
        keep_scores : bool
            Whether to include score information in the generated candidates.
        scores_fillna_value : float, optional, default ``None``
            The value to fill missing scores with, if any. If None, missing scores are not filled.
        ranks_fillna_value : float, optional, default ``None``
            The value to fill missing ranks with, if any. If None, missing ranks are not filled.
        """
        self.model = model
        self.num_candidates = num_candidates
        self.keep_ranks = keep_ranks
        self.keep_scores = keep_scores
        self.scores_fillna_value = scores_fillna_value
        self.ranks_fillna_value = ranks_fillna_value
        self.is_fitted_for_train = False
        self.is_fitted_for_recommend = False

    def fit(self, dataset: Dataset, for_train: bool) -> None:
        """
        Fit the model using the provided dataset, configuring the generator for either training or recommendation.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model with. This should contain the necessary data for training or recommending.
        for_train : bool
            If True, configure the generator for training; otherwise, configure it for recommendation.
        """
        self.model.fit(dataset)
        if for_train:
            self.is_fitted_for_train = True  # TODO: keep multiple fitted instances?
            self.is_fitted_for_recommend = False
        else:
            self.is_fitted_for_train = False
            self.is_fitted_for_recommend = True

    def generate_candidates(
        self,
        users: ExternalIds,
        dataset: Dataset,
        filter_viewed: bool,
        for_train: bool,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        on_unsupported_targets: ErrorBehaviour = "raise",
    ) -> pd.DataFrame:
        """
        Generate candidates for recommendations.

        Parameters
        ----------
        users : ExternalIds
            The users for whom to generate recommendation candidates.
        dataset : Dataset
            The dataset containing user-item interactions and additional data needed for recommendation.
        filter_viewed : bool
            Whether to filter out items that have already been viewed by the user.
        for_train : bool
            Whether the candidates are being generated for training purposes.
        items_to_recommend : ExternalIds, optional, default ``None``
            Specific items to recommend. If None, recommend from all available items.
        on_unsupported_targets : ErrorBehaviour, default ``"raise"``
            Behavior when encountering unsupported targets. Can be "raise" to raise an error.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the generated recommendation candidates.
        """
        if for_train and not self.is_fitted_for_train:
            raise NotFittedForStageError(self.model.__class__.__name__, "train")
        if not for_train and not self.is_fitted_for_recommend:
            raise NotFittedForStageError(self.model.__class__.__name__, "recommend")

        candidates = self.model.recommend(
            users=users,
            dataset=dataset,
            k=self.num_candidates,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
            add_rank_col=self.keep_ranks,
            on_unsupported_targets=on_unsupported_targets,
        )
        if not self.keep_scores:
            candidates.drop(columns=Columns.Score, inplace=True)
        return candidates


class CandidateRankingModel(ModelBase):
    """Candidate Ranking Model for recommendation systems."""

    def __init__(
        self,
        candidate_generators: tp.List[CandidateGenerator],
        splitter: Splitter,
        reranker: Reranker,
        sampler: NegativeSamplerBase = PerUserNegativeSampler(),
        feature_collector: CandidateFeatureCollector = CandidateFeatureCollector(),
        verbose: int = 0,
    ) -> None:
        """
        Initialize the CandidateRankingModel with candidate generators, splitter, reranker, sampler
        and feature collector.

        Parameters
        ----------
        candidate_generators : list(CandidateGenerator)
            List of candidate generators.
        splitter : Splitter
            Splitter for dataset splitting.
        reranker : Reranker
            Reranker for reranking candidates.
        sampler : NegativeSamplerBase, default ``PerUserNegativeSampler()``
            Sampler for negative sampling.
        feature_collector : CandidateFeatureCollector, default ``CandidateFeatureCollector()``
            Collector for user-item features.
        verbose : int, default ``0``
            Verbosity level.
        """
        super().__init__(verbose=verbose)

        if hasattr(splitter, "n_splits"):
            assert splitter.n_splits == 1  # TODO: handle softly
        self.splitter = splitter
        self.sampler = sampler
        self.reranker = reranker
        self.cand_gen_dict = self._create_cand_gen_dict(candidate_generators)
        self.feature_collector = feature_collector

    def _create_cand_gen_dict(
        self, candidate_generators: tp.List[CandidateGenerator]
    ) -> tp.Dict[str, CandidateGenerator]:
        """
        Create a dictionary of candidate generators with unique identifiers.

        Parameters
        ----------
        candidate_generators : list(CandidateGenerator)
            List of candidate generators.

        Returns
        -------
        dict(str -> CandidateGenerator)
            Dictionary with candidate generator identifiers as keys and candidate generators as values.
        """
        model_count: tp.Dict[str, int] = defaultdict(int)
        cand_gen_dict = {}
        for candgen in candidate_generators:
            model_name = candgen.model.__class__.__name__
            model_count[model_name] += 1
            identifier = f"{model_name}_{model_count[model_name]}"
            cand_gen_dict[identifier] = candgen
        return cand_gen_dict

    def split_to_history_dataset_and_train_targets(
        self, dataset: Dataset, splitter: Splitter
    ) -> tp.Tuple[Dataset, pd.DataFrame, tp.Dict[str, tp.Any]]:
        """
        Split interactions into history and train sets for first-stage and second-stage model training.

        Parameters
        ----------
        dataset : Dataset
            The dataset to split.
        splitter : Splitter
            The splitter to use for splitting the dataset.

        Returns
        -------
        pd.DataFrame, pd.DataFrame, dict(str -> any)
            Tuple containing the history dataset, train targets, and fold information.
        """
        split_iterator = splitter.split(dataset.interactions, collect_fold_stats=True)

        train_ids, test_ids, fold_info = next(iter(split_iterator))  # splitter has only one fold

        history_dataset = dataset.filter_interactions(train_ids)
        interactions = dataset.get_raw_interactions()
        train_targets = interactions.iloc[test_ids]

        return history_dataset, train_targets, fold_info

    def _fit(self, dataset: Dataset, *args: tp.Any, refit_candidate_generators: bool = True, **kwargs: tp.Any) -> None:
        """
        Fits all first-stage models on history dataset
        Generates candidates
        Sets targets
        Samples negatives
        Collects features for candidates
        Trains reranker on prepared train
        Fits all first-stage models on full dataset
        """
        train_with_target = self.get_train_with_targets_for_reranker(dataset)
        self.reranker.fit(train_with_target, **kwargs)  # TODO: add a flag to keep user/item id features somewhere
        if refit_candidate_generators:
            self._fit_candidate_generators(dataset, for_train=False)

    def get_train_with_targets_for_reranker(self, dataset: Dataset) -> pd.DataFrame:
        """
        Prepare training data for the reranker.

        Parameters
        ----------
        dataset : Dataset
            The dataset to prepare training data from.

        Returns
        -------
        pd.DataFrame
            DataFrame containing training data with targets and 2 extra columns: `Columns.User`, `Columns.Item`.
        """
        history_dataset, train_targets, fold_info = self.split_to_history_dataset_and_train_targets(
            dataset, self.splitter
        )

        candidates = self.get_full_candidates_with_targets(train_targets, history_dataset)
        candidates = self.sampler.sample_negatives(candidates)

        train_with_target = self.feature_collector.collect_features(candidates, history_dataset, fold_info)

        return train_with_target

    def get_full_candidates_with_targets(self, train_targets: pd.DataFrame, history_dataset: Dataset) -> pd.DataFrame:
        """
        Prepare candidates with target values set from first-stage candidate generators.

        Parameters
        ----------
        train_targets : pd.DataFrame
            DataFrame containing training targets.
        history_dataset : Dataset
            The dataset to fit the candidate generators on.

        Returns
        -------
        pd.DataFrame
            DataFrame with target values set.
        """
        self._fit_candidate_generators(history_dataset, for_train=True)

        candidates = self._get_candidates_from_first_stage(
            users=train_targets[Columns.User].unique(),
            dataset=history_dataset,
            filter_viewed=self.splitter.filter_already_seen,  # TODO: think about it
            for_train=True,
        )
        candidates = self._set_targets_to_candidates(candidates, train_targets)
        return candidates

    def _set_targets_to_candidates(self, candidates: pd.DataFrame, train_targets: pd.DataFrame) -> pd.DataFrame:
        """
        Set target values to the candidate items.

        Parameters
        ----------
        candidates : pd.DataFrame
            DataFrame containing candidate items.
        train_targets : pd.DataFrame
            DataFrame containing training targets.

        Returns
        -------
        pd.DataFrame
            DataFrame with target values set.
        """
        train_targets[Columns.Target] = 1

        # Remember that this way we exclude positives that weren't present in candidates
        train = pd.merge(
            candidates,
            train_targets[[Columns.User, Columns.Item, Columns.Target]],
            how="left",
            on=Columns.UserItem,
        )

        train[Columns.Target] = train[Columns.Target].fillna(0).astype("int32")
        return train

    def _fit_candidate_generators(self, dataset: Dataset, for_train: bool) -> None:
        """
        Fit the first-stage candidate generators on the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the candidate generators on.
        for_train : bool
            Whether the fitting is for training or not.
        """
        for candgen in self.cand_gen_dict.values():
            candgen.fit(dataset, for_train)

    def _get_candidates_from_first_stage(
        self,
        users: ExternalIds,
        dataset: Dataset,
        filter_viewed: bool,
        for_train: bool,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        on_unsupported_targets: ErrorBehaviour = "raise",
    ) -> pd.DataFrame:
        """
        Get candidates from the first-stage models.

        Parameters
        ----------
        users : ExternalIds
            List of user IDs to get candidates for.
        dataset : Dataset
            The dataset to get candidates from.
        filter_viewed : bool
            Whether to filter already viewed items.
        for_train : bool
            Whether the candidates are for training or not.
        items_to_recommend : ExternalIds, optional, default ``None``
            List of items to recommend.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the candidates.
        """
        candidates_dfs = []

        for identifier, candgen in self.cand_gen_dict.items():
            candidates = candgen.generate_candidates(
                users=users,
                dataset=dataset,
                filter_viewed=filter_viewed,
                for_train=for_train,
                items_to_recommend=items_to_recommend,
                on_unsupported_targets=on_unsupported_targets,
            )

            # Process ranks and scores as features
            rank_col_name, score_col_name = f"{identifier}_rank", f"{identifier}_score"

            candidates.rename(
                columns={Columns.Rank: rank_col_name, Columns.Score: score_col_name},
                inplace=True,
            )
            candidates_dfs.append(candidates)

        # Merge all candidates together and process missing ranks and scores
        all_candidates = reduce(lambda a, b: a.merge(b, how="outer", on=Columns.UserItem), candidates_dfs)
        first_stage_results = self._process_ranks_and_scores(all_candidates)

        return first_stage_results

    def _process_ranks_and_scores(
        self,
        all_candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process ranks and scores of the candidates.

        Parameters
        ----------
        all_candidates : pd.DataFrame
            DataFrame containing all candidates.

        Returns
        -------
        pd.DataFrame
            DataFrame with processed ranks and scores.
        """
        for identifier, candgen in self.cand_gen_dict.items():
            rank_col_name, score_col_name = f"{identifier}_rank", f"{identifier}_score"
            if candgen.keep_ranks and candgen.ranks_fillna_value is not None:
                all_candidates[rank_col_name] = all_candidates[rank_col_name].fillna(candgen.ranks_fillna_value)
            if candgen.keep_scores and candgen.scores_fillna_value is not None:
                all_candidates[score_col_name] = all_candidates[score_col_name].fillna(candgen.scores_fillna_value)

        return all_candidates

    def recommend(
        self,
        users: ExternalIds,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        add_rank_col: bool = True,
        on_unsupported_targets: ErrorBehaviour = "raise",
        force_fit_candidate_generators: bool = False,
    ) -> pd.DataFrame:
        """
        Generate k recommendations for specified users using the dataset.

        Parameters
        ----------
        users : ExternalIds
            List of user ids for whom recommendations are generated.
        dataset : Dataset
            Dataset containing user-item interaction data and possibly additional features.
        k : int
            The number of recommendations to generate for each user.
        filter_viewed : bool
            If true, viewed items will be excluded from the recommendations.
        items_to_recommend : ExternalIds, optional, default ``None``
            List of item ids from which recommendations should be generated.
            If not provided, it will include all items available in the dataset.
        add_rank_col : bool, default ``True``
            If true, a rank column is added to the returned DataFrame.
            The rank column shows the position of the item in the sorted order of predictions.
        on_unsupported_targets : ErrorBehaviour, default ``"raise"``
            Controls the behavior when a target is encountered during prediction,
            for which the Model makes no prediction.
            If "raise", a ValueError is raised. If "warn", it outputs a warning,
            and if "ignore", it silently continues.
        force_fit_candidate_generators : bool, default ``False``
            If true, the candidate generators are fitted even if they are already fitted.

        Returns
        -------
        pd.DataFrame
            DataFrame with the recommended items for users.
        """
        self._check_is_fitted()
        self._check_k(k)

        if force_fit_candidate_generators or not all(
            generator.is_fitted_for_recommend for generator in self.cand_gen_dict.values()
        ):
            self._fit_candidate_generators(dataset, for_train=False)

        candidates = self._get_candidates_from_first_stage(
            users=users,
            dataset=dataset,
            filter_viewed=filter_viewed,
            items_to_recommend=items_to_recommend,
            for_train=False,
            on_unsupported_targets=on_unsupported_targets,
        )

        train = self.feature_collector.collect_features(candidates, dataset, fold_info=None)

        scored_pairs = candidates.reindex(columns=Columns.UserItem)
        scored_pairs[Columns.Score] = self.reranker.predict_scores(train)

        return self.reranker.recommend(scored_pairs, k=k, add_rank_col=add_rank_col)
