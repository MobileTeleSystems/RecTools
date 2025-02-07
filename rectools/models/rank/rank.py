import typing as tp
from enum import Enum

from scipy import sparse

from rectools import InternalIds
from rectools.models.base import Scores
from rectools.types import InternalIdsArray


class Distance(Enum):
    """Distance metric"""

    DOT = 1  # Bigger value means closer vectors
    COSINE = 2  # Bigger value means closer vectors
    EUCLIDEAN = 3  # Smaller value means closer vectors


class Ranker(tp.Protocol):
    """Protocol for all rankers"""

    def rank(
        self,
        subject_ids: InternalIds,
        k: tp.Optional[int] = None,
        filter_pairs_csr: tp.Optional[sparse.csr_matrix] = None,
        sorted_object_whitelist: tp.Optional[InternalIdsArray] = None,
    ) -> tp.Tuple[InternalIds, InternalIds, Scores]:  # pragma: no cover
        """Rank objects by corresponding embeddings.

        Parameters
        ----------
        subject_ids : InternalIds
            Array of ids to recommend for.
        k : int, optional, default ``None``
            Derived number of recommendations for every subject id.
            Return all recs if None.
        filter_pairs_csr : sparse.csr_matrix, optional, default ``None``
            Subject-object interactions that should be filtered from recommendations.
            This is relevant for u2i case.
        sorted_object_whitelist : sparse.csr_matrix, optional, default ``None``
            Whitelist of object ids.
            If given, only these items will be used for recommendations.
            Otherwise all items from dataset will be used.

        Returns
        -------
        (InternalIds, InternalIds, Scores)
            Array of subject ids, array of recommended items, sorted by score descending and array of scores.
        """
