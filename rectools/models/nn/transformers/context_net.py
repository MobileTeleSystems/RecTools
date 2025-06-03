import typing as tp
import warnings

import torch
import typing_extensions as tpe
from torch import nn

from rectools.dataset.dataset import DatasetSchema

# TODO: support non-string values in feature names/values


class ContextNetBase(torch.nn.Module):
    """TODO."""

    def __init__(self, n_factors: int, dropout_rate: float, **kwargs: tp.Any):
        super().__init__()

    def forward(self, seqs: torch.Tensor, batch: tp.Dict[str, torch.Tensor]) -> torch.Tensor:
        """TODO."""
        raise NotImplementedError

    @classmethod
    def from_dataset_schema(
        cls, dataset_schema: DatasetSchema, *args: tp.Any, **kwargs: tp.Any
    ) -> tp.Optional[tpe.Self]:
        """Construct ItemNet from Dataset schema."""
        raise NotImplementedError()

    @property
    def out_dim(self) -> int:
        """Return item embedding output dimension."""
        raise NotImplementedError()


class CatFeaturesContextNet(ContextNetBase):
    """TODO."""

    def __init__(
        self,
        n_factors: int,
        dropout_rate: float,
        n_cat_feature_values: int,
        batch_key: str = "context_cat_inputs",
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(n_factors, dropout_rate, **kwargs)
        print(n_cat_feature_values)
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=n_cat_feature_values, embedding_dim=n_factors, mode="sum")
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_key = batch_key

    @classmethod
    def from_dataset_schema(  # TODO: decide about target aware schema
        cls, dataset_schema: DatasetSchema, n_factors: int, dropout_rate: float, **kwargs: tp.Any
    ) -> tp.Optional[tpe.Self]:
        """TODO."""
        if dataset_schema.interactions is None:
            warnings.warn("No interactions schema found in dataset schema, context net will not be constructed")
            return None
        if dataset_schema.interactions.direct_feature_names:
            warnings.warn("Direct features are not supported in context net")
        if len(dataset_schema.interactions.cat_feature_names_w_values) == 0:
            warnings.warn("No categorical features found in dataset schema, context net will not be constructed")
            return None
        n_cat_feature_values = len(dataset_schema.interactions.cat_feature_names_w_values)
        return cls(n_factors=n_factors, dropout_rate=dropout_rate, n_cat_feature_values=n_cat_feature_values)

    def forward(self, seqs: torch.Tensor, batch: tp.Dict[str, torch.Tensor]) -> torch.Tensor:
        """TODO."""
        batch_size, session_max_len, n_factors = seqs.shape
        inputs = batch[self.batch_key].view(batch_size * session_max_len, -1)
        context_embs = self.embedding_bag(input=inputs)
        context_embs = self.dropout(context_embs)
        context_embs = context_embs.view(batch_size, session_max_len, n_factors)
        return context_embs

    @property
    def out_dim(self) -> int:
        """Return output dimension."""
        return self.embedding_bag.embedding_dim
