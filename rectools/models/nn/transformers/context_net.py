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

    def __init__(self, n_factors: int, dropout_rate: float, n_cat_feature_values: int, **kwargs: tp.Any) -> None:
        super().__init__(n_factors, dropout_rate, **kwargs)
        print(n_cat_feature_values)
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=n_cat_feature_values, embedding_dim=n_factors, mode="sum")
        self.dropout = nn.Dropout(dropout_rate)

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
        # TODO: check correctness and remove offsets from batch
        b, l, f = seqs.shape
        offsets = batch["context_cat_offsets"].view(-1)
        offsets = torch.cat([torch.zeros(1, dtype=offsets.dtype, device=offsets.device), offsets])
        offsets = offsets.cumsum(dim=0)[:-1]

        inputs = batch["context_cat_inputs"]
        new_inputs = inputs.view(b * l, -1)
        context_embs = self.embedding_bag(input=new_inputs)
        context_embs = self.dropout(context_embs)
        context_embs = context_embs.view(b, l, f)
        return seqs + context_embs

    @property
    def out_dim(self) -> int:
        """Return categorical item embedding output dimension."""
        return self.embedding_bag.embedding_dim
