#  Copyright 2025 MTS (Mobile Telesystems)
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

import typing as tp

import torch


class TransformerNegativeSamplerBase:
    """Base class for negative sampler. To negative sampling inherit
    from this class and pass your custom data preparator to your model parameters.

    Parameters
    ----------
    session_max_len : int
        Maximum length of user sequence.
    n_negatives : int
        Number of negatives.
    """

    def __init__(
        self,
        n_negatives: int,
        **kwargs: tp.Any,
    ) -> None:
        self.n_negatives = n_negatives

    def get_negatives(self, batch_dict: tp.Dict, n_item_extra_tokens: int, n_items: int) -> torch.Tensor:
        """Return sampled negatives."""
        raise NotImplementedError()


class CatalogUniformSampler(TransformerNegativeSamplerBase):
    """Class to sample negatives uniformly from all catalog items."""

    def get_negatives(self, batch_dict: tp.Dict, n_item_extra_tokens: int, n_items: int) -> torch.Tensor:
        """Return sampled negatives."""
        negatives = torch.randint(
            low=n_item_extra_tokens,
            high=n_items,
            size=(batch_dict["x"].shape[0], batch_dict["x"].shape[1], self.n_negatives),
        )  # [batch_size, session_max_len, n_negatives]
        return negatives
