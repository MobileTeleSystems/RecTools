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
    """Base class for negative sampler. To create custom sampling logic inherit
    from this class and pass your custom negative sampler to your model parameters.

    Parameters
    ----------
    n_negatives : int
        Number of negatives.
    """

    def __init__(
        self,
        n_negatives: int,
        **kwargs: tp.Any,
    ) -> None:
        self.n_negatives = n_negatives

    def get_negatives(
        self,
        batch_dict: tp.Dict,
        lowest_id: int,
        highest_id: int,
        session_len_limit: tp.Optional[int] = None,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Return sampled negatives."""
        raise NotImplementedError()


class CatalogUniformSampler(TransformerNegativeSamplerBase):
    """Class to sample negatives uniformly from all catalog items.

    Parameters
    ----------
    n_negatives : int
        Number of negatives.
    """

    def get_negatives(
        self,
        batch_dict: tp.Dict,
        lowest_id: int,
        highest_id: int,
        session_len_limit: tp.Optional[int] = None,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        """Return sampled negatives."""
        session_len = session_len_limit if session_len_limit is not None else batch_dict["x"].shape[1]
        negatives = torch.randint(
            low=lowest_id,
            high=highest_id,
            size=(batch_dict["x"].shape[0], session_len, self.n_negatives),
        )  # [batch_size, session_max_len, n_negatives]
        return negatives
