import typing as tp
from abc import ABC, abstractmethod

import torch
from torch import nn


class QuantizerOutput(tp.NamedTuple):
    """Quantizer outputs."""

    sem_ids: tp.List[tp.Tuple[int, ...]]
    loss: torch.Tensor


class Quantizer(ABC, nn.Module):
    """Base class for quantizers like RQ-VAE and RK-Means."""

    def __init__(self, input_dim: int, codebook_sizes: tp.List[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook_sizes = codebook_sizes
        self.codebooks = nn.ModuleList()

    @abstractmethod
    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor) -> None:
        """Initialize codebooks for the quantizer.

        Parameters
        ----------
        data : torch.Tensor
            Data to initialize codebooks with
        """

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> QuantizerOutput:
        """Quantizer forward pass."""
