import torch
from torch import nn
from torch.nn.functional import mse_loss


class CodebookLoss(nn.Module):
    """Creates a criterion that measures codebook quantization loss between residuals and codebook embeddings."""

    def __init__(self, mu: float = 1.0, beta: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.mu = mu
        self.reduction = reduction

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.mu * mse_loss(y_true.detach(), y_pred, reduction=self.reduction) + self.beta * mse_loss(
            y_true, y_pred.detach(), reduction=self.reduction
        )
