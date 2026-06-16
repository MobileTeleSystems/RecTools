import typing as tp

from torch import Tensor, nn


class MLP(nn.Module):
    """An implementation of multilayer perceptron.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    hidden_dims : tp.List[int]
        Dimensions of hidden layers.
    out_dim : int
        Output dimension.
    dropout : float, optional
        Dropout probability, by default 0.0
    normalize : bool, optional
        Whether to apply batch normalization, by default False
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tp.List[int],
        out_dim: int,
        dropout: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout
        self.normalize = normalize

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]

        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))

            if self.normalize:
                self.mlp.append(nn.BatchNorm1d(num_features=out_d))

            if i != len(dims) - 2:
                self.mlp.append(nn.ReLU())

            if dropout != 0:
                self.mlp.append(nn.Dropout(dropout))

        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass."""
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)

    # We just initialize the module with normal distribution as the paper said
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
