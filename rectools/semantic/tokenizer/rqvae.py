import typing as tp

import torch
from k_means_constrained import KMeansConstrained
from torch import nn
from torch.nn.functional import mse_loss
from tqdm.auto import tqdm

from rectools.semantic.modules import MLP

from .base_quantizer import Quantizer, QuantizerOutput
from .loss import CodebookLoss


def _get_pca_projection(embeddings: torch.Tensor, out_dim: int) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0)
    _, _, vh = torch.linalg.svd(centered, full_matrices=True)  # pylint: disable=not-callable
    return vh[:out_dim].T.contiguous()


class WhiteningAdapter(nn.Module):
    r"""Applies an adapter consisting of PCA projection and an optional linear transformation to incoming item
    embeddings.

    The adapter itself is a linear transformation that modifies input embedddings `x` in the following manner:

    .. math::
        \text{out}(x) = W \times (x - b)

    Where :math:`W` is weights matrix initialized as principal component matrix and :math:`b` is a vector of
    biases initialized as mean of embedding.

    Parameters
    ----------
    adapter_type : tp.Literal[&quot;ffn&quot;, &quot;identity&quot;], optional
        Transformation applied after PCA projection, by default "identity"
    emb_dim : int, optional
        Dimension of incoming embeddings, by default 768
    proj_dim : int, optional
        How many principal components to use, by default 512
    hidden_units : int, optional
        Size of hidden layer in linear adapter, by default 256
    dropout : float, optional
        Dropout probability for linear adapter, by default 0.1
    device : str, optional
        Device the adapter is on, by default "cpu"
    """

    def __init__(
        self,
        adapter_type: tp.Literal["ffn", "identity"] = "identity",
        emb_dim: int = 768,
        proj_dim: int = 512,
        hidden_units: int = 256,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.adapter_type = adapter_type
        self.hidden_units = hidden_units
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.device = device
        self.weight = nn.Parameter(torch.empty(self.emb_dim, self.proj_dim))
        self.bias = nn.Parameter(torch.empty(self.emb_dim))

        self.head: nn.Module
        if self.adapter_type == "ffn":
            self.head = MLP(self.proj_dim, [self.hidden_units], self.proj_dim, dropout=self.dropout)
        else:
            self.head = nn.Identity()

    def freeze_adapter(self) -> None:
        """Freeze PCA transform's weights and biases."""
        self.bias.requires_grad = False
        self.weight.requires_grad = False

    @torch.no_grad()
    def init_from_embeds(self, embeddings: torch.Tensor) -> None:
        """Initialize PCA projection weights using item metadata embeddigns.

        Parameters
        ----------
        embeddings : torch.Tensor
            Item metadata embeddings
        """
        proj = _get_pca_projection(embeddings, self.proj_dim)

        self.bias.data.copy_(embeddings.mean(dim=0))
        self.weight.data.copy_(proj)
        self.freeze_adapter()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Adapter forward pass."""
        # PCA projection is precision-sensitive (1024-dim matmul accumulates
        # bf16 rounding errors). Force fp32 even inside autocast.
        with torch.amp.autocast(self.device, enabled=False):
            projected = (inputs.float() - self.bias.float()) @ self.weight.float()
        return self.head(projected)


class Codebook(nn.Module):
    """An RQ-VAE codebook. Takes in RQ-VAE encoder outputs or their residuals from previous codebooks and outputs
    codes corresponding to embeddings closest to input residuals.

    Parameters
    ----------
    code_dim : int
        Size of codebook embeddings.
    n_codes : int
        Number of codebook embeddings.
    """

    def __init__(
        self,
        code_dim: int,
        n_codes: int,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.n_codes = n_codes
        self.code_embs = nn.Embedding(n_codes, code_dim)
        self.cb_loss = CodebookLoss()
        self.kmeans_initted_ = False

    def init_from_centroids(self, centroids: torch.Tensor) -> None:
        """Initialize codebook using predetermined centroids as embeddings.

        Parameters
        ----------
        centroids : torch.Tensor
            precomputed centroids of shape [n_codes, emb_dim]
        """
        assert centroids.shape == (self.n_codes, self.code_dim)
        with torch.no_grad():
            self.code_embs.weight.data.copy_(centroids)
        self.kmeans_initted_ = True

    def forward(self, inputs: torch.Tensor) -> tp.Tuple:
        """Codebook forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input residuals expected to be of shape [B, emb_dim]

        Returns
        -------
        tp.Tuple
            Codes corresponding to embeddings closest to input residuals, embeddings themselves, resuduals,
            and codebook loss.
        """
        # inputs: [B, emb_dim] -> [B, 1, emb_dim]
        # code_embs: [cb_size, emb_dim] -> [1, cb_size, emb_dim]
        # This maps to diffs: [B, cb_size, emb_dim]
        diffs = inputs.unsqueeze(1) - self.code_embs.weight.unsqueeze(0)
        dists = torch.linalg.vector_norm(diffs, dim=-1)  # pylint: disable=not-callable
        codes = dists.argmin(dim=-1)  # out: [B]

        quantized = self.code_embs(codes)
        residuals = inputs - quantized

        # Straight-through estimator: forward value is quantized,
        # but gradients pass through to the encoder as if quantized == inputs.
        quantized_st = inputs + (quantized - inputs).detach()

        loss = self.cb_loss(inputs, quantized)

        return codes, quantized_st, residuals, loss


class RQVAE(Quantizer):
    """Apply Residual-Quantized Variational AutoEncoder (RQ-VAE) -- a multi-level vector quantizer
    that applies quantization on residuals to generate a tuple of codewords (aka Semantic IDs).
    The Autoencoder is jointly trained by updating the quantization codebook and the DNN
    encoder-decoder parameters.

    Parameters
    ----------
    input_dim : int
        Input embeddings size.
    codebook_dim : int
        Size of codebook embeddings.
    hidden_dims : tp.List[int]
        Encoder and decoder hidden layer sizes.
    codebook_sizes : tp.List[int]
        Number of embeddings each codebook contains.
    adapter_type : tp.Literal[&quot;ffn&quot;, &quot;identity&quot;], optional
        Transformation applied after PCA projection, by default "identity"
    adapter_proj_dim : int, optional
        How many principal components to use, by default 512
    """

    def __init__(
        self,
        input_dim: int,
        codebook_dim: int,
        hidden_dims: tp.List[int],
        codebook_sizes: tp.List[int],
        adapter_type: tp.Literal["ffn", "identity"] = "identity",
        adapter_proj_dim: int = 512,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            codebook_sizes=codebook_sizes,
        )
        self.codebook_dim = codebook_dim
        for n_codes in codebook_sizes:
            self.codebooks.append(Codebook(codebook_dim, n_codes))

        self.adapter_type = adapter_type
        self.adapter_proj_dim = adapter_proj_dim
        self.adapter = WhiteningAdapter(
            adapter_type=adapter_type,
            emb_dim=input_dim,
            proj_dim=adapter_proj_dim,
        )

        self.encoder = MLP(adapter_proj_dim, hidden_dims, codebook_dim)
        self.decoder = MLP(codebook_dim, hidden_dims[::-1], adapter_proj_dim)

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor) -> None:
        """Initialize codebook centroids using constrained k-means.

        Runs the encoder on `data`, then for each codebook level fits
        KMeansConstrained (with size_min per cluster) on the residuals
        and copies the resulting centroids into the codebook embeddings.

        Parameters
        ----------
        data : torch.Tensor
            Item metadata embeddings.
        """
        self.adapter.init_from_embeds(data)
        embs = self.adapter(data)
        encoded = self.encoder(embs).cpu().numpy()
        residuals = encoded

        for layer in tqdm(self.codebooks, desc="Initializing codebooks with constrained k-means"):
            size_min = min(len(data) // (layer.n_codes * 2), 50)
            size_max = size_min * 4 if layer.n_codes * size_min * 4 > len(data) else len(data)
            km = KMeansConstrained(
                n_clusters=layer.n_codes,
                size_min=size_min,
                size_max=size_max,
                random_state=0,
                max_iter=10,
                n_init=10,
                n_jobs=10,
                verbose=False,
            )
            km.fit(residuals)
            centroids = torch.from_numpy(km.cluster_centers_).to(data.device)
            layer.init_from_centroids(centroids)

            # Compute residuals for the next level
            assignments = km.labels_
            residuals = residuals - km.cluster_centers_[assignments]

    def forward(self, inputs: torch.Tensor) -> QuantizerOutput:
        """RQ-VAE forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings

        Returns
        -------
        tp.Tuple
            Semantic IDs corresponding to input embeddings, RQ-VAE loss (codebook losses +
            reconstruction loss).
        """
        inputs_adapted = self.adapter(inputs)
        inputs_enc = self.encoder(inputs_adapted)

        loss = torch.tensor(0.0, device=next(self.codebooks.parameters()).device)

        res_prev = inputs_enc
        embeds, sem_ids = [], []
        for layer in self.codebooks:
            codes, quantized_st, res, loss_upd = layer(res_prev)
            res_prev = res
            loss += loss_upd
            embeds.append(quantized_st)
            sem_ids.append(codes)

        embeds_tensor = torch.stack(embeds, dim=1).sum(dim=1)
        sem_ids_tensor = torch.stack(sem_ids, dim=1).to(torch.int32)

        # Convert semids from tensors to tuples of ints
        sem_ids = [
            tuple(sem_ids_tensor[idx, :].flatten().detach().cpu().tolist()) for idx in range(sem_ids_tensor.size(0))
        ]

        embeds_dec = self.decoder(embeds_tensor)

        # Reconstruction loss
        loss += mse_loss(inputs_adapted, embeds_dec, reduction="mean")

        return QuantizerOutput(sem_ids=sem_ids, loss=loss)
