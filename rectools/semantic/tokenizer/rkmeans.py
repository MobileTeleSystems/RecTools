import typing as tp

import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from torch import nn
from torch.nn.functional import mse_loss, normalize
from tqdm.auto import tqdm

from .base_quantizer import Quantizer, QuantizerOutput


class Codebook(nn.Module):
    """An RK-Means codebook. Takes in item embeddings or their residuals from previous codebooks and outputs
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
        """Run a forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input residuals.

        Returns
        -------
        tp.Tuple
            Corresponding codes, quantized reprezentations, residuals between quantized
            representations and inputs.
        """
        diffs = inputs.unsqueeze(1) - self.code_embs.weight.unsqueeze(0)
        dists = torch.linalg.vector_norm(diffs, dim=-1)  # pylint: disable=not-callable
        codes = dists.argmin(dim=-1)

        quantized = self.code_embs.weight[codes, :]
        residuals = inputs - quantized

        # Straight-through estimator: forward value is quantized,
        # but gradients pass through to the encoder as if quantized == inputs.
        quantized = inputs + (quantized - inputs).detach()

        return codes, quantized, residuals


class RKmeans(Quantizer):
    """Apply Residual Mini-Batch K-Means (RK-Means) -- a multi-level vector quantizer
    that applies quantization on residuals to generate a tuple of codewords (aka Semantic IDs).

    Parameters
    ----------
    input_dim : int
        Input embedding dimension.
    codebook_sizes : tp.List[int]
        Number of embeddings contained in RK-Means codebooks.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_sizes: tp.List[int],
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            codebook_sizes=codebook_sizes,
        )
        for n_codes in codebook_sizes:
            self.codebooks.append(Codebook(self.input_dim, n_codes))

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor) -> None:
        """Initialize codebook centroids using constrained k-means.

        Runs the encoder on `data`, then for each codebook level fits
        KMeansConstrained (with size_min per cluster) on the residuals
        and copies the resulting centroids into the codebook embeddings.

        Parameters
        ----------
        data : torch.Tensor
            Input embeddings.
        """
        residuals = normalize(data).cpu().numpy()

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
            residuals = residuals / (np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-12)

    def forward(self, inputs: torch.Tensor) -> QuantizerOutput:
        """Run a forward pass

        Parameters
        ----------
        inputs : torch.Tensor
            Input item embeddings.

        Returns
        -------
        tp.Tuple
            Semantic IDs corresponding to input embeddings, reconstruction loss.
        """
        res_prev = normalize(inputs)
        embeds, sem_ids = [], []
        for layer in self.codebooks:
            codes, quantized, res = layer(res_prev)
            res_prev = res
            res_prev = normalize(res_prev)
            embeds.append(quantized)
            sem_ids.append(codes)

        # embeds: [B] x num_codebooks -> [B, num_codebooks] -> [B]
        embeds_tensor = torch.stack(embeds, dim=1).sum(dim=1)
        sem_ids_tensor = torch.stack(sem_ids, dim=1).to(torch.int32)

        # Convert semids from tensors to tuples of ints
        sem_ids = [
            tuple(sem_ids_tensor[idx, :].flatten().detach().cpu().tolist()) for idx in range(sem_ids_tensor.size(0))
        ]

        # Reconstruction loss
        loss = mse_loss(inputs, embeds_tensor, reduction="mean")
        return QuantizerOutput(sem_ids=sem_ids, loss=loss)
