import json
import os
import typing as tp
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from scipy import sparse

WINDOW_AVG = 3


class RecallCallback(Callback):  # type: ignore
    """
    Callback for computing Recall@k metric during validation.

    Parameters
    ----------
    k : int
        Number of top recommendations to consider for Recall computation.
    progress_bar : bool, default=True
        Whether to show Recall in the progress bar during validation.
    """

    base_name: str = "recall"

    def __init__(self, k: int, progress_bar: bool = True) -> None:
        self.k = k
        self.name = self.base_name + f"@{k}"
        self.progress_bar = progress_bar

        self.batch_recall_per_users: tp.List[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: tp.Dict[str, torch.Tensor],
        batch: tp.Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Process a single validation batch to compute Recall@k per user.

        - Computes logits (if not present in outputs).
        - Filters known items from recommendations using a sparse mask.
        - Computes top-k recommendations.
        - Compares recommendations with ground truth targets to calculate Recall.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            The model being trained/evaluated.
        outputs : Dict[str, torch.Tensor]
            Model outputs, including logits or session embeddings.
        batch : Dict[str, torch.Tensor]
            Batch of validation data, including input sequences and targets.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, default=0
            Index of the dataloader (useful for multiple validation sets).
        """
        if "logits" not in outputs:
            session_embs = pl_module.torch_model.encode_sessions(batch, pl_module.item_embs)[:, -1, :]
            logits = pl_module.torch_model.similarity_module(session_embs, pl_module.item_embs)
        else:
            logits = outputs["logits"]

        x = batch["x"]
        users = x.shape[0]
        row_ind = np.arange(users).repeat(x.shape[1])
        col_ind = x.flatten().detach().cpu().numpy()
        mask = col_ind != 0
        data = np.ones_like(row_ind[mask])
        filter_csr = sparse.csr_matrix(
            (data, (row_ind[mask], col_ind[mask])),
            shape=(users, pl_module.torch_model.item_model.n_items),
        )
        mask = torch.from_numpy((filter_csr != 0).toarray()).to(logits.device)
        scores = torch.masked_fill(logits, mask, float("-inf"))

        _, batch_recos = scores.topk(k=self.k)

        targets = batch["y"]

        # assume all users have the same amount of TP
        liked = targets.shape[1]
        tp_mask = torch.stack([torch.isin(batch_recos[uid], targets[uid]) for uid in range(batch_recos.shape[0])])
        recall_per_users = tp_mask.sum(dim=1) / liked

        self.batch_recall_per_users.append(recall_per_users)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Aggregate Recall@k results from all batches and log them.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            The model being evaluated.
        """
        recall = float(torch.concat(self.batch_recall_per_users).mean())
        self.log_dict({self.name: recall}, on_step=False, on_epoch=True, prog_bar=self.progress_bar)

        self.batch_recall_per_users.clear()


class BestModelLoadCallback(Callback):
    """
    Callback for loading the best model checkpoint at the end of training.

    Parameters
    ----------
    ckpt_path : str
        Path to the best checkpoint file (without .ckpt extension).
    """

    def __init__(self, ckpt_path: str) -> None:
        self.ckpt_path = ckpt_path + ".ckpt"

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Load the best model weights from the checkpoint after training finishes.

        If `trainer.log_dir` is not set, a warning is raised and no weights are loaded.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer instance.
        pl_module : LightningModule
            The model being trained.
        """
        if trainer.log_dir is None:
            warnings.warn("Trainer has no log dir and weights were not updated from checkpoint")
            return
        log_dir: str = trainer.log_dir
        ckpt_path = Path(log_dir) / "checkpoints" / self.ckpt_path
        checkpoint = torch.load(ckpt_path, weights_only=False)
        pl_module.load_state_dict(checkpoint["state_dict"])
        self.ckpt_full_path = str(ckpt_path)  # pylint: disable = attribute-defined-outside-init


def get_logs(log_dir_path: str) -> tp.Tuple[pd.DataFrame, ...]:
    """
    Load training/validation loss  from a CSV file.

    Parameters
    ----------
    log_dir_path : str
        Path to the directory containing the `metrics.csv` file.

    Returns
    -------
    loss_df : pd.DataFrame
        DataFrame containing training and validation loss per epoch.
    metrics_df : pd.DataFrame
        DataFrame containing other metrics (excluding loss) per epoch.
    """
    log_path = os.path.join(log_dir_path, "metrics.csv")
    epoch_metrics_df = pd.read_csv(log_path)

    loss_df = epoch_metrics_df[["epoch", "train_loss"]].dropna()
    val_loss_df = epoch_metrics_df[["epoch", "val_loss"]].dropna()
    loss_df = pd.merge(loss_df, val_loss_df, how="inner", on="epoch")
    loss_df.reset_index(drop=True, inplace=True)

    metrics_df = epoch_metrics_df.drop(columns=["train_loss", "val_loss"]).dropna()
    metrics_df.reset_index(drop=True, inplace=True)

    return loss_df, metrics_df


def create_subplots_grid(n_plots: int) -> tp.Tuple[Figure, np.ndarray[Axes]]:
    """
    Create a grid of subplots with 2 columns per row.

    Parameters
    ----------
    n_plots : int
        Number of plots to display.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : np.ndarray of matplotlib.axes.Axes
        Array of subplot axes.
    """
    n_rows = (n_plots + 1) // 2
    figsize = (12, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    if n_plots % 2 == 1:
        axes[-1, -1].axis("off")

    return fig, axes


def rolling_avg(
    x: pd.Series,
    y: pd.Series,
    window: int,
) -> tp.Tuple[pd.Series, pd.Series]:
    """
    Compute rolling average of y values over x.

    Parameters
    ----------
    x : pd.Series
        X-axis values (e.g. epoch numbers).
    y : pd.Series
        Y-axis values (e.g. metric values).
    window : int
        Size of the rolling window.

    Returns
    -------
    pd.Series, pd.Series
        Smoothed x and y values.
    """
    df = pd.DataFrame({"x": x, "y": y}).sort_values("x")
    df["y_smooth"] = df["y"].rolling(window=window, center=True).mean()
    return df["x"], df["y_smooth"]


def show_val_metrics(train_stage_metrics: dict[str, tp.Any]) -> None:
    """
    Plot validation and training loss for all models.

    Parameters
    ----------
    train_stage_metrics : dict[str, tuple]
        Dictionary mapping model names to their training metrics (loss_df, metrics_df).
    """
    n_plots = len(train_stage_metrics)
    models_name = list(train_stage_metrics.keys())
    fig, axes = create_subplots_grid(n_plots=n_plots)

    for ax, model_name in zip(axes.flat, models_name):
        y1 = train_stage_metrics[model_name][0]["val_loss"]
        y2 = train_stage_metrics[model_name][0]["train_loss"]
        x = train_stage_metrics[model_name][0]["epoch"]
        ax.plot(x, y1, label="val_loss")
        ax.plot(x, y2, label="train_loss")
        ax.set_title(f"{model_name}")
        ax.legend()
    plt.show()


def get_results(path_to_load_res: str, metrics_to_show: tp.List[str], show_loss: bool = False) -> pd.DataFrame:
    """
    Load and visualize training results from a JSON file.

    Parameters
    ----------
    path_to_load_res : str
        Path to the JSON file containing experiment data.
    metrics_to_show : List[str]
        List of metric names to include in the output table.
    show_loss : bool, default=False
        Whether to plot training and validation loss curves.

    Returns
    -------
    pd.DataFrame
        A DataFrame with mean values of specified metrics per model.
    """
    with open(path_to_load_res, "r", encoding="utf-8") as f:
        exp_data = json.load(f)

    train_stage_metrics = {
        model_name: get_logs(log_dir_path) for model_name, log_dir_path in exp_data["models_log_dir"].items()
    }
    if show_loss:
        show_val_metrics(train_stage_metrics)

    plt.figure(figsize=(10, 6))
    for model_name, tr_results in train_stage_metrics.items():
        x = tr_results[1]["epoch"]
        y = tr_results[1]["recall@10"]
        x_smooth, y_smooth = rolling_avg(x, y, window=WINDOW_AVG)
        plt.plot(x_smooth, y_smooth, label=model_name)

    plt.grid(False)
    ax = plt.gca()
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.5)
    legend = plt.legend(frameon=True, edgecolor="black", facecolor="white", framealpha=1, fontsize=10)
    legend.get_frame().set_linewidth(1.5)
    plt.title("Validation smoothed recall@10 dynamic")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@10")
    plt.legend()
    plt.show()

    pivot_results = (
        pd.DataFrame(exp_data["metrics"]).drop(columns="i_split").groupby(["model"], sort=False).agg(["mean"])
    )
    pivot_results.columns = pivot_results.columns.droplevel(1)
    return pivot_results[metrics_to_show]
