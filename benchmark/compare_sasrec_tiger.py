"""Compare RecTools SASRecModel and semantic TIGER on ML-20M.

Both models are trained on MovieLens 20M data and the benchmark writes a
Markdown report to ``benchmark/comparison_sasrec_vs_tiger.md``.

Usage:
```bash
python3 -m benchmark.compare_sasrec_tiger
```

The script downloads ML-20M automatically if it is not present locally.
Movie metadata is taken from ``movies.csv`` and embedded with
``Qwen/Qwen3-Embedding-0.6B`` for the TIGER tokenizer.
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
# pylint: disable=import-error,unsubscriptable-object,protected-access,cyclic-import

import argparse
import gc
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sentence_transformers import SentenceTransformer
from tqdm.auto import trange

from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import MRR, NDCG, CatalogCoverage, HitRate
from rectools.models import SASRecModel
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.models.nn.transformers.utils import leave_one_out_mask
from rectools.semantic.data_handling import k_core, loo_split
from rectools.semantic.metrics import gini_k
from rectools.semantic.tiger import TIGERModel
from rectools.semantic.tokenizer import SIDTokenizer

BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = BENCHMARK_DIR / "data" / "ml-20m"
DEFAULT_REPORT_PATH = BENCHMARK_DIR / "comparison_sasrec_vs_tiger.md"
ML20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RecTools SASRec against semantic TIGER on MovieLens 20M."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="Directory for downloaded data and cached artifacts.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to the Markdown report produced by the benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override. Defaults to cuda -> mps -> cpu.",
    )
    parser.add_argument(
        "--top-k-main",
        type=int,
        default=10,
        help="K for NDCG, MRR, coverage, and gini.",
    )
    parser.add_argument(
        "--max-length", type=int, default=200, help="Maximum context length."
    )
    parser.add_argument("--top-k-hit", type=int, default=10, help="K for hit rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Optional cap on users after preprocessing.",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=-1.0,
        help="Optional minimum rating filter. Use -1 to keep all ratings.",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=5,
        help="Optional minimum number of interactions per item.",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=2,
        help="Optional minimum number of interactions per user.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model used to embed movie metadata.",
    )
    parser.add_argument(
        "--tokenizer-max-epochs",
        type=int,
        default=1000,
        help="Tokenizer training epochs.",
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=256,
        help="Batch size for metadata embedding and tokenizer training.",
    )
    parser.add_argument(
        "--tokenizer-patience",
        type=int,
        default=100,
        help="Tokenizer early stopping patience.",
    )
    parser.add_argument(
        "--sasrec-epochs", type=int, default=10, help="Maximum SASRec epochs."
    )
    parser.add_argument(
        "--sasrec-patience",
        type=int,
        default=10,
        help="SASRec early stopping patience.",
    )
    parser.add_argument(
        "--sasrec-batch-size", type=int, default=256, help="SASRec batch size."
    )
    parser.add_argument(
        "--sasrec-num-workers", type=int, default=4, help="SASRec dataloader workers."
    )
    parser.add_argument(
        "--tiger-epochs", type=int, default=10, help="Maximum TIGER epochs."
    )
    parser.add_argument(
        "--tiger-patience", type=int, default=10, help="TIGER early stopping patience."
    )
    parser.add_argument(
        "--tiger-batch-size", type=int, default=256, help="TIGER train batch size."
    )
    parser.add_argument(
        "--tiger-eval-batch-size", type=int, default=256, help="TIGER eval batch size."
    )
    parser.add_argument(
        "--tiger-num-workers", type=int, default=4, help="TIGER dataloader workers."
    )
    return parser.parse_args()


def resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_index = torch.device(device).index or 0
        return f"{device} ({torch.cuda.get_device_name(gpu_index)})"
    if device == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def maybe_download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst}")
    urlretrieve(url, dst)  # nosec B310


def download_ml20m(workdir: Path) -> tuple[Path, Path]:
    ratings_path = workdir / "ratings.csv"
    movies_path = workdir / "movies.csv"
    if ratings_path.exists() and movies_path.exists():
        return ratings_path, movies_path

    archive_path = workdir / "ml-20m.zip"
    maybe_download(ML20M_URL, archive_path)

    print(f"Extracting {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            basename = Path(member).name
            if basename not in {
                "ratings.csv",
                "movies.csv",
                "README.txt",
                "links.csv",
                "tags.csv",
            }:
                continue
            target = workdir / basename
            if target.exists():
                continue
            with archive.open(member) as src, open(target, "wb") as dst_stream:
                shutil.copyfileobj(src, dst_stream)

    return ratings_path, movies_path


def load_interactions(ratings_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    interactions = pd.read_csv(ratings_path)
    interactions.columns = ["user_id", "item_id", "rating", "timestamp"]

    if args.min_rating > 0:
        interactions = interactions[interactions["rating"] >= args.min_rating]

    use_k_core = args.min_item_interactions > 0 or args.min_user_interactions > 0
    if use_k_core:
        print(
            f"Performing K-Core filtering (user>={args.min_user_interactions}, item>={args.min_item_interactions})..."
        )
        interactions = k_core(
            interactions,
            user_min_interactions=max(args.min_user_interactions, 0),
            item_min_interactions=max(args.min_item_interactions, 0),
        )

    if args.limit_users is not None:
        kept_users = interactions["user_id"].drop_duplicates().iloc[: args.limit_users]
        interactions = interactions[
            interactions["user_id"].isin(kept_users)
        ].reset_index(drop=True)
        if use_k_core:
            interactions = k_core(
                interactions,
                user_min_interactions=max(args.min_user_interactions, 0),
                item_min_interactions=max(args.min_item_interactions, 0),
            )

    interactions = interactions.sort_values(
        ["user_id", "timestamp", "item_id"]
    ).reset_index(drop=True)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit="s")

    return interactions


def save_preprocessed_files(
    workdir: Path,
    interactions: pd.DataFrame,
    meta: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_targets: pd.DataFrame,
) -> dict[str, Path]:
    preprocessed_dir = workdir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "interactions": preprocessed_dir / "interactions.csv",
        "item_metadata": preprocessed_dir / "item_metadata.csv",
        "train": preprocessed_dir / "train.csv",
        "val": preprocessed_dir / "val.csv",
        "test": preprocessed_dir / "test.csv",
        "test_targets": preprocessed_dir / "test_targets.csv",
    }
    interactions.to_csv(paths["interactions"], index=False)
    meta.to_csv(paths["item_metadata"], index=False)
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    test_targets.to_csv(paths["test_targets"], index=False)

    return paths


def build_meta(movies_path: Path, interactions: pd.DataFrame) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    movies = movies.rename(
        columns={"movieId": "item_id", "title": "title", "genres": "genres"}
    )
    item_ids = set(interactions["item_id"].unique())

    meta = movies[movies["item_id"].isin(item_ids)].copy()
    meta["title"] = meta["title"].fillna("Unknown").astype(str)
    meta["genres"] = meta["genres"].fillna("Unknown").astype(str)
    meta["text"] = meta["title"] + " Genres: " + meta["genres"]
    meta = meta[["item_id", "text"]].drop_duplicates("item_id")

    missing_items = sorted(item_ids - set(meta["item_id"]))
    if missing_items:
        missing_meta = pd.DataFrame(
            {
                "item_id": missing_items,
                "text": ["Unknown title. Genres: Unknown"] * len(missing_items),
            }
        )
        meta = pd.concat([meta, missing_meta], ignore_index=True)

    return meta.sort_values("item_id").reset_index(drop=True)


def map_ids(
    interactions: pd.DataFrame, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_codes = pd.Index(interactions["user_id"].unique())
    item_codes = pd.Index(interactions["item_id"].unique())

    user_map = pd.Series(np.arange(len(user_codes), dtype=np.int64), index=user_codes)
    item_map = pd.Series(np.arange(len(item_codes), dtype=np.int64), index=item_codes)

    mapped_interactions = interactions.copy()
    mapped_interactions["user_id"] = (
        mapped_interactions["user_id"].map(user_map).astype(np.int64)
    )
    mapped_interactions["item_id"] = (
        mapped_interactions["item_id"].map(item_map).astype(np.int64)
    )

    mapped_meta = meta[meta["item_id"].isin(item_map.index)].copy()
    mapped_meta["item_id"] = mapped_meta["item_id"].map(item_map).astype(np.int64)
    mapped_meta = mapped_meta.sort_values("item_id").reset_index(drop=True)

    return mapped_interactions, mapped_meta


def encode_texts(
    texts: list[str], model_name: str, device: str, batch_size: int
) -> np.ndarray:
    embeds = []
    model = SentenceTransformer(model_name, device=device).eval()
    with torch.no_grad():
        for start in trange(0, len(texts), batch_size, desc="Embedding movie metadata"):
            end = min(start + batch_size, len(texts))
            embeds.append(model.encode(texts[start:end], show_progress_bar=False))
    return np.vstack(embeds)


def make_sasrec_dataset(val_df: pd.DataFrame) -> Dataset:
    sasrec_df = val_df.rename(columns={"timestamp": Columns.Datetime}).copy()
    sasrec_df[Columns.Weight] = 1.0
    return Dataset.construct(
        interactions_df=sasrec_df[
            [Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]
        ]
    )


def get_last_item_targets(test_df: pd.DataFrame) -> pd.DataFrame:
    targets = (
        test_df.sort_values(["user_id", "timestamp"])
        .groupby("user_id", sort=False)
        .tail(1)
        .copy()
    )
    targets["weight"] = 1.0
    return targets[["user_id", "item_id", "weight"]]


def reco_to_topk_lists(reco: pd.DataFrame, k: int) -> list[list[int]]:
    ordered = reco[reco[Columns.Rank] <= k].sort_values([Columns.User, Columns.Rank])
    grouped = ordered.groupby(Columns.User, sort=False)[Columns.Item].agg(list)
    return grouped.tolist()


def evaluate_recommendations(
    reco: pd.DataFrame,
    targets: pd.DataFrame,
    catalog_size: int,
    top_k_main: int,
    top_k_hit: int,
) -> dict[str, float]:
    return {
        f"NDCG@{top_k_main}": NDCG(k=top_k_main).calc(reco, targets),
        f"HR@{top_k_hit}": HitRate(k=top_k_hit).calc(reco, targets),
        f"MRR@{top_k_main}": MRR(k=top_k_main).calc(reco, targets),
        f"Coverage@{top_k_main}": CatalogCoverage(k=top_k_main, normalize=True).calc(
            reco, catalog=list(range(catalog_size))
        ),
        f"Gini@{top_k_main}": gini_k(reco_to_topk_lists(reco, top_k_main), top_k_main),
    }


def train_sasrec(
    dataset: Dataset,
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    max_length: int,
    min_user_interactions: int,
) -> SASRecModel:
    accelerator = "gpu" if device.startswith("cuda") else device
    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": 1,
        "min_epochs": 1,
        "max_epochs": max_epochs,
        "callbacks": [
            EarlyStopping(
                monitor=SASRecModel.val_loss_name,
                patience=patience,
                mode="min",
            )
        ],
        "deterministic": True,
    }
    if device.startswith("cuda"):
        trainer_kwargs["precision"] = "bf16-mixed"

    trainer = Trainer(**trainer_kwargs)

    model = SASRecModel(
        n_factors=200,
        n_blocks=1,
        n_heads=2,
        dropout_rate=0.1,
        train_min_user_interactions=min_user_interactions,
        session_max_len=max_length,
        item_net_block_types=(IdEmbeddingsItemNet,),
        get_val_mask_func=leave_one_out_mask,
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        verbose=1,
        deterministic=True,
        recommend_torch_device=device,
    )
    model._trainer = trainer
    model.fit(dataset)
    return model


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_sasrec(
    val_df: pd.DataFrame,
    test_targets: pd.DataFrame,
    device: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, float]:
    print("Converting validation history to RecTools dataset for SASRec")
    dataset = make_sasrec_dataset(val_df)

    print("Fitting RecTools SASRec")
    fit_start = time.perf_counter()
    model = train_sasrec(
        dataset=dataset,
        device=device,
        max_epochs=args.sasrec_epochs,
        patience=args.sasrec_patience,
        batch_size=args.sasrec_batch_size,
        num_workers=args.sasrec_num_workers,
        max_length=args.max_length,
        min_user_interactions=args.min_user_interactions,
    )
    train_seconds = time.perf_counter() - fit_start

    users = test_targets["user_id"].to_numpy()
    inference_start = time.perf_counter()
    reco = model.recommend(
        users=users,
        dataset=dataset,
        k=max(args.top_k_main, args.top_k_hit),
        filter_viewed=False,
    )
    inference_seconds = time.perf_counter() - inference_start

    metrics = evaluate_recommendations(
        reco=reco,
        targets=test_targets,
        catalog_size=val_df["item_id"].nunique(),
        top_k_main=args.top_k_main,
        top_k_hit=args.top_k_hit,
    )
    return metrics, train_seconds, inference_seconds


def benchmark_tiger(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_targets: pd.DataFrame,
    meta: pd.DataFrame,
    device: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], float, float]:
    meta = meta.sort_values("item_id").drop_duplicates("item_id")
    embeddings = encode_texts(
        meta["text"].tolist(),
        model_name=args.embedding_model,
        device=device,
        batch_size=args.tokenizer_batch_size,
    )
    tokenizer = SIDTokenizer(
        input_dim=embeddings.shape[1],
        codebook_sizes=[256, 256, 256],
        codebook_dim=32,
        hidden_dims=[768, 256, 128, 64],
        quantizer="rqvae",
        device=device,
    )
    tokenizer.fit(
        item_ids=meta["item_id"].tolist(),
        embeddings=embeddings,
        max_epochs=args.tokenizer_max_epochs,
        patience=args.tokenizer_patience,
        batch_size=args.tokenizer_batch_size,
        save_dir=str(args.workdir / "tokenizer_checkpoints"),
    )

    tiger = TIGERModel(
        tokenizer=tokenizer,
        hidden_units=128,
        num_blocks=4,
        num_heads=6,
        dropout_rate=0.1,
        max_length=args.max_length,
        lr=1e-3,
        lr_schedule="cosine",
        warmup_steps=0,
        max_epochs=args.tiger_epochs,
        patience=args.tiger_patience,
        batch_size=args.tiger_batch_size,
        eval_batch_size=args.tiger_eval_batch_size,
        num_workers=args.tiger_num_workers,
        beam_size=max(20, args.top_k_hit),
        top_k=args.top_k_main,
        d_kv=64,
        ff_dim=1024,
        device=device,
        random_seed=args.seed,
    )

    print("Fitting TIGER")
    fit_start = time.perf_counter()
    tiger.fit(train_df=train_df, val_df=val_df)
    train_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    reco = tiger.predict(val_df, top_k=max(args.top_k_main, args.top_k_hit))
    inference_seconds = time.perf_counter() - inference_start

    metrics = evaluate_recommendations(
        reco=reco,
        targets=test_targets,
        catalog_size=val_df["item_id"].nunique(),
        top_k_main=args.top_k_main,
        top_k_hit=args.top_k_hit,
    )
    return metrics, train_seconds, inference_seconds


def collect_dataset_info(
    interactions: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: pd.DataFrame,
) -> dict[str, object]:
    timestamps = interactions["timestamp"]
    return {
        "n_interactions": len(interactions),
        "n_users": interactions["user_id"].nunique(),
        "n_items": interactions["item_id"].nunique(),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_meta_items": len(meta),
        "min_timestamp": timestamps.min(),
        "max_timestamp": timestamps.max(),
        "avg_interactions_per_user": len(interactions)
        / interactions["user_id"].nunique(),
        "avg_interactions_per_item": len(interactions)
        / interactions["item_id"].nunique(),
    }


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    device: str,
    data_info: dict[str, object],
    preprocessed_paths: dict[str, Path],
    results: dict[str, dict[str, float]],
    started_at: datetime,
    finished_at: datetime,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_date = finished_at.strftime("%Y-%m-%d %H:%M:%S")
    duration_seconds = (finished_at - started_at).total_seconds()
    dataset_label = "MovieLens 20M"

    metric_names = [
        f"NDCG@{args.top_k_main}",
        f"HR@{args.top_k_hit}",
        f"MRR@{args.top_k_main}",
        f"Coverage@{args.top_k_main}",
        f"Gini@{args.top_k_main}",
    ]

    lines = [
        "# SASRec vs TIGER Comparison",
        "",
        f"**Date:** {report_date}  ",
        f"**Duration:** {duration_seconds:.1f}s  ",
        f"**Device:** {describe_device(device)}  ",
        f"**Dataset:** {dataset_label}  ",
        f"**Report path:** `{report_path}`",
        "",
        "## Run Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Data directory | `{args.workdir}` |",
        f"| Ratings source | `{args.workdir / 'ratings.csv'}` |",
        f"| Movies source | `{args.workdir / 'movies.csv'}` |",
        f"| Embedding model | `{args.embedding_model}` |",
        f"| Torch version | `{torch.__version__}` |",
        f"| PyTorch Lightning version | `{pl.__version__}` |",
        f"| Seed | {args.seed} |",
        f"| Max sequence length | {args.max_length} |",
        f"| top_k_main | {args.top_k_main} |",
        f"| top_k_hit | {args.top_k_hit} |",
        "",
        "## Dataset Statistics",
        "",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Interactions | {data_info['n_interactions']:,} |",
        f"| Users | {data_info['n_users']:,} |",
        f"| Items | {data_info['n_items']:,} |",
        f"| Metadata rows | {data_info['n_meta_items']:,} |",
        f"| Train interactions | {data_info['n_train']:,} |",
        f"| Validation interactions | {data_info['n_val']:,} |",
        f"| Test interactions | {data_info['n_test']:,} |",
        f"| Avg interactions per user | {data_info['avg_interactions_per_user']:.2f} |",
        f"| Avg interactions per item | {data_info['avg_interactions_per_item']:.2f} |",
        f"| First timestamp | {data_info['min_timestamp']} |",
        f"| Last timestamp | {data_info['max_timestamp']} |",
        "",
        "## Filtering and Data Preparation",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Minimum rating | {args.min_rating} |",
        f"| Minimum item interactions | {args.min_item_interactions} |",
        f"| Minimum user interactions | {args.min_user_interactions} |",
        f"| User cap | {args.limit_users if args.limit_users is not None else 'None'} |",
        "| Interaction count filter | Iterative k-core |",
        "| Split strategy | Leave-one-out per user |",
        "| TIGER metadata text | `title + genres` from `movies.csv` |",
        "",
        "## Preprocessed Files",
        "",
        "| File | Path |",
        "|---|---|",
        f"| Interactions | `{preprocessed_paths['interactions']}` |",
        f"| Item metadata | `{preprocessed_paths['item_metadata']}` |",
        f"| Train split | `{preprocessed_paths['train']}` |",
        f"| Validation split | `{preprocessed_paths['val']}` |",
        f"| Test split | `{preprocessed_paths['test']}` |",
        f"| Test targets | `{preprocessed_paths['test_targets']}` |",
        "",
        "## Model Configuration",
        "",
        "| Parameter | SASRec | TIGER |",
        "|---|---|---|",
        f"| Epochs | {args.sasrec_epochs} | {args.tiger_epochs} |",
        f"| Patience | {args.sasrec_patience} | {args.tiger_patience} |",
        f"| Batch size | {args.sasrec_batch_size} | {args.tiger_batch_size} |",
        f"| Eval batch size | - | {args.tiger_eval_batch_size} |",
        f"| Num workers | {args.sasrec_num_workers} | {args.tiger_num_workers} |",
        f"| Max length | {args.max_length} | {args.max_length} |",
        "| Architecture notes | RecTools SASRec defaults from this script | TIGER + SIDTokenizer (RQ-VAE) |",
        "",
        "## Results",
        "",
        "| Model | Train time, s | Inference time, s | "
        + " | ".join(metric_names)
        + " |",
        "|---|---:|---:|" + "---:" * len(metric_names) + "|",
    ]

    for model_name, model_metrics in results.items():
        row = [
            model_name,
            f"{model_metrics['train_seconds']:.3f}",
            f"{model_metrics['inference_seconds']:.3f}",
        ]
        row.extend(f"{model_metrics[metric_name]:.6f}" for metric_name in metric_names)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The script downloads ML-20M automatically if the data files are missing.",
            "- TIGER uses stochastic SID collision resolution; this benchmark fixes it with the run seed.",
            "- Coverage is computed against the total number of unique mapped items in the evaluation catalog.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


def main() -> None:
    args = parse_args()
    started_at = datetime.now()
    pl.seed_everything(args.seed, workers=True)

    device = resolve_device(args.device)
    args.workdir.mkdir(parents=True, exist_ok=True)

    ratings_path, movies_path = download_ml20m(args.workdir)

    interactions = load_interactions(ratings_path, args)
    meta = build_meta(movies_path, interactions)
    interactions, meta = map_ids(interactions, meta)

    train_df, val_df, test_df = loo_split(interactions, timestamp_col="timestamp")
    test_targets = get_last_item_targets(test_df)
    preprocessed_paths = save_preprocessed_files(
        workdir=args.workdir,
        interactions=interactions,
        meta=meta,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        test_targets=test_targets,
    )
    data_info = collect_dataset_info(interactions, train_df, val_df, test_df, meta)

    print(
        "Prepared dataset:",
        f"{data_info['n_interactions']:,} interactions,",
        f"{data_info['n_users']:,} users,",
        f"{data_info['n_items']:,} items",
    )

    sasrec_metrics, sasrec_train_s, sasrec_infer_s = benchmark_sasrec(
        val_df=val_df,
        test_targets=test_targets,
        device=device,
        args=args,
    )
    cleanup()

    tiger_metrics, tiger_train_s, tiger_infer_s = benchmark_tiger(
        train_df=train_df,
        val_df=val_df,
        test_targets=test_targets,
        meta=meta,
        device=device,
        args=args,
    )

    results = {
        "SASRec": {
            "train_seconds": sasrec_train_s,
            "inference_seconds": sasrec_infer_s,
            **sasrec_metrics,
        },
        "TIGER": {
            "train_seconds": tiger_train_s,
            "inference_seconds": tiger_infer_s,
            **tiger_metrics,
        },
    }

    finished_at = datetime.now()
    write_report(
        report_path=args.report_path,
        args=args,
        device=device,
        data_info=data_info,
        preprocessed_paths=preprocessed_paths,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )


if __name__ == "__main__":
    main()
