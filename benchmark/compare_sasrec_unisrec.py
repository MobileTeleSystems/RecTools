#  Copyright 2026 MTS (Mobile Telesystems)
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

"""Compare RecTools SASRec vs UniSRec-ID on ML-20M.

Both use full softmax, Adam, n_factors=256, 10 epochs.
MIN_RATING=-1 (no filter), MIN_ITEM_INTERACTIONS=5, MIN_USER_INTERACTIONS=2.
Writes results to benchmark/comparison_report.md.

Usage:
    python benchmark/compare_sasrec_unisrec.py

Data is downloaded automatically if not present.
If pretrained embeddings are not found, random embeddings are generated
(sufficient for ID-only comparison).
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel

import gc
import io
import time
import typing as tp
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]
import torch
from tqdm import tqdm

from rectools import Columns
from rectools.dataset import Dataset
from rectools.fast_transformers import UniSRecModel
from rectools.fast_transformers.preprocessing import build_sequences
from rectools.models import SASRecModel

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data" / "ml-20m"
RATINGS_PATH = DATA_DIR / "ratings.csv"
CACHE_EMB_PATH = DATA_DIR / "qwen_embeddings.pt"
REPORT_PATH = BENCHMARK_DIR / "comparison_report.md"

ML20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"

MIN_RATING = -1
MIN_ITEM_INTERACTIONS = 5
MIN_USER_INTERACTIONS = 2

EPOCHS = 10
PATIENCE: tp.Optional[int] = None
BATCH_SIZE = 128
SESSION_MAX_LEN = 200
N_FACTORS = 256
N_BLOCKS = 2
N_HEADS = 1
LR = 1e-3


def download_ml20m() -> None:
    """Download and extract ML-20M if not present."""
    if RATINGS_PATH.exists():
        return
    print(f"Downloading ML-20M from {ML20M_URL} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(ML20M_URL, stream=True, timeout=600)
    resp.raise_for_status()
    buf = io.BytesIO()
    total = int(resp.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True, desc="Download") as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            buf.write(chunk)
            pbar.update(len(chunk))
    print("Extracting...")
    with zipfile.ZipFile(buf) as zf:
        for member in zf.namelist():
            # ml-20m/ratings.csv -> DATA_DIR/ratings.csv
            basename = Path(member).name
            if not basename:
                continue
            target = DATA_DIR / basename
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"Extracted to {DATA_DIR}")


def load_and_preprocess() -> pd.DataFrame:
    download_ml20m()
    ratings = pd.read_csv(RATINGS_PATH)
    ratings.columns = ["user_id", "item_id", "rating", "timestamp"]

    if MIN_RATING > 0:
        ratings = ratings[ratings["rating"] >= MIN_RATING]

    if MIN_ITEM_INTERACTIONS > 0:
        item_counts = ratings.groupby("item_id").size()
        popular = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
        ratings = ratings[ratings["item_id"].isin(popular)]

    if MIN_USER_INTERACTIONS > 0:
        user_counts = ratings.groupby("user_id").size()
        valid = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
        ratings = ratings[ratings["user_id"].isin(valid)]

    return ratings


def split_eval(ratings: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["user_id", "timestamp"])
    grouped = ratings.groupby("user_id")
    test_idx = grouped.tail(1).index
    remaining = ratings.drop(test_idx)
    val_idx = remaining.groupby("user_id").tail(1).index
    train_idx = remaining.drop(val_idx).index
    return ratings.loc[train_idx], ratings.loc[val_idx], ratings.loc[test_idx]


def to_tensors(df: pd.DataFrame) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(df["user_id"].values, dtype=torch.long),
        torch.tensor(df["item_id"].values, dtype=torch.long),
        torch.tensor(df["timestamp"].values, dtype=torch.long),
    )


def get_pretrained_embeddings(item_ids: pd.Series, dim: int = 1024) -> torch.Tensor:
    """Load cached embeddings or generate random ones for ID-only comparison."""
    if CACHE_EMB_PATH.exists():
        print(f"Loading pretrained embeddings from {CACHE_EMB_PATH}")
        return torch.load(CACHE_EMB_PATH, weights_only=True)

    max_id = int(item_ids.max())
    print(f"No pretrained embeddings found at {CACHE_EMB_PATH}")
    print(f"Generating random embeddings ({max_id + 1}, {dim}) for ID-only comparison")
    torch.manual_seed(42)
    emb = torch.randn(max_id + 1, dim)
    emb[0] = 0.0
    return emb


@torch.no_grad()
def evaluate_unisrec(
    model: UniSRecModel, train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10, batch_size: int = 256
) -> tp.Dict[str, tp.Any]:
    net = model.net
    net.cuda().eval()
    device = torch.device("cuda")
    maxlen = model.session_max_len

    item_embs = net.project_all()
    unique_items = model.item_id_mapping
    assert unique_items is not None
    ext_to_int = {int(unique_items[i].item()): i + 1 for i in range(len(unique_items))}

    train_grouped = train_df.sort_values("timestamp").groupby("user_id")["item_id"].agg(list).to_dict()
    test_grouped = test_df.groupby("user_id")["item_id"].first().to_dict()
    test_users = list(test_grouped.keys())

    hits, ndcg_sum, mrr_sum, total = 0, 0.0, 0.0, 0
    for start in tqdm(range(0, len(test_users), batch_size), desc="Eval UniSRec"):
        batch_users = test_users[start : start + batch_size]
        seqs, targets = [], []
        for uid in batch_users:
            history = train_grouped.get(uid, [])
            mapped = [ext_to_int[iid] for iid in history if iid in ext_to_int]
            if not mapped:
                continue
            seq = mapped[-maxlen:]
            seqs.append([0] * (maxlen - len(seq)) + seq)
            targets.append(ext_to_int.get(test_grouped[uid]))
        if not seqs:
            continue
        x = torch.tensor(seqs, dtype=torch.long, device=device)
        h = net.encode_last(x)
        scores = h @ item_embs.T
        scores[:, 0] = float("-inf")
        for i, target_int in enumerate(targets):
            if target_int is None:
                continue
            _, topk_idx = scores[i].topk(k)
            topk = topk_idx.cpu().tolist()
            if target_int in topk:
                rank = topk.index(target_int)
                hits += 1
                ndcg_sum += 1.0 / np.log2(rank + 2)
                mrr_sum += 1.0 / (rank + 1)
            total += 1
    return {"HR@10": hits / total, "NDCG@10": ndcg_sum / total, "MRR@10": mrr_sum / total, "n_users": total}


def evaluate_sasrec(
    model: SASRecModel, dataset_for_recommend: Dataset, test_df: pd.DataFrame, k: int = 10
) -> tp.Dict[str, tp.Any]:
    test_users = test_df["user_id"].unique()
    reco = model.recommend(users=test_users, dataset=dataset_for_recommend, k=k, filter_viewed=False)

    test_targets = test_df.groupby("user_id")["item_id"].first().to_dict()
    hits, ndcg_sum, mrr_sum, total = 0, 0.0, 0.0, 0
    for uid, group in reco.groupby(Columns.User):
        target = test_targets.get(uid)
        if target is None:
            continue
        items = group[Columns.Item].tolist()
        if target in items:
            rank = items.index(target)
            hits += 1
            ndcg_sum += 1.0 / np.log2(rank + 2)
            mrr_sum += 1.0 / (rank + 1)
        total += 1
    return {"HR@10": hits / total, "NDCG@10": ndcg_sum / total, "MRR@10": mrr_sum / total, "n_users": total}


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def write_report(timings: dict, metrics: dict, data_info: dict) -> str:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset_str = (
        f"ML-20M (min_rating={MIN_RATING}," f" min_item={MIN_ITEM_INTERACTIONS}," f" min_user={MIN_USER_INTERACTIONS})"
    )
    lines = [
        "# SASRec vs UniSRec-ID Comparison",
        "",
        f"**Date:** {date_str}  ",
        f"**GPU:** {gpu_name}  ",
        f"**Dataset:** {dataset_str}",
        "",
        "## Data",
        "",
        "| | Count |",
        "|---|---:|",
        f"| Interactions | {data_info['n_interactions']:,} |",
        f"| Users | {data_info['n_users']:,} |",
        f"| Items | {data_info['n_items']:,} |",
        f"| Train | {data_info['n_train']:,} |",
        f"| Val | {data_info['n_val']:,} |",
        f"| Test | {data_info['n_test']:,} |",
        "",
        "## Config",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| n_factors | {N_FACTORS} |",
        f"| n_blocks | {N_BLOCKS} |",
        f"| n_heads | {N_HEADS} |",
        f"| session_max_len | {SESSION_MAX_LEN} |",
        f"| batch_size | {BATCH_SIZE} |",
        f"| lr | {LR} |",
        "| loss | softmax |",
        "| optimizer | Adam |",
        f"| epochs | {EPOCHS} |",
        f"| patience | {PATIENCE} |",
        "| dropout | 0.1 |",
        "",
        "## Timing",
        "",
        "| Stage | SASRec | UniSRec ID |",
        "|---|---:|---:|",
    ]

    for stage in ["data_load", "preprocessing", "model_init", "training", "eval"]:
        s = timings.get(f"sasrec_{stage}", 0)
        u = timings.get(f"unisrec_{stage}", 0)
        label = {
            "data_load": "Data load & split",
            "preprocessing": "Preprocessing",
            "model_init": "Model init",
            "training": f"Training ({EPOCHS} epochs)",
            "eval": "Evaluation",
        }[stage]
        lines.append(f"| {label} | {s:.1f}s | {u:.1f}s |")

    s_total = sum(timings.get(f"sasrec_{s}", 0) for s in ["preprocessing", "model_init", "training", "eval"])
    u_total = sum(timings.get(f"unisrec_{s}", 0) for s in ["preprocessing", "model_init", "training", "eval"])
    lines.append(f"| **Total** | **{s_total:.1f}s** | **{u_total:.1f}s** |")

    s_epoch = timings.get("sasrec_training", 0) / max(timings.get("sasrec_epochs_done", 1), 1)
    u_epoch = timings.get("unisrec_training", 0) / max(timings.get("unisrec_epochs_done", 1), 1)
    s_epochs_done = timings.get("sasrec_epochs_done", EPOCHS)
    u_epochs_done = timings.get("unisrec_epochs_done", EPOCHS)
    prep_speedup = timings.get("prep_speedup", 0)
    lines.extend(
        [
            "",
            "| | SASRec | UniSRec ID |",
            "|---|---:|---:|",
            f"| Epochs completed | {s_epochs_done} | {u_epochs_done} |",
            f"| Time per epoch | {s_epoch:.1f}s | {u_epoch:.1f}s |",
            f"| Preprocessing speedup | — | {prep_speedup:.0f}x |",
        ]
    )

    n_test_users = metrics["sasrec"]["n_users"]
    lines.extend(
        [
            "",
            f"## Quality (test set, {n_test_users:,} users)",
            "",
            "| Model | HR@10 | NDCG@10 | MRR@10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, key in [("SASRec", "sasrec"), ("UniSRec ID", "unisrec")]:
        m = metrics[key]
        lines.append(f"| {name} | {m['HR@10']:.4f} | {m['NDCG@10']:.4f} | {m['MRR@10']:.4f} |")

    hr_diff = (metrics["unisrec"]["HR@10"] / metrics["sasrec"]["HR@10"] - 1) * 100
    ndcg_diff = (metrics["unisrec"]["NDCG@10"] / metrics["sasrec"]["NDCG@10"] - 1) * 100
    lines.extend(
        [
            "",
            f"UniSRec vs SASRec: HR@10 {hr_diff:+.1f}%, NDCG@10 {ndcg_diff:+.1f}%",
        ]
    )

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report)
    print(f"\nReport written to {REPORT_PATH}")
    return report


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA. No GPU detected.")
    torch.set_float32_matmul_precision("high")
    timings = {}

    print(f"SASRec vs UniSRec-ID | {EPOCHS} epochs | n_factors={N_FACTORS} | Adam | softmax")
    print("=" * 70)

    # ── Data ──
    t0 = time.time()
    ratings = load_and_preprocess()
    train_ratings, val_ratings, test_ratings = split_eval(ratings)
    train_with_val = pd.concat([train_ratings, val_ratings])
    timings["data_load"] = time.time() - t0

    data_info = {
        "n_interactions": len(ratings),
        "n_users": ratings["user_id"].nunique(),
        "n_items": ratings["item_id"].nunique(),
        "n_train": len(train_ratings),
        "n_val": len(val_ratings),
        "n_test": len(test_ratings),
    }
    n_int = data_info["n_interactions"]
    n_usr = data_info["n_users"]
    n_itm = data_info["n_items"]
    print(f"Data: {n_int:,} interactions, {n_usr:,} users, {n_itm:,} items")
    print(f"Split: train={data_info['n_train']:,}, val={data_info['n_val']:,}, test={data_info['n_test']:,}")

    user_ids_t, item_ids_t, timestamps_t = to_tensors(train_with_val)
    pretrained = get_pretrained_embeddings(ratings["item_id"])

    # ══════════════════════════════════════════════════════════════
    # 1. SASRec (RecTools)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"1. SASRec (RecTools) — {EPOCHS} epochs")
    print(f"{'=' * 70}")

    # Preprocessing
    t0 = time.time()
    df_rectools = pd.DataFrame(
        {
            Columns.User: train_with_val["user_id"].values,
            Columns.Item: train_with_val["item_id"].values,
            Columns.Weight: 1.0,
            Columns.Datetime: pd.to_datetime(train_with_val["timestamp"], unit="s"),
        }
    )
    dataset = Dataset.construct(df_rectools)
    timings["sasrec_preprocessing"] = time.time() - t0
    print(f"  Preprocessing (Dataset.construct): {timings['sasrec_preprocessing']:.2f}s")

    # Model init + training
    def sasrec_trainer(**kwargs: tp.Any) -> tp.Any:
        import pytorch_lightning as pl

        callbacks: tp.List[tp.Any] = []
        if PATIENCE is not None:
            from pytorch_lightning.callbacks import EarlyStopping

            callbacks.append(EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"))
        return pl.Trainer(
            max_epochs=EPOCHS,
            min_epochs=1,
            callbacks=callbacks or None,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=True,
            enable_progress_bar=True,
            devices=1,
        )

    sasrec_kwargs: tp.Dict[str, tp.Any] = {
        "n_factors": N_FACTORS,
        "n_blocks": N_BLOCKS,
        "n_heads": N_HEADS,
        "session_max_len": SESSION_MAX_LEN,
        "dropout_rate": 0.1,
        "loss": "softmax",
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "train_min_user_interactions": MIN_USER_INTERACTIONS,
        "dataloader_num_workers": 0,
        "verbose": 1,
        "get_trainer_func": sasrec_trainer,
    }
    if PATIENCE is not None:

        def sasrec_val_mask(interactions_df: pd.DataFrame, **kwargs: tp.Any) -> pd.Series:
            idx = interactions_df.groupby(Columns.User).tail(1).index
            mask = pd.Series(False, index=interactions_df.index)
            mask.loc[idx] = True
            return mask

        sasrec_kwargs["get_val_mask_func"] = sasrec_val_mask

    t0 = time.time()
    sasrec = SASRecModel(**sasrec_kwargs)
    timings["sasrec_model_init"] = time.time() - t0

    t0 = time.time()
    sasrec.fit(dataset)
    timings["sasrec_training"] = time.time() - t0
    timings["sasrec_epochs_done"] = sasrec.fit_trainer.current_epoch + 1 if sasrec.fit_trainer else EPOCHS
    print(f"  Training: {timings['sasrec_training']:.1f}s, {timings['sasrec_epochs_done']} epochs")

    # Eval
    print("  Evaluating...")
    t0 = time.time()
    sasrec_metrics = evaluate_sasrec(sasrec, dataset, test_ratings)
    timings["sasrec_eval"] = time.time() - t0
    print(f"  Eval: {timings['sasrec_eval']:.1f}s")
    hr = sasrec_metrics["HR@10"]
    ndcg = sasrec_metrics["NDCG@10"]
    mrr = sasrec_metrics["MRR@10"]
    print(f"  HR@10={hr:.4f}  NDCG@10={ndcg:.4f}  MRR@10={mrr:.4f}")
    del sasrec
    cleanup()

    # ══════════════════════════════════════════════════════════════
    # 2. UniSRec ID
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"2. UniSRec ID — {EPOCHS} epochs")
    print(f"{'=' * 70}")

    # Preprocessing
    torch.cuda.synchronize()
    t0 = time.time()
    _ = build_sequences(user_ids_t, item_ids_t, timestamps_t, max_len=SESSION_MAX_LEN, device="cuda")
    torch.cuda.synchronize()
    timings["unisrec_preprocessing"] = time.time() - t0
    print(f"  Preprocessing (build_sequences): {timings['unisrec_preprocessing']:.4f}s")
    timings["prep_speedup"] = timings["sasrec_preprocessing"] / max(timings["unisrec_preprocessing"], 1e-6)
    print(f"  Speedup vs Dataset.construct: {timings['prep_speedup']:.0f}x")

    # Model init
    t0 = time.time()
    unisrec_id = UniSRecModel(
        pretrained_item_embeddings=pretrained,
        n_factors=N_FACTORS,
        projection_hidden=N_FACTORS,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        session_max_len=SESSION_MAX_LEN,
        dropout=0.1,
        adaptor_dropout=0.2,
        adaptor_type="pca",
        use_adaptor_ffn=True,
        epochs=EPOCHS,
        lr=LR,
        optimizer="adam",
        grad_clip=1.0,
        weight_decay=0.0,
        loss="softmax",
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        dataloader_num_workers=0,
        train_min_user_interactions=MIN_USER_INTERACTIONS,
        device="cuda",
        verbose=1,
    )
    timings["unisrec_model_init"] = time.time() - t0

    # Training
    t0 = time.time()
    unisrec_id.fit(user_ids_t, item_ids_t, timestamps_t)
    timings["unisrec_training"] = time.time() - t0
    timings["unisrec_epochs_done"] = EPOCHS
    print(f"  Training (total fit): {timings['unisrec_training']:.1f}s")

    # Eval
    print("  Evaluating...")
    t0 = time.time()
    unisrec_metrics = evaluate_unisrec(unisrec_id, train_with_val, test_ratings)
    timings["unisrec_eval"] = time.time() - t0
    print(f"  Eval: {timings['unisrec_eval']:.1f}s")
    hr = unisrec_metrics["HR@10"]
    ndcg = unisrec_metrics["NDCG@10"]
    mrr = unisrec_metrics["MRR@10"]
    print(f"  HR@10={hr:.4f}  NDCG@10={ndcg:.4f}  MRR@10={mrr:.4f}")
    del unisrec_id
    cleanup()

    # ── Report ──
    metrics = {"sasrec": sasrec_metrics, "unisrec": unisrec_metrics}
    report = write_report(timings, metrics, data_info)
    print("\n" + report)


if __name__ == "__main__":
    main()
