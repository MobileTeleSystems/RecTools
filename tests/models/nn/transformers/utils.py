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

import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import typing as tp
from rectools import Columns
from rectools import ExternalIds

def leave_one_out_mask(
    interactions: pd.DataFrame,
    n_val_users: tp.Optional[tp.Union[ExternalIds, int]] = None
) -> np.ndarray:
    groups = interactions.groupby(Columns.User)
    time_order = (
        groups[Columns.Datetime]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    n_interactions = groups.transform("size").astype(int)
    inv_ranks = n_interactions - time_order
    last_interact_mask  = inv_ranks == 0
    val_users = interactions[Columns.User].unique()
    if isinstance(n_val_users, int):
        val_users = val_users[:n_val_users]
    elif n_val_users is not None:
        val_users = n_val_users

    mask = (interactions[Columns.User].isin(val_users)) & last_interact_mask
    return mask.values

def custom_trainer() -> Trainer:
    return Trainer(
        max_epochs=3,
        min_epochs=3,
        deterministic=True,
        accelerator="cpu",
        enable_checkpointing=False,
        devices=1,
    )

def custom_trainer_ckpt() -> Trainer:
    return Trainer(
        max_epochs=3,
        min_epochs=3,
        deterministic=True,
        accelerator="cpu",
        devices=1,
        callbacks=ModelCheckpoint(filename="last_epoch"),
    )

def custom_trainer_multiple_ckpt() -> Trainer:
    return Trainer(
        max_epochs=3,
        min_epochs=3,
        deterministic=True,
        accelerator="cpu",
        devices=1,
        callbacks=ModelCheckpoint(
            monitor="train_loss",
            save_top_k=3,
            every_n_epochs=1,
            filename="{epoch}",
        ),
    )
