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

import sys

import pandas as pd
import pytest

try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
except ImportError:
    pass

from rectools import Columns

pytestmark = pytest.mark.skipif(sys.version_info >= (3, 13), reason="`torch` is not compatible with Python >= 3.13")


def leave_one_out_mask(interactions: pd.DataFrame) -> pd.Series:
    rank = (
        interactions.sort_values(Columns.Datetime, ascending=False, kind="stable")
        .groupby(Columns.User, sort=False)
        .cumcount()
    )
    return rank == 0


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
