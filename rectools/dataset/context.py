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

from rectools import Columns
from rectools.dataset import Interactions


def get_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract initial interaction context for each user.

    For each user, finds the earliest index base on datetime and uses it to define
    the initial contextual data. If the item column is present, it is dropped from the result,
    as it's not part of the user context.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing user interactions with at least
        user ID and datetime columns.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per user, representing the earliest
        context data for that user.
    """
    df = df.copy()
    if Columns.Weight not in df.columns:
        df[Columns.Weight] = 1.0
    Interactions.convert_weight_and_datetime_types(df)
    earliest = df.groupby(Columns.User)[Columns.Datetime].idxmin()
    context = df.loc[earliest]
    if Columns.Item in context:
        context.drop(columns=[Columns.Item], inplace=True)
    return context
