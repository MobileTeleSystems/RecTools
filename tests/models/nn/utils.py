import pandas as pd

from rectools import Columns


def leave_one_out_mask(interactions: pd.DataFrame) -> pd.Series:
    rank = (
        interactions.sort_values(Columns.Datetime, ascending=False, kind="stable")
        .groupby(Columns.User, sort=False)
        .cumcount()
    )
    return rank == 0
