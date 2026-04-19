from __future__ import annotations

import pandas as pd

from . import constants as C


def add_time_columns(df: pd.DataFrame, time_col: str = C.COL_LOCAL_DT) -> pd.DataFrame:
    out = df.copy()
    t = out[time_col]
    out["hour"] = t.dt.hour
    out["dayofweek"] = t.dt.dayofweek  # Monday=0
    out["month"] = t.dt.month
    out["date"] = t.dt.floor("D")
    return out


def add_harmattan_flag(
    df: pd.DataFrame,
    time_col: str = C.COL_LOCAL_DT,
    start: str = "2025-12-01",
    end: str = "2026-02-28",
) -> pd.DataFrame:
    """Simple date-based Harmattan proxy; adjust boundaries as needed."""
    out = df.copy()
    t = out[time_col]
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    out["harmattan"] = (t >= start_dt) & (t <= end_dt)
    return out

