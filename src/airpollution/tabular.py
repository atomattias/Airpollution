from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as C
from .features import add_harmattan_flag, add_time_columns


@dataclass(frozen=True)
class TabularFeatureSpec:
    # Lags in hours for particulate history
    lags_hours: tuple[int, ...] = tuple(list(range(1, 49)) + [72, 96, 120, 144, 168])
    # Rolling windows in hours
    rolling_windows: tuple[int, ...] = (3, 6, 24, 72, 168)
    # Which numeric columns to create lag/rolling features from (if present)
    base_series_cols: tuple[str, ...] = (
        C.COL_PM25_CORR,
        C.COL_PM1,
        C.COL_PM10,
        C.COL_PARTICLE_03,
        C.COL_TEMP_CORR,
        C.COL_HUMIDITY_CORR,
        C.COL_CO2_CORR,
    )
    add_fourier_daily: bool = True
    add_fourier_weekly: bool = True


def _fourier_terms(t: pd.Series, period_hours: float, n_harmonics: int = 3) -> pd.DataFrame:
    # t is datetime-like series
    hours = (t.astype("int64") / 1e9) / 3600.0
    out = {}
    for k in range(1, n_harmonics + 1):
        angle = 2 * np.pi * k * (hours / period_hours)
        out[f"sin_{int(period_hours)}h_{k}"] = np.sin(angle)
        out[f"cos_{int(period_hours)}h_{k}"] = np.cos(angle)
    return pd.DataFrame(out, index=t.index)


def make_supervised_tabular(
    df: pd.DataFrame,
    horizon_hours: int,
    spec: TabularFeatureSpec | None = None,
    time_col: str = C.COL_LOCAL_DT,
    add_location_onehot: bool = True,
) -> pd.DataFrame:
    """
    Create a single supervised tabular dataset for a fixed horizon.

    Output columns:
    - meta: location, time
    - y: target at t+horizon
    - X_*: engineered features at time t
    """
    s = spec or TabularFeatureSpec()
    df = add_time_columns(add_harmattan_flag(df, time_col=time_col), time_col=time_col)
    df = df.sort_values([C.COL_LOCATION_NAME, time_col]).reset_index(drop=True)

    base_cols = [c for c in s.base_series_cols if c in df.columns]
    keep_cols = [C.COL_LOCATION_NAME, time_col, C.COL_PM25_CORR, "hour", "dayofweek", "month", "harmattan"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols + [c for c in base_cols if c not in keep_cols]].copy()

    parts = []
    for loc, loc_df in df.groupby(C.COL_LOCATION_NAME, sort=False):
        loc_df = loc_df.sort_values(time_col).copy()

        # Target at t + horizon
        loc_df["y"] = loc_df[C.COL_PM25_CORR].shift(-horizon_hours)
        loc_df["target_time"] = loc_df[time_col].shift(-horizon_hours)

        # Lag features
        for col in base_cols:
            series = loc_df[col]
            for lag in s.lags_hours:
                loc_df[f"lag_{col}_{lag}h"] = series.shift(lag)

        # Rolling stats (use history only -> shift by 1)
        for col in base_cols:
            series = loc_df[col]
            shifted = series.shift(1)
            for w in s.rolling_windows:
                roll = shifted.rolling(window=w, min_periods=max(2, w // 3))
                loc_df[f"roll_mean_{col}_{w}h"] = roll.mean()
                loc_df[f"roll_std_{col}_{w}h"] = roll.std()
                loc_df[f"roll_max_{col}_{w}h"] = roll.max()

        parts.append(loc_df)

    out = pd.concat(parts, ignore_index=True)

    # Fourier terms on absolute time
    if s.add_fourier_daily or s.add_fourier_weekly:
        ft = []
        if s.add_fourier_daily:
            ft.append(_fourier_terms(out[time_col], period_hours=24.0, n_harmonics=3))
        if s.add_fourier_weekly:
            ft.append(_fourier_terms(out[time_col], period_hours=24.0 * 7.0, n_harmonics=3))
        out = pd.concat([out.reset_index(drop=True)] + [x.reset_index(drop=True) for x in ft], axis=1)

    # One-hot location (for pooled models)
    if add_location_onehot:
        loc_oh = pd.get_dummies(out[C.COL_LOCATION_NAME], prefix="loc", dtype=float)
        out = pd.concat([out, loc_oh], axis=1)

    # Drop rows with missing target or missing key features
    out = out[out["y"].notna() & out["target_time"].notna()].copy()

    return out

