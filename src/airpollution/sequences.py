"""Build leakage-safe sequence tensors for direct multi-horizon forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .features import add_harmattan_flag


@dataclass(frozen=True)
class SequenceSpec:
    horizon_hours: int
    seq_len: int
    time_col: str = C.COL_LOCAL_DT


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        C.COL_PM25_CORR,
        C.COL_PM1,
        C.COL_PM10,
        C.COL_PARTICLE_03,
        C.COL_TEMP_CORR,
        C.COL_HUMIDITY_CORR,
        C.COL_CO2_CORR,
        C.COL_HEAT_INDEX,
        C.COL_TVOC,
        C.COL_TVOC_INDEX,
        C.COL_NOX_INDEX,
    ]
    return [c for c in cols if c in df.columns]


def build_sequence_arrays(
    df: pd.DataFrame,
    spec: SequenceSpec,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame, np.ndarray]:
    """
    For each location, construct windows ending at time t that predict PM2.5 at t+H.

    Returns:
        X: (N, seq_len, F) float32
        y: (N,) float32 — PM2.5 corrected at label time
        target_time: (N,) datetime64 for split (leakage-safe)
        meta: DataFrame with location name per row (for stratified metrics)
        harmattan_y: (N,) float 0/1 at label time if available, else zeros
    """
    fc = feature_cols or default_feature_columns(df)
    if not fc:
        raise ValueError("No feature columns available for sequences.")

    df = add_harmattan_flag(df, time_col=spec.time_col)
    if "harmattan" in df.columns:
        df["harmattan"] = df["harmattan"].astype(np.float32)
        if "harmattan" not in fc:
            fc = list(fc) + ["harmattan"]

    df = df.sort_values([C.COL_LOCATION_NAME, spec.time_col]).reset_index(drop=True)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    tt_list: list[pd.Timestamp] = []
    loc_list: list[str] = []
    harm_list: list[float] = []

    H = spec.horizon_hours
    L = spec.seq_len
    has_harm = "harmattan" in df.columns

    for loc, g in df.groupby(C.COL_LOCATION_NAME, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        pm = g[C.COL_PM25_CORR].to_numpy(dtype=float)
        times = pd.to_datetime(g[spec.time_col])
        feat = g[fc].to_numpy(dtype=float)
        har = g["harmattan"].astype(np.float32).to_numpy() if has_harm else np.zeros(n, dtype=np.float32)

        # End index of window (inclusive): last observed step is at index `end`
        for end in range(L - 1, n - H):
            start = end - L + 1
            window = feat[start : end + 1]
            if not np.isfinite(window).all():
                continue
            target_idx = end + H
            yv = pm[target_idx]
            if not np.isfinite(yv):
                continue
            X_list.append(window.astype(np.float32))
            y_list.append(float(yv))
            tt_list.append(times.iloc[target_idx])
            loc_list.append(str(loc))
            harm_list.append(float(har[target_idx]))

    if not X_list:
        raise ValueError("No valid sequence samples; check seq_len, horizon, or missing data.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    target_time = pd.Series(tt_list)
    meta = pd.DataFrame({C.COL_LOCATION_NAME: loc_list})
    harmattan_y = np.asarray(harm_list, dtype=np.float32)
    return X, y, target_time, meta, harmattan_y
