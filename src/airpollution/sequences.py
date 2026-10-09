"""Build leakage-safe sequence tensors for direct multi-horizon forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .features import add_harmattan_flag, add_time_columns


@dataclass(frozen=True)
class SequenceSpec:
    horizon_hours: int
    seq_len: int
    time_col: str = C.COL_LOCAL_DT
    add_location_onehot: bool = True
    add_time_features: bool = True
    add_fourier_daily: bool = True
    add_fourier_weekly: bool = True


@dataclass(frozen=True)
class MultiHorizonSequenceSpec:
    horizons_hours: tuple[int, ...]
    seq_len: int
    time_col: str = C.COL_LOCAL_DT
    add_location_onehot: bool = True
    add_time_features: bool = True
    add_fourier_daily: bool = True
    add_fourier_weekly: bool = True


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


def _cyclic_sin_cos(x: pd.Series, period: float, name: str) -> pd.DataFrame:
    """Return sin/cos cyclic encoding for integer-like series."""
    xf = x.astype(float)
    ang = 2.0 * np.pi * (xf / period)
    return pd.DataFrame({f"{name}_sin": np.sin(ang), f"{name}_cos": np.cos(ang)}, index=x.index)


def _fourier_terms(t: pd.Series, period_hours: float, n_harmonics: int = 3, prefix: str | None = None) -> pd.DataFrame:
    """Fourier terms on absolute time (mirrors tabular design)."""
    pref = prefix or f"{int(period_hours)}h"
    hours = (t.astype("int64") / 1e9) / 3600.0
    out: dict[str, np.ndarray] = {}
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * (hours / period_hours)
        out[f"sin_{pref}_{k}"] = np.sin(angle)
        out[f"cos_{pref}_{k}"] = np.cos(angle)
    return pd.DataFrame(out, index=t.index)


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

    df = add_time_columns(add_harmattan_flag(df, time_col=spec.time_col), time_col=spec.time_col)
    if "harmattan" in df.columns:
        df["harmattan"] = df["harmattan"].astype(np.float32)
        if "harmattan" not in fc:
            fc = list(fc) + ["harmattan"]

    # Add calendar + Fourier features (shared across all locations).
    if spec.add_time_features and spec.time_col in df.columns:
        df = df.copy()
        cyc = [
            _cyclic_sin_cos(df["hour"], period=24.0, name="hour"),
            _cyclic_sin_cos(df["dayofweek"], period=7.0, name="dow"),
            _cyclic_sin_cos(df["month"], period=12.0, name="month"),
        ]
        df = pd.concat([df] + cyc, axis=1)
        for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            if c not in fc:
                fc.append(c)

        if spec.add_fourier_daily:
            ft_d = _fourier_terms(df[spec.time_col], period_hours=24.0, n_harmonics=3, prefix="24h")
            df = pd.concat([df, ft_d], axis=1)
            for c in ft_d.columns:
                if c not in fc:
                    fc.append(c)

        if spec.add_fourier_weekly:
            ft_w = _fourier_terms(df[spec.time_col], period_hours=24.0 * 7.0, n_harmonics=3, prefix="168h")
            df = pd.concat([df, ft_w], axis=1)
            for c in ft_w.columns:
                if c not in fc:
                    fc.append(c)

    df = df.sort_values([C.COL_LOCATION_NAME, spec.time_col]).reset_index(drop=True)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    tt_list: list[pd.Timestamp] = []
    loc_list: list[str] = []
    harm_list: list[float] = []

    H = spec.horizon_hours
    L = spec.seq_len
    has_harm = "harmattan" in df.columns

    # Location one-hot mapping (kept stable across all horizons).
    loc_names = sorted(df[C.COL_LOCATION_NAME].astype(str).unique().tolist())
    loc_index = {name: i for i, name in enumerate(loc_names)}
    n_loc = len(loc_names)

    for loc, g in df.groupby(C.COL_LOCATION_NAME, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        pm = g[C.COL_PM25_CORR].to_numpy(dtype=float)
        times = pd.to_datetime(g[spec.time_col])
        feat = g[fc].to_numpy(dtype=float)
        har = g["harmattan"].astype(np.float32).to_numpy() if has_harm else np.zeros(n, dtype=np.float32)

        # Append location identity as constant one-hot over time.
        if spec.add_location_onehot:
            oh = np.zeros((n, n_loc), dtype=float)
            oh[:, loc_index[str(loc)]] = 1.0
            feat = np.concatenate([feat, oh], axis=1)

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


def build_multihorizon_sequence_arrays(
    df: pd.DataFrame,
    spec: MultiHorizonSequenceSpec,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.DataFrame, np.ndarray]:
    """
    Construct one dataset where each row predicts multiple horizons.

    Splitting:
      `target_time` is the label time for the *maximum* horizon in `spec.horizons_hours`.
      This ensures blocked splits are leakage-safe for all horizons simultaneously.

    Returns:
        X: (N, seq_len, F) float32
        y: (N, K) float32 — PM2.5 corrected at each label time (t + horizons[k])
        target_time: (N,) datetime64 at label time for max horizon
        meta: DataFrame with location name per row
        harmattan_y: (N,) float 0/1 at label time for max horizon if available, else zeros
    """
    horizons = tuple(int(h) for h in spec.horizons_hours)
    if not horizons:
        raise ValueError("MultiHorizonSequenceSpec.horizons_hours is empty.")
    if any(h <= 0 for h in horizons):
        raise ValueError("All horizons must be positive integers (hours).")

    # Reuse the single-horizon feature engineering logic by building a temporary SequenceSpec
    # that matches the toggles. We then compute multi-target y in a single pass.
    tmp = SequenceSpec(
        horizon_hours=max(horizons),
        seq_len=spec.seq_len,
        time_col=spec.time_col,
        add_location_onehot=spec.add_location_onehot,
        add_time_features=spec.add_time_features,
        add_fourier_daily=spec.add_fourier_daily,
        add_fourier_weekly=spec.add_fourier_weekly,
    )

    fc = feature_cols or default_feature_columns(df)
    if not fc:
        raise ValueError("No feature columns available for sequences.")

    # Apply the same feature augmentation used by `build_sequence_arrays` (harmattan + time + Fourier + loc).
    df2 = add_time_columns(add_harmattan_flag(df, time_col=tmp.time_col), time_col=tmp.time_col)
    if "harmattan" in df2.columns:
        df2["harmattan"] = df2["harmattan"].astype(np.float32)
        if "harmattan" not in fc:
            fc = list(fc) + ["harmattan"]

    if tmp.add_time_features and tmp.time_col in df2.columns:
        df2 = df2.copy()
        cyc = [
            _cyclic_sin_cos(df2["hour"], period=24.0, name="hour"),
            _cyclic_sin_cos(df2["dayofweek"], period=7.0, name="dow"),
            _cyclic_sin_cos(df2["month"], period=12.0, name="month"),
        ]
        df2 = pd.concat([df2] + cyc, axis=1)
        for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            if c not in fc:
                fc.append(c)

        if tmp.add_fourier_daily:
            ft_d = _fourier_terms(df2[tmp.time_col], period_hours=24.0, n_harmonics=3, prefix="24h")
            df2 = pd.concat([df2, ft_d], axis=1)
            for c in ft_d.columns:
                if c not in fc:
                    fc.append(c)

        if tmp.add_fourier_weekly:
            ft_w = _fourier_terms(df2[tmp.time_col], period_hours=24.0 * 7.0, n_harmonics=3, prefix="168h")
            df2 = pd.concat([df2, ft_w], axis=1)
            for c in ft_w.columns:
                if c not in fc:
                    fc.append(c)

    df2 = df2.sort_values([C.COL_LOCATION_NAME, tmp.time_col]).reset_index(drop=True)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    tt_list: list[pd.Timestamp] = []
    loc_list: list[str] = []
    harm_list: list[float] = []

    L = tmp.seq_len
    Hmax = max(horizons)
    K = len(horizons)
    has_harm = "harmattan" in df2.columns

    loc_names = sorted(df2[C.COL_LOCATION_NAME].astype(str).unique().tolist())
    loc_index = {name: i for i, name in enumerate(loc_names)}
    n_loc = len(loc_names)

    for loc, g in df2.groupby(C.COL_LOCATION_NAME, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        pm = g[C.COL_PM25_CORR].to_numpy(dtype=float)
        times = pd.to_datetime(g[tmp.time_col])
        feat = g[fc].to_numpy(dtype=float)
        har = g["harmattan"].astype(np.float32).to_numpy() if has_harm else np.zeros(n, dtype=np.float32)

        if tmp.add_location_onehot:
            oh = np.zeros((n, n_loc), dtype=float)
            oh[:, loc_index[str(loc)]] = 1.0
            feat = np.concatenate([feat, oh], axis=1)

        for end in range(L - 1, n - Hmax):
            start = end - L + 1
            window = feat[start : end + 1]
            if not np.isfinite(window).all():
                continue

            # Multi-target y (one per horizon), and split key by max-horizon label time.
            target_idx_max = end + Hmax
            yv = np.empty((K,), dtype=np.float32)
            ok = True
            for i, h in enumerate(horizons):
                target_idx = end + h
                val = pm[target_idx]
                if not np.isfinite(val):
                    ok = False
                    break
                yv[i] = float(val)
            if not ok:
                continue

            X_list.append(window.astype(np.float32))
            y_list.append(yv)
            tt_list.append(times.iloc[target_idx_max])
            loc_list.append(str(loc))
            harm_list.append(float(har[target_idx_max]))

    if not X_list:
        raise ValueError("No valid multi-horizon sequence samples; check seq_len/horizons/missing data.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    target_time = pd.Series(tt_list)
    meta = pd.DataFrame({C.COL_LOCATION_NAME: loc_list})
    harmattan_y = np.asarray(harm_list, dtype=np.float32)
    return X, y, target_time, meta, harmattan_y
