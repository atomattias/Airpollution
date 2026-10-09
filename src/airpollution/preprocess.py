from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C
from .io import sort_by_location_time


@dataclass(frozen=True)
class PreprocessOptions:
    use_local_time: bool = True
    resample_hourly: bool = True
    max_interp_gap_hours: int = 3
    # If a location has a high fraction of pm25==0, treat zeros as missing.
    pm25_zero_as_missing_threshold: float = 0.10


def _set_suspicious_pm25_zeros_to_nan(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(C.COL_LOCATION_NAME)[C.COL_PM25_CORR]
    zero_rate = (g.apply(lambda s: (s == 0).mean())).rename("zero_rate")
    bad_locs = zero_rate[zero_rate >= threshold].index.tolist()
    if bad_locs:
        mask = out[C.COL_LOCATION_NAME].isin(bad_locs) & (out[C.COL_PM25_CORR] == 0)
        out.loc[mask, C.COL_PM25_CORR] = np.nan
        # For consistency, also null out related particulate variables when pm25 is missing.
        for c in [C.COL_PM25_RAW, C.COL_PM1, C.COL_PM10, C.COL_PARTICLE_03]:
            if c in out.columns:
                out.loc[mask, c] = np.nan
    return out


def _resample_location_hourly(loc_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    loc_df = loc_df.sort_values(time_col)
    loc_name = loc_df[C.COL_LOCATION_NAME].iloc[0]

    loc_df = loc_df.set_index(time_col)
    start = loc_df.index.min().floor("h")
    end = loc_df.index.max().ceil("h")
    idx = pd.date_range(start=start, end=end, freq="h")

    # Reindex to hourly timestamps; keep the last observation if duplicates exist.
    loc_df = loc_df[~loc_df.index.duplicated(keep="last")]
    loc_df = loc_df.reindex(idx)
    loc_df[C.COL_LOCATION_NAME] = loc_name
    return loc_df.reset_index(names=time_col)


def preprocess(df_raw: pd.DataFrame, options: PreprocessOptions | None = None) -> pd.DataFrame:
    opts = options or PreprocessOptions()
    time_col = C.COL_LOCAL_DT if opts.use_local_time else C.COL_UTC_DT

    df = sort_by_location_time(df_raw, use_local_time=opts.use_local_time).copy()

    # Remove rows with bad timestamps early.
    df = df[df[time_col].notna()].copy()

    # Clean suspicious zero blocks.
    df = _set_suspicious_pm25_zeros_to_nan(df, opts.pm25_zero_as_missing_threshold)

    if opts.resample_hourly:
        parts = []
        for _, loc_df in df.groupby(C.COL_LOCATION_NAME, sort=False):
            parts.append(_resample_location_hourly(loc_df, time_col=time_col))
        df = pd.concat(parts, ignore_index=True)

    # Interpolate short gaps per location on numeric columns.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [C.COL_LOCATION_ID]]

    parts = []
    for loc, loc_df in df.groupby(C.COL_LOCATION_NAME, sort=False):
        loc_df = loc_df.sort_values(time_col).copy()
        loc_df = loc_df.set_index(time_col)
        loc_df[numeric_cols] = loc_df[numeric_cols].interpolate(
            method="time",
            limit=opts.max_interp_gap_hours,
            limit_direction="both",
        )
        loc_df[C.COL_LOCATION_NAME] = loc
        parts.append(loc_df.reset_index())

    df = pd.concat(parts, ignore_index=True)

    return df

