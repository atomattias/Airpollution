from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import constants as C


@dataclass(frozen=True)
class LoadOptions:
    csv_path: str = C.RAW_CSV_PATH
    parse_timestamps: bool = True
    drop_empty_location_name: bool = True


def load_raw(options: LoadOptions | None = None) -> pd.DataFrame:
    """Load the raw CSV with minimal type normalization."""
    opts = options or LoadOptions()

    df = pd.read_csv(
        opts.csv_path,
        low_memory=False,
    )

    if opts.drop_empty_location_name:
        df = df[df[C.COL_LOCATION_NAME].notna()].copy()

    # Normalize a few obvious string quirks early.
    df[C.COL_LOCATION_NAME] = df[C.COL_LOCATION_NAME].astype(str).str.strip()

    if opts.parse_timestamps:
        # Local time is used for diurnal analysis; UTC is useful for cross-checks.
        df[C.COL_LOCAL_DT] = pd.to_datetime(df[C.COL_LOCAL_DT], errors="coerce")
        df[C.COL_UTC_DT] = pd.to_datetime(df[C.COL_UTC_DT], errors="coerce")

    return df


def sort_by_location_time(df: pd.DataFrame, use_local_time: bool = True) -> pd.DataFrame:
    tcol = C.COL_LOCAL_DT if use_local_time else C.COL_UTC_DT
    return df.sort_values([C.COL_LOCATION_NAME, tcol]).reset_index(drop=True)

