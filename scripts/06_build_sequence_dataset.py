"""Build NumPy sequence tensors for LSTM / LSTM+attention (one file per horizon)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from airpollution import constants as C
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.sequences import SequenceSpec, build_sequence_arrays
from airpollution.utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ensure_dir(ROOT / "data" / "sequences")

# Same horizons as tabular pipeline (hours ahead)
HORIZONS = {"h24": 24, "h168": 168, "h336": 336, "h672": 672}


def seq_len_for_horizon(h: int) -> int:
    """Lookback length in hours (longer memory for longer horizons)."""
    if h <= 24:
        return 168  # 7 days
    if h <= 168:
        return 336  # 14 days for weekly-ahead
    if h <= 336:
        return 672  # 28 days for 14d-ahead
    return 1344  # 56 days for 28d-ahead


def main() -> None:
    df_raw = load_raw()
    df = preprocess(df_raw, PreprocessOptions(use_local_time=True))

    proc_path = ROOT / "data" / "processed" / "tema_hourly_preprocessed.parquet"
    ensure_dir(proc_path.parent)
    df.to_parquet(proc_path, index=False)

    for name, h in HORIZONS.items():
        sl = seq_len_for_horizon(h)
        spec = SequenceSpec(
            horizon_hours=h,
            seq_len=sl,
            time_col=C.COL_LOCAL_DT,
            add_location_onehot=True,
            add_time_features=True,
            add_fourier_daily=True,
            add_fourier_weekly=True,
        )
        X, y, target_time, meta, harmattan_y = build_sequence_arrays(df, spec)

        out_path = OUT_DIR / f"sequences_{name}_L{sl}.npz"
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            harmattan_y=harmattan_y,
            target_time=target_time.astype("datetime64[ns]").to_numpy(),
            location=meta[C.COL_LOCATION_NAME].to_numpy(),
            horizon_hours=h,
            seq_len=sl,
            feature_note="see constants.py / default_feature_columns",
        )
        print(f"Wrote {out_path} X={X.shape} y={y.shape} seq_len={sl} horizon={h}h")


if __name__ == "__main__":
    main()
