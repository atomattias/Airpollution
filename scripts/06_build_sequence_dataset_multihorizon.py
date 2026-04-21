"""
Build a single multi-horizon sequence dataset for joint training.

This is an optional alternative to `06_build_sequence_dataset.py`.
Instead of writing one .npz per horizon, it writes one .npz that contains:
  - X: (N, seq_len, F)
  - y: (N, K) where K=len(HORIZONS)
  - target_time: label time for the *max* horizon (leakage-safe split key)

Run before: scripts/07_train_lstm_multihorizon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from airpollution import constants as C
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.sequences import MultiHorizonSequenceSpec, build_multihorizon_sequence_arrays
from airpollution.utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ensure_dir(ROOT / "data" / "sequences")

# Joint training horizons (hours ahead)
HORIZONS = [24, 168, 336, 672]


def main() -> None:
    df_raw = load_raw()
    df = preprocess(df_raw, PreprocessOptions(use_local_time=True))

    # For multi-horizon training, choose a lookback that is long enough for the longest horizon.
    # Default: 56 days (helps 28d-ahead event persistence + regime/seasonality).
    seq_len = 1344

    spec = MultiHorizonSequenceSpec(
        horizons_hours=tuple(HORIZONS),
        seq_len=seq_len,
        time_col=C.COL_LOCAL_DT,
        add_location_onehot=True,
        add_time_features=True,
        add_fourier_daily=True,
        add_fourier_weekly=True,
    )

    X, y, target_time, meta, harm_y = build_multihorizon_sequence_arrays(df, spec)
    out_path = OUT_DIR / f"sequences_mh_h{HORIZONS[-1]}_L{seq_len}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        harmattan_y=harm_y,
        target_time=target_time.astype("datetime64[ns]").to_numpy(),
        location=meta[C.COL_LOCATION_NAME].to_numpy(),
        horizons_hours=np.asarray(HORIZONS, dtype=np.int32),
        seq_len=seq_len,
        feature_note="see sequences.py for engineered features",
    )
    print(f"Wrote {out_path} X={X.shape} y={y.shape} seq_len={seq_len} horizons={HORIZONS}")


if __name__ == "__main__":
    main()

