"""Build isolated per-horizon sequences for the 84-day regime LSTM+MHA run.

Writes to data/sequences_regime84/. Does not overwrite data/sequences/.
Does not build the 28-day / L=1344 file (n_train=0 on the 84-day split).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import project_path  # noqa: F401

from airpollution import constants as C
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.sequences import SequenceSpec, build_sequence_arrays
from airpollution.utils import ensure_dir

ROOT = project_path.ROOT
OUT = ensure_dir(ROOT / "data" / "sequences_regime84")
JOBS = (
    (24, 168, "h24"),
    (168, 336, "h168"),
    (336, 672, "h336"),
)


def main() -> None:
    existing_h24 = ROOT / "data" / "sequences" / "sequences_h24_L168.npz"
    df = None
    for h, sl, name in JOBS:
        out = OUT / f"sequences_{name}_L{sl}.npz"
        if out.exists():
            print(f"exists {out}", flush=True)
            continue
        if name == "h24" and existing_h24.exists():
            out.write_bytes(existing_h24.read_bytes())
            print(f"copied {existing_h24} -> {out}", flush=True)
            continue
        if df is None:
            df = preprocess(load_raw(), PreprocessOptions(use_local_time=True))
        spec = SequenceSpec(
            horizon_hours=h,
            seq_len=sl,
            time_col=C.COL_LOCAL_DT,
            add_location_onehot=True,
            add_time_features=True,
            add_fourier_daily=True,
            add_fourier_weekly=True,
        )
        X, y, target_time, meta, harm_y = build_sequence_arrays(df, spec)
        np.savez_compressed(
            out,
            X=X,
            y=y,
            harmattan_y=harm_y,
            target_time=target_time.astype("datetime64[ns]").to_numpy(),
            location=meta[C.COL_LOCATION_NAME].to_numpy(),
            horizon_hours=h,
            seq_len=sl,
        )
        print(f"Wrote {out} X={X.shape}", flush=True)


if __name__ == "__main__":
    main()
