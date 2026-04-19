from __future__ import annotations

from pathlib import Path

import pandas as pd

from airpollution import constants as C
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.tabular import TabularFeatureSpec, make_supervised_tabular
from airpollution.utils import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ensure_dir(ROOT / "data" / "features")

HORIZONS_HOURS = {
    "h24": 24,
    "h168": 168,
    "h336": 336,
    "h672": 672,
}


def main() -> None:
    df_raw = load_raw()
    df = preprocess(df_raw, PreprocessOptions(use_local_time=True))

    # Save the preprocessed hourly-aligned dataset for reuse
    proc_path = ROOT / "data" / "processed" / "tema_hourly_preprocessed.parquet"
    ensure_dir(proc_path.parent)
    df.to_parquet(proc_path, index=False)

    spec = TabularFeatureSpec()
    for name, h in HORIZONS_HOURS.items():
        ds = make_supervised_tabular(df, horizon_hours=h, spec=spec, time_col=C.COL_LOCAL_DT)
        out_path = OUT_DIR / f"tabular_{name}.parquet"
        ds.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} with shape={ds.shape}")


if __name__ == "__main__":
    main()

