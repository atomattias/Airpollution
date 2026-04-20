from __future__ import annotations

from pathlib import Path

import pandas as pd

import project_path  # noqa: F401 — side effect: repo src/ on sys.path

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
    proc_dir = ensure_dir(ROOT / "data" / "processed")
    proc_parquet = proc_dir / "tema_hourly_preprocessed.parquet"
    proc_csv = proc_dir / "tema_hourly_preprocessed.csv.gz"
    # Always write a compressed CSV so the pipeline works without Parquet engines.
    df.to_csv(proc_csv, index=False, compression="gzip")
    # Best-effort Parquet (may fail if pyarrow/fastparquet missing).
    try:
        df.to_parquet(proc_parquet, index=False)
    except Exception:
        pass

    spec = TabularFeatureSpec()
    for name, h in HORIZONS_HOURS.items():
        ds = make_supervised_tabular(df, horizon_hours=h, spec=spec, time_col=C.COL_LOCAL_DT)
        out_parquet = OUT_DIR / f"tabular_{name}.parquet"
        out_csv = OUT_DIR / f"tabular_{name}.csv.gz"
        ds.to_csv(out_csv, index=False, compression="gzip")
        try:
            ds.to_parquet(out_parquet, index=False)
        except Exception:
            pass
        print(f"Wrote {out_csv.name} (and Parquet if available) with shape={ds.shape}")


if __name__ == "__main__":
    main()

