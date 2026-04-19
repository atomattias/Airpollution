from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

from airpollution import constants as C
from airpollution.io import load_raw, sort_by_location_time
from airpollution.utils import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ensure_dir(ROOT / "reports" / "tables")
FIGS_DIR = ensure_dir(ROOT / "reports" / "figures")


def main() -> None:
    df = sort_by_location_time(load_raw(), use_local_time=True)

    # Basic per-location coverage stats
    g = df.groupby(C.COL_LOCATION_NAME, dropna=False)
    coverage = g.agg(
        n_rows=(C.TARGET_COL, "size"),
        start=(C.COL_LOCAL_DT, "min"),
        end=(C.COL_LOCAL_DT, "max"),
        n_unique_times=(C.COL_LOCAL_DT, "nunique"),
    ).reset_index()
    coverage["expected_hourly_rows"] = (
        (coverage["end"] - coverage["start"]).dt.total_seconds() / 3600.0
    ).round().astype("Int64") + 1
    coverage["completeness"] = (
        coverage["n_unique_times"] / coverage["expected_hourly_rows"]
    ).astype(float)

    coverage.to_csv(TABLES_DIR / "qc_location_coverage.csv", index=False)

    # Missingness summary overall
    cols_of_interest = [
        C.COL_LOCATION_NAME,
        C.COL_LOCAL_DT,
        C.COL_UTC_DT,
        C.COL_AGG_RECORDS,
        C.COL_PLACE_OPEN,
        C.COL_PM25_RAW,
        C.COL_PM25_CORR,
        C.COL_PM1,
        C.COL_PM10,
        C.COL_PARTICLE_03,
        C.COL_CO2_CORR,
        C.COL_TEMP_CORR,
        C.COL_HUMIDITY_CORR,
        C.COL_HEAT_INDEX,
        C.COL_TVOC,
        C.COL_TVOC_INDEX,
        C.COL_NOX_INDEX,
    ]
    cols_present = [c for c in cols_of_interest if c in df.columns]
    miss = (
        df[cols_present]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .to_frame()
        .reset_index(names="column")
    )
    miss["missing_pct"] = (miss["missing_rate"] * 100).round(2)
    miss.to_csv(TABLES_DIR / "qc_missingness_overall.csv", index=False)

    # Duplicates by (location, local time)
    dup_mask = df.duplicated([C.COL_LOCATION_NAME, C.COL_LOCAL_DT], keep=False)
    dups = df.loc[dup_mask, [C.COL_LOCATION_NAME, C.COL_LOCAL_DT]].copy()
    dups = dups.value_counts().rename("count").reset_index()
    dups.to_csv(TABLES_DIR / "qc_duplicates_location_time.csv", index=False)

    # Suspicious zeros in PM2.5 corrected (likely sensor dropouts for some sites)
    pm0 = (
        df.assign(pm25_zero=(df[C.COL_PM25_CORR] == 0))
        .groupby(C.COL_LOCATION_NAME)["pm25_zero"]
        .agg(zero_count="sum", n="size")
        .reset_index()
    )
    pm0["zero_rate"] = pm0["zero_count"] / pm0["n"]
    pm0.sort_values("zero_rate", ascending=False).to_csv(
        TABLES_DIR / "qc_pm25_zero_rates.csv", index=False
    )

    # Missingness over time (overall) – plot daily missing rate for key sensors.
    key_cols = [C.COL_PM25_CORR, C.COL_TVOC, C.COL_NOX_INDEX]
    key_cols = [c for c in key_cols if c in df.columns]
    daily = (
        df.set_index(C.COL_LOCAL_DT)[key_cols]
        .isna()
        .resample("D")
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 4))
    for c in key_cols:
        plt.plot(daily[C.COL_LOCAL_DT], daily[c], label=f"missing({c})")
    plt.ylim(0, 1)
    plt.ylabel("Missing rate")
    plt.xlabel("Date (local)")
    plt.title("Daily missingness rate (key variables)")
    plt.legend(fontsize=8, ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "qc_missingness_daily.png", dpi=200)
    plt.close()

    print("Wrote QC outputs to reports/tables and reports/figures.")


if __name__ == "__main__":
    main()

