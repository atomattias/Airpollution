from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from airpollution import constants as C
from airpollution.features import add_harmattan_flag
from airpollution.io import load_raw, sort_by_location_time
from airpollution.utils import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ensure_dir(ROOT / "reports" / "tables")
FIGS_DIR = ensure_dir(ROOT / "reports" / "figures")


def skewness(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3:
        return float("nan")
    return float(x.skew())


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = sort_by_location_time(load_raw(), use_local_time=True)
    df = add_harmattan_flag(df)

    # Summary table per location for corrected PM2.5
    g = df.groupby(C.COL_LOCATION_NAME)[C.COL_PM25_CORR]
    stats = g.agg(
        mean="mean",
        std="std",
        p25=lambda s: s.quantile(0.25),
        median="median",
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
        p95=lambda s: s.quantile(0.95),
        max="max",
        n="count",
    ).reset_index()
    stats["iqr"] = stats["p75"] - stats["p25"]
    stats["skewness"] = g.apply(skewness).values
    stats.to_csv(TABLES_DIR / "eda_pm25_by_location_summary.csv", index=False)

    # Boxplot by location
    plt.figure(figsize=(10, 4.5))
    order = stats.sort_values("mean", ascending=False)[C.COL_LOCATION_NAME].tolist()
    sns.boxplot(
        data=df,
        x=C.COL_LOCATION_NAME,
        y=C.COL_PM25_CORR,
        order=order,
        showfliers=False,
    )
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("PM2.5 corrected (μg/m³)")
    plt.xlabel("")
    plt.title("Distribution of corrected PM2.5 by location")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "eda_pm25_boxplot_by_location.png", dpi=250)
    plt.close()

    # Regime split boxplot (pre-Harmattan vs Harmattan)
    # Keep it compact: pooled + per-location small multiples is too big; use grouped boxplot.
    df_reg = df[[C.COL_LOCATION_NAME, C.COL_PM25_CORR, "harmattan"]].copy()
    df_reg["regime"] = np.where(df_reg["harmattan"], "Harmattan", "Pre-Harmattan")
    plt.figure(figsize=(10, 4.8))
    sns.boxplot(
        data=df_reg,
        x=C.COL_LOCATION_NAME,
        y=C.COL_PM25_CORR,
        hue="regime",
        order=order,
        showfliers=False,
    )
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("PM2.5 corrected (μg/m³)")
    plt.xlabel("")
    plt.title("Corrected PM2.5 by location, split by regime")
    plt.legend(title="", fontsize=8, frameon=False, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "eda_pm25_boxplot_by_location_regime.png", dpi=250)
    plt.close()

    print("Wrote EDA distribution outputs to reports/tables and reports/figures.")


if __name__ == "__main__":
    main()

