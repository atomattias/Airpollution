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
from airpollution.features import add_harmattan_flag, add_time_columns
from airpollution.io import load_raw, sort_by_location_time
from airpollution.utils import ensure_dir


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ensure_dir(ROOT / "reports" / "tables")
FIGS_DIR = ensure_dir(ROOT / "reports" / "figures")


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = sort_by_location_time(load_raw(), use_local_time=True)
    df = add_time_columns(add_harmattan_flag(df))

    # Pooled diurnal summary
    diurnal = (
        df.groupby("hour")[C.COL_PM25_CORR]
        .agg(
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    diurnal.to_csv(TABLES_DIR / "eda_diurnal_pooled.csv", index=False)

    plt.figure(figsize=(7.5, 4))
    plt.plot(diurnal["hour"], diurnal["median"], label="Median", color="#1f77b4")
    plt.fill_between(diurnal["hour"], diurnal["p25"], diurnal["p75"], alpha=0.2, color="#1f77b4", label="IQR")
    plt.xticks(range(0, 24, 3))
    plt.xlabel("Hour of day (local)")
    plt.ylabel("PM2.5 corrected (μg/m³)")
    plt.title("Diurnal pattern of corrected PM2.5 (pooled)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "eda_diurnal_curve_pooled.png", dpi=250)
    plt.close()

    # Diurnal curves per location (facets)
    per_loc = (
        df.groupby([C.COL_LOCATION_NAME, "hour"])[C.COL_PM25_CORR]
        .median()
        .reset_index()
        .rename(columns={C.COL_PM25_CORR: "median_pm25"})
    )
    per_loc.to_csv(TABLES_DIR / "eda_diurnal_by_location_median.csv", index=False)

    g = sns.FacetGrid(
        per_loc,
        col=C.COL_LOCATION_NAME,
        col_wrap=4,
        height=2.2,
        sharey=True,
        sharex=True,
    )
    g.map_dataframe(sns.lineplot, x="hour", y="median_pm25", color="#1f77b4")
    g.set_titles("{col_name}", size=9)
    g.set_axis_labels("Hour", "Median PM2.5")
    for ax in g.axes.flatten():
        ax.set_xticks(range(0, 24, 6))
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "eda_diurnal_curves_by_location.png", dpi=250)
    plt.close()

    # Hour x Day-of-week heatmap (pooled median)
    heat = (
        df.pivot_table(
            index="dayofweek",
            columns="hour",
            values=C.COL_PM25_CORR,
            aggfunc="median",
        )
        .sort_index()
    )
    heat.to_csv(TABLES_DIR / "eda_heatmap_hour_x_dow_median.csv")

    plt.figure(figsize=(10, 3.2))
    sns.heatmap(heat, cmap="mako", cbar_kws={"label": "Median PM2.5"})
    plt.xlabel("Hour of day (local)")
    plt.ylabel("Day of week (Mon=0)")
    plt.title("Median corrected PM2.5 by hour × day-of-week (pooled)")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "eda_heatmap_hour_x_dow.png", dpi=250)
    plt.close()

    # Optional: regime-split pooled diurnal curves
    reg = (
        df.groupby(["harmattan", "hour"])[C.COL_PM25_CORR]
        .median()
        .reset_index()
        .rename(columns={C.COL_PM25_CORR: "median_pm25"})
    )
    reg["regime"] = np.where(reg["harmattan"], "Harmattan", "Pre-Harmattan")
    reg.to_csv(TABLES_DIR / "eda_diurnal_pooled_by_regime_median.csv", index=False)

    plt.figure(figsize=(7.5, 4))
    sns.lineplot(data=reg, x="hour", y="median_pm25", hue="regime")
    plt.xticks(range(0, 24, 3))
    plt.xlabel("Hour of day (local)")
    plt.ylabel("Median PM2.5 corrected (μg/m³)")
    plt.title("Diurnal pattern by regime (pooled)")
    plt.legend(frameon=False, title="")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "eda_diurnal_curve_by_regime.png", dpi=250)
    plt.close()

    print("Wrote EDA diurnal outputs to reports/tables and reports/figures.")


if __name__ == "__main__":
    main()

