"""
Summarise test-set target variability by forecast horizon (tabular label times).

Helps interpret horizon-specific difficulty (e.g. apparent 14d MAE behaviour).
Reads tabular feature parquet per horizon, applies the same leakage-safe split as script 05.

Outputs:
  - reports/tables/results_horizon_target_variability_test.csv
  - reports/figures/results_horizon_target_variability_box.png (optional, non-fatal)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import project_path  # noqa: F401

from airpollution.eval import SplitConfig, time_split_by_target_time

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"
TABLES_DIR = ROOT / "reports" / "tables"
FIGS_DIR = ROOT / "reports" / "figures"

HORIZONS = ["h24", "h168", "h336", "h672"]


def _acf1(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    x = x - np.mean(x)
    num = np.dot(x[:-1], x[1:])
    den = np.dot(x, x) + 1e-12
    return float(num / den)


def main() -> None:
    cfg = SplitConfig(
        val_days=int(os.environ.get("AIRP_VAL_DAYS", "14")),
        test_days=int(os.environ.get("AIRP_TEST_DAYS", "28")),
    )

    rows: list[dict] = []
    series_for_plot: dict[str, np.ndarray] = {}

    for hz in HORIZONS:
        path_parquet = FEATURE_DIR / f"tabular_{hz}.parquet"
        path_gz = FEATURE_DIR / f"tabular_{hz}.csv.gz"
        ds = None
        if path_parquet.exists():
            try:
                ds = pd.read_parquet(path_parquet)
            except ImportError:
                print(f"[{hz}] Parquet engine missing; try {path_gz.name} or install pyarrow.", flush=True)
                ds = None
        if ds is None:
            if not path_gz.exists():
                print(f"Skip {hz}: missing {path_parquet.name} and {path_gz.name}")
                continue
            ds = pd.read_csv(path_gz)
        ds["target_time"] = pd.to_datetime(ds["target_time"])
        _, _, test, _ = time_split_by_target_time(ds, cfg=cfg)
        y = test["y"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 5:
            continue
        q25, q50, q75 = np.percentile(y, [25, 50, 75])
        rows.append(
            {
                "horizon": hz,
                "n_test": int(len(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "variance": float(np.var(y)),
                "iqr": float(q75 - q25),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "acf_lag1": _acf1(y),
            }
        )
        series_for_plot[hz] = y

    if not rows:
        raise SystemExit("No variability rows produced (missing feature files?).")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES_DIR / "results_horizon_target_variability_test.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIGS_DIR.mkdir(parents=True, exist_ok=True)
        data = [series_for_plot[h] for h in HORIZONS if h in series_for_plot]
        labels = [h for h in HORIZONS if h in series_for_plot]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$) test $y$")
        ax.set_xlabel("Horizon (tabular label)")
        ax.set_title("Test-set target distribution by horizon")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = FIGS_DIR / "results_horizon_target_variability_box.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Wrote {p}")
    except Exception as e:
        print(f"Figure skipped: {e}")


if __name__ == "__main__":
    main()
