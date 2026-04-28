"""Plot MAE by horizon for key models (reads reports/tables/results_merged_wide_mae.csv)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
FIGS_DIR = ROOT / "reports" / "figures"


def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "results_merged_wide_mae.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run scripts/08_merge_results.py first.")

    df = pd.read_csv(path)
    df = df.set_index("pipeline_model")

    series_map = {
        "deep:mh_lstm": "Joint LSTM (mh_lstm)",
        "deep:mh_lstm_mha": "Joint LSTM + MHA (mh_lstm_mha)",
        "deep:lstm": "LSTM (per-horizon)",
        "deep:lstm_mha": "LSTM + MHA (per-horizon)",
        "tabular:ridge_ar_lag_fourier": "Ridge AR+Fourier",
        "tabular:seasonal_naive_168h": "Seasonal naïve (168 h)",
        "tabular:linear_svr": "Linear SVR",
    }

    horizons = ["h24", "h168", "h336", "h672"]
    labels_h = ["24 h", "7 d", "14 d", "28 d"]

    rows = []
    for key, name in series_map.items():
        if key not in df.index:
            continue
        rows.append((name, df.loc[key, horizons].astype(float).values))

    if not rows:
        raise SystemExit("No matching models in wide MAE table.")

    x = np.arange(len(horizons))
    n = len(rows)
    width = min(0.8 / n, 0.14)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, vals) in enumerate(rows):
        offset = width * (i - (n - 1) / 2)
        ax.bar(x + offset, vals, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_h)
    ax.set_ylabel(r"MAE ($\mu$g/m$^3$)")
    ax.set_xlabel("Forecast horizon (direct)")
    ax.set_title("Out-of-sample MAE by horizon (pooled multi-site test)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "results_mae_by_horizon.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
