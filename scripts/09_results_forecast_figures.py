"""Plot MAE by horizon for key models (reads reports/tables/results_merged_long.csv)."""
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
    long_path = TABLES_DIR / "results_merged_long.csv"
    if not long_path.exists():
        raise SystemExit(f"Missing {long_path}; run scripts/08_merge_results.py first.")

    df = pd.read_csv(long_path)
    has_ci = "mae_lower95" in df.columns and "mae_upper95" in df.columns

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

    rows: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for key, name in series_map.items():
        pipe, _, model = key.partition(":")
        mae_vals = []
        el, eh = [], []
        ok = True
        for hz in horizons:
            sub = df[(df["pipeline"] == pipe) & (df["model"] == model) & (df["horizon"] == hz)]
            if sub.empty:
                ok = False
                break
            row = sub.iloc[0]
            mae_vals.append(float(row["mae"]))
            if has_ci and pd.notna(row.get("mae_lower95")) and pd.notna(row.get("mae_upper95")):
                m = float(row["mae"])
                lo = float(row["mae_lower95"])
                hi = float(row["mae_upper95"])
                el.append(m - lo)
                eh.append(hi - m)
            else:
                el.append(0.0)
                eh.append(0.0)
        if not ok:
            continue
        yerr = None
        if has_ci and (sum(el) > 0 or sum(eh) > 0):
            yerr = np.vstack([np.asarray(el, dtype=float), np.asarray(eh, dtype=float)])
        rows.append((name, np.asarray(mae_vals, dtype=float), yerr))

    if not rows:
        raise SystemExit("No matching models in merged long metrics table.")

    x = np.arange(len(horizons))
    n = len(rows)
    width = min(0.8 / n, 0.14)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, vals, yerr) in enumerate(rows):
        offset = width * (i - (n - 1) / 2)
        ax.bar(
            x + offset,
            vals,
            width,
            label=name,
            yerr=yerr,
            capsize=2 if yerr is not None else 0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_h)
    ax.set_ylabel(r"MAE ($\mu$g/m$^3$)")
    ax.set_xlabel("Forecast horizon (direct)")
    title = "Out-of-sample MAE by horizon (pooled multi-site test)"
    if has_ci:
        title += "\n(error bars: block-bootstrap 95% CI for MAE)"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "results_mae_by_horizon.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
