"""
Block bootstrap confidence intervals for MAE, RMSE, and upper-decile MAE (MAE^90).

Reads per-row prediction CSVs under reports/predictions/ (written by scripts 05 and 07).
Respects temporal dependence via moving block bootstrap on test row indices.

Env:
  AIRP_BOOTSTRAP_B (default 1000) — number of bootstrap samples
  AIRP_BOOTSTRAP_BLOCK_LEN (default 168) — block length in hours (rows)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import project_path  # noqa: F401

from airpollution.eval import top_decile_mask

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "reports" / "predictions"
OUT_CSV = ROOT / "reports" / "tables" / "results_confidence_intervals.csv"


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    block_len = max(1, min(int(block_len), n))
    if n <= block_len:
        return rng.choice(n, size=n, replace=True)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx: list[int] = []
    for s in starts:
        idx.extend(range(int(s), int(s) + block_len))
    return np.asarray(idx[:n], dtype=int)


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    if len(yt) < 5:
        return float("nan"), float("nan"), float("nan")
    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))
    td = top_decile_mask(yt)
    if td.any():
        m90 = float(mean_absolute_error(yt[td], yp[td]))
    else:
        m90 = float("nan")
    return mae, rmse, m90


def _bootstrap_distributions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int,
    block_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes = np.full(n_boot, np.nan)
    rmses = np.full(n_boot, np.nan)
    m90s = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = _block_bootstrap_indices(n, block_len, rng)
        maes[b], rmses[b], m90s[b] = _point_metrics(y_true[idx], y_pred[idx])
    return maes, rmses, m90s


def _quantiles(x: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def main() -> None:
    if not PRED_DIR.is_dir():
        raise SystemExit(f"Missing predictions directory {PRED_DIR}; run 05_train_tabular_models.py / 07_*.py first.")

    n_boot = int(os.environ.get("AIRP_BOOTSTRAP_B", "1000"))
    block_len = int(os.environ.get("AIRP_BOOTSTRAP_BLOCK_LEN", "168"))
    seed = int(os.environ.get("AIRP_BOOTSTRAP_SEED", "42"))

    rows: list[dict] = []
    for path in sorted(PRED_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        need = {"y_true", "y_pred", "pipeline", "model", "horizon"}
        if not need.issubset(set(df.columns)):
            continue
        yt = df["y_true"].to_numpy(dtype=float)
        yp = df["y_pred"].to_numpy(dtype=float)
        pipeline = str(df["pipeline"].iloc[0])
        model = str(df["model"].iloc[0])
        horizon = str(df["horizon"].iloc[0])

        mae_p, rmse_p, m90_p = _point_metrics(yt, yp)
        maes, rmses, m90s = _bootstrap_distributions(
            yt, y_pred=yp, n_boot=n_boot, block_len=block_len, seed=seed
        )

        mae_lo, mae_hi = _quantiles(maes)
        rmse_lo, rmse_hi = _quantiles(rmses)
        m90_lo, m90_hi = _quantiles(m90s)

        rows.append(
            {
                "source_file": path.name,
                "pipeline": pipeline,
                "horizon": horizon,
                "model": model,
                "n_test": int(len(yt)),
                "n_bootstrap": n_boot,
                "block_len": block_len,
                "mae_point": mae_p,
                "mae_lower95": mae_lo,
                "mae_upper95": mae_hi,
                "rmse_point": rmse_p,
                "rmse_lower95": rmse_lo,
                "rmse_upper95": rmse_hi,
                "mae_top_decile_point": m90_p,
                "mae_top_decile_lower95": m90_lo,
                "mae_top_decile_upper95": m90_hi,
            }
        )

    if not rows:
        raise SystemExit(f"No prediction CSVs with required columns under {PRED_DIR}.")

    out = pd.DataFrame(rows)
    out_path = OUT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} model–horizon rows)")


if __name__ == "__main__":
    main()
