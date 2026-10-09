"""28-day attention × climatology blend. Robustness check, not headline.

Does not retrain models or overwrite headline prediction files.

Climatology is the training-period mean of corrected PM2.5 by hour-of-day
and Harmattan regime (calendar flag). Applied at the *label* time.

    ŷ = α ŷ_attn + (1 − α) ȳ_clim
    α ∈ {0.5, 0.7, 0.9}  (pre-registered; not tuned)

α = 0 (climatology only) and α = 1 (headline attention) are reported as
references only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import project_path  # noqa: F401

from airpollution import constants as C
from airpollution.features import add_harmattan_flag, add_time_columns
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
ATTN_PRED = ROOT / "reports" / "predictions" / "deep_mh_h672_mh_lstm_mha.csv"
LSTM_PRED = ROOT / "reports" / "predictions" / "deep_mh_h672_mh_lstm.csv"
TABLES = ensure_dir(ROOT / "reports" / "tables")
PRED_DIR = ensure_dir(ROOT / "reports" / "predictions_h28_clim_blend")
ALPHAS = (0.5, 0.7, 0.9)
HEADLINE_ATTN_MAE = 13.56
HEADLINE_LSTM_MAE = 14.92


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _lookup_clim(hour: pd.Series, harmattan: pd.Series, clim: pd.Series, fallback: float) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays([hour.to_numpy(), harmattan.to_numpy()])
    out = clim.reindex(keys).to_numpy(dtype=float)
    if np.isnan(out).any():
        hour_only = clim.groupby(level=0).mean()
        miss = np.isnan(out)
        out[miss] = hour_only.reindex(hour.to_numpy()[miss]).to_numpy(dtype=float)
        out[np.isnan(out)] = fallback
    return out


def main() -> None:
    if not ATTN_PRED.exists():
        raise FileNotFoundError(f"Missing {ATTN_PRED}")

    attn = pd.read_csv(ATTN_PRED)
    attn["target_time"] = pd.to_datetime(attn["target_time"])
    val_start = pd.to_datetime(attn["split_val_start"].iloc[0])

    raw = preprocess(load_raw(), PreprocessOptions(use_local_time=True))
    obs = add_time_columns(add_harmattan_flag(raw, time_col=C.COL_LOCAL_DT), time_col=C.COL_LOCAL_DT)
    obs = obs[[C.COL_LOCAL_DT, C.COL_PM25_CORR, "hour", "harmattan"]].dropna(subset=[C.COL_PM25_CORR])
    obs["harmattan"] = obs["harmattan"].astype(int)

    train_obs = obs[obs[C.COL_LOCAL_DT] < val_start]
    if train_obs.empty:
        raise ValueError("No training-period observations before val_start.")

    clim = train_obs.groupby(["hour", "harmattan"])[C.COL_PM25_CORR].mean()
    global_mean = float(train_obs[C.COL_PM25_CORR].mean())
    n_cells = int(clim.shape[0])
    n_harm_cells = int((clim.index.get_level_values("harmattan") == 1).sum())

    pred = attn.copy()
    pred["hour"] = pred["target_time"].dt.hour
    pred["harmattan"] = pred["harmattan"].fillna(1).astype(int)
    pred["y_clim"] = _lookup_clim(pred["hour"], pred["harmattan"], clim, global_mean)

    y = pred["y_true"].to_numpy(float)
    y_attn = pred["y_pred"].to_numpy(float)
    y_clim = pred["y_clim"].to_numpy(float)

    rows: list[dict] = []
    refs = [
        (0.0, y_clim, "climatology only (reference)"),
        (1.0, y_attn, "headline attention (reference)"),
    ]
    for alpha, yhat, label in refs:
        rows.append(
            {
                "horizon": "28d",
                "model": label,
                "alpha": alpha,
                "tuned": 0,
                "framing": "robustness_reference",
                "n_test": int(len(y)),
                "val_start": str(val_start),
                "n_train_obs": int(len(train_obs)),
                "n_clim_cells": n_cells,
                "n_harmattan_hour_cells": n_harm_cells,
                "clim_global_mean": global_mean,
                "mae": _mae(y, yhat),
                "bias": float((yhat - y).mean()),
                "headline_attn_mae": HEADLINE_ATTN_MAE,
                "headline_joint_lstm_mae": HEADLINE_LSTM_MAE,
                "pred_file": "",
            }
        )

    for alpha in ALPHAS:
        yhat = alpha * y_attn + (1.0 - alpha) * y_clim
        mae = _mae(y, yhat)
        pred_path = PRED_DIR / f"deep_mh_h672_mh_lstm_mha_clim_a{str(alpha).replace('.', '')}.csv"
        out = pred.copy()
        out["y_pred"] = yhat
        out["pipeline"] = "h28_clim_blend"
        out["model"] = f"mh_lstm_mha_clim_a{alpha}"
        out.to_csv(pred_path, index=False)
        rows.append(
            {
                "horizon": "28d",
                "model": f"attention × climatology (α={alpha})",
                "alpha": alpha,
                "tuned": 0,
                "framing": "robustness",
                "n_test": int(len(y)),
                "val_start": str(val_start),
                "n_train_obs": int(len(train_obs)),
                "n_clim_cells": n_cells,
                "n_harmattan_hour_cells": n_harm_cells,
                "clim_global_mean": global_mean,
                "mae": mae,
                "bias": float((yhat - y).mean()),
                "headline_attn_mae": HEADLINE_ATTN_MAE,
                "headline_joint_lstm_mae": HEADLINE_LSTM_MAE,
                "pred_file": pred_path.name,
            }
        )
        print(f"α={alpha:.1f}  MAE={mae:.3f}", flush=True)

    if LSTM_PRED.exists():
        lstm = pd.read_csv(LSTM_PRED)
        rows.append(
            {
                "horizon": "28d",
                "model": "headline joint LSTM (reference)",
                "alpha": np.nan,
                "tuned": 0,
                "framing": "robustness_reference",
                "n_test": int(len(lstm)),
                "val_start": str(val_start),
                "n_train_obs": int(len(train_obs)),
                "n_clim_cells": n_cells,
                "n_harmattan_hour_cells": n_harm_cells,
                "clim_global_mean": global_mean,
                "mae": _mae(lstm["y_true"].to_numpy(float), lstm["y_pred"].to_numpy(float)),
                "bias": float((lstm["y_pred"].to_numpy(float) - lstm["y_true"].to_numpy(float)).mean()),
                "headline_attn_mae": HEADLINE_ATTN_MAE,
                "headline_joint_lstm_mae": HEADLINE_LSTM_MAE,
                "pred_file": "",
            }
        )

    table = pd.DataFrame(rows)
    path = TABLES / "results_h28_climatology_blend.csv"
    table.to_csv(path, index=False)
    print(table.to_string(index=False), flush=True)
    print(f"Wrote {path}", flush=True)
    print(
        f"Climatology: {len(train_obs)} train hours before {val_start}, "
        f"{n_cells} hour×regime cells ({n_harm_cells} Harmattan hours), "
        f"global mean {global_mean:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
