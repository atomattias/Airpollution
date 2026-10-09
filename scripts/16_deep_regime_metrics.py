"""
Assemble 84-day LSTM regime MAE rows for Table 9.

Reads isolated 07 outputs (AIRP_TABLES_DIR) and writes:
  reports/tables/results_deep_regime_metrics_wide.csv

Does not retrain. Run after:
  AIRP_TEST_DAYS=84 AIRP_PRED_DIR=... AIRP_TABLES_DIR=... python scripts/07_train_lstm_models.py
  AIRP_TEST_DAYS=84 AIRP_PRED_DIR=... AIRP_TABLES_DIR=... python scripts/07_train_lstm_multihorizon.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import project_path  # noqa: F401

ROOT = project_path.ROOT
DIRECT_DIRS = [
    Path(p)
    for p in os.environ.get(
        "AIRP_DIRECT_TABLES_DIRS",
        f"{ROOT / 'reports' / 'tables_regime84'}:{ROOT / 'reports' / 'tables_regime84_mha'}",
    ).split(":")
    if p.strip()
]
JOINT_DIR = Path(os.environ.get("AIRP_JOINT_TABLES_DIR", ROOT / "reports" / "tables_regime84_L336"))
OUT_DIR = ROOT / "reports" / "tables"
OUT_PATH = OUT_DIR / "results_deep_regime_metrics_wide.csv"

MODEL_LABELS = {
    "lstm": "LSTM (per-horizon direct)",
    "lstm_mha": "LSTM + attention (per-horizon)",
    "mh_lstm": "Joint multi-horizon LSTM (mh_lstm)",
    "mh_lstm_mha": "Joint multi-horizon LSTM + attention (mh_lstm_mha)",
}

KEEP = {"lstm", "lstm_mha", "mh_lstm", "mh_lstm_mha"}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    frames = []
    for d in DIRECT_DIRS:
        frames.append(_load(d / "results_deep_model_metrics.csv"))
    frames.append(_load(JOINT_DIR / "results_deep_model_metrics_multihorizon.csv"))
    frames = [d for d in frames if not d.empty]
    if not frames:
        raise FileNotFoundError("No deep metrics found in DIRECT_DIRS or JOINT_DIR")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["model"].isin(KEEP)].copy()
    hz = df["horizon_hours"].astype(int)
    df["horizon"] = hz.map({24: "h24", 168: "h168", 336: "h336", 672: "h672"})
    df["label"] = df["model"].map(MODEL_LABELS)
    wide = pd.DataFrame(
        {
            "horizon": df["horizon"],
            "model": df["model"],
            "label": df["label"],
            "mae_overall": df["mae"],
            "mae_pre_harmattan": df["mae_pre_harmattan"],
            "mae_harmattan": df["mae_harmattan"],
            "n_pre_harmattan": df.get("n_pre_harmattan", pd.Series(dtype=float)),
            "n_harmattan": df.get("n_harmattan", pd.Series(dtype=float)),
            "n_test": df["n_test_split"],
            "test_start": df["test_start"],
        }
    )
    wide = wide.sort_values(["horizon", "model"]).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
