"""Merge tabular and deep-learning metrics into one table for Results / LaTeX."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"

TABULAR_CSV = TABLES / "results_tabular_model_metrics.csv"
DEEP_CSV = TABLES / "results_deep_model_metrics.csv"
STATUS_JSON = TABLES / "results_deep_models_status.json"
OUT_LONG = TABLES / "results_merged_long.csv"
OUT_WIDE = TABLES / "results_merged_wide_mae.csv"


HOURS_TO_HORIZON = {24: "h24", 168: "h168", 336: "h336", 672: "h672"}
HORIZON_TO_HOURS = {v: k for k, v in HOURS_TO_HORIZON.items()}


def _load_tabular() -> pd.DataFrame:
    if not TABULAR_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TABULAR_CSV)
    df["pipeline"] = "tabular"
    df["horizon_hours"] = df["horizon"].map(HORIZON_TO_HOURS)
    # Canonical column order helpers
    keep = [
        "pipeline",
        "horizon",
        "horizon_hours",
        "model",
        "mae",
        "rmse",
        "r2",
    ]
    extra = [c for c in df.columns if c not in keep and c not in ("horizon",)]
    cols = [c for c in keep if c in df.columns] + [c for c in extra if c in df.columns]
    return df[[c for c in cols if c in df.columns]]


def _load_deep() -> pd.DataFrame:
    if not DEEP_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DEEP_CSV)
    if df.empty:
        return pd.DataFrame()
    df["pipeline"] = "deep"
    df["horizon"] = df["horizon_hours"].map(HOURS_TO_HORIZON)
    df["model"] = df["model"].astype(str)
    rename_map = {
        "n_train_split": "n_train",
        "n_val_split": "n_val",
        "n_test_split": "n_test",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    base = [
        "pipeline",
        "horizon",
        "horizon_hours",
        "model",
        "mae",
        "rmse",
        "r2",
    ]
    rest = [c for c in df.columns if c not in base]
    cols = [c for c in base if c in df.columns] + [c for c in rest if c in df.columns]
    return df[[c for c in cols if c in df.columns]]


def _wide_mae(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=model (+pipeline), columns=horizon, values=MAE."""
    if long_df.empty or "mae" not in long_df.columns:
        return pd.DataFrame()
    sub = long_df[["pipeline", "horizon", "model", "mae"]].dropna(subset=["mae"])
    sub["key"] = sub["pipeline"] + ":" + sub["model"].astype(str)
    wide = sub.pivot_table(index="key", columns="horizon", values="mae", aggfunc="first")
    wide = wide.reset_index().rename(columns={"key": "pipeline_model"})
    # Order horizon columns
    order = [h for h in ["h24", "h168", "h336", "h672"] if h in wide.columns]
    return wide[["pipeline_model"] + order]


def main() -> None:
    tab = _load_tabular()
    deep = _load_deep()

    parts = []
    if not tab.empty:
        parts.append(tab)
    if not deep.empty:
        parts.append(deep)

    if not parts:
        print(f"No inputs found. Expected at least {TABULAR_CSV}")
        print("Run scripts/05_train_tabular_models.py first.")
        return

    long_df = pd.concat(parts, ignore_index=True)
    # Stable sort: horizon order then pipeline then mae
    hz_order = {h: i for i, h in enumerate(["h24", "h168", "h336", "h672"])}
    long_df["_hz"] = long_df["horizon"].map(lambda x: hz_order.get(str(x), 99))
    long_df = long_df.sort_values(["_hz", "pipeline", "mae", "model"]).drop(columns=["_hz"])

    long_df.to_csv(OUT_LONG, index=False)
    print(f"Wrote {OUT_LONG} ({len(long_df)} rows)")

    wide = _wide_mae(long_df)
    if not wide.empty:
        wide.to_csv(OUT_WIDE, index=False)
        print(f"Wrote {OUT_WIDE}")

    if STATUS_JSON.exists():
        print(f"Note: {STATUS_JSON.name} present (deep training may have been skipped).")


if __name__ == "__main__":
    main()
