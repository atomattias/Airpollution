"""
Export tabular-model test metrics stratified by Harmattan regime (pre-Harmattan vs Harmattan).

Uses the same feature pipelines and model definitions as scripts/05_train_tabular_models.py.

**Important:** The default 28-day test window in the main paper often lies entirely inside one
seasonal regime (e.g. all Harmattan), so ``mae_pre_harmattan`` / ``mae_harmattan`` cannot both
be estimated. This script therefore defaults to a **longer test window** (84 days) so the test
set spans both regimes, matching the regime-spanning naïve benchmarks discussed in the paper.

Override with ``AIRP_REGIME_TEST_DAYS`` (and ``AIRP_VAL_DAYS`` if needed).

Writes: reports/tables/results_tabular_regime_metrics_wide.csv

Run: python scripts/11_regime_metrics_export.py
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

import project_path  # noqa: F401

ROOT = project_path.ROOT
FEATURE_DIR = ROOT / "data" / "features"
TABLES_DIR = ROOT / "reports" / "tables"

HORIZONS = ["h24", "h168", "h336", "h672"]


def _load_train_tab():
    p = Path(__file__).resolve().parent / "05_train_tabular_models.py"
    spec = importlib.util.spec_from_file_location("train_tabular_models", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    t = _load_train_tab()
    # Longer test window so label times include both pre-Harmattan and Harmattan rows.
    regime_test_days = int(os.environ.get("AIRP_REGIME_TEST_DAYS", "84"))
    cfg = t.SplitConfig(
        val_days=int(os.environ.get("AIRP_VAL_DAYS", "14")),
        test_days=regime_test_days,
    )
    print(f"Regime metrics split: val_days={cfg.val_days}, test_days={cfg.test_days}", flush=True)

    wide_rows: list[dict] = []

    for hz in HORIZONS:
        path_parquet = FEATURE_DIR / f"tabular_{hz}.parquet"
        path_csv = FEATURE_DIR / f"tabular_{hz}.csv.gz"
        if path_parquet.exists():
            ds = pd.read_parquet(path_parquet)
        elif path_csv.exists():
            ds = pd.read_csv(path_csv)
        else:
            raise FileNotFoundError(f"Missing tabular dataset for {hz}")

        ds["target_time"] = pd.to_datetime(ds["target_time"])
        train, val, test, _meta = t.time_split_by_target_time(ds, cfg=cfg)

        models = {"ridge": t.Ridge(alpha=1.0, random_state=42)}
        if not t.FAST_MODE:
            models["rf"] = t.RandomForestRegressor(
                n_estimators=t.RF_TREES,
                random_state=42,
                n_jobs=t.RF_N_JOBS,
                max_depth=t.RF_MAX_DEPTH,
                min_samples_leaf=3,
                max_features=t.RF_MAX_FEATURES,
            )
            models["linear_svr"] = t.LinearSVR(
                random_state=42,
                C=1.0,
                epsilon=0.2,
                max_iter=20_000,
                tol=1e-3,
                dual=True,
            )
        if t._HAS_XGBOOST and not t.FAST_MODE:  # noqa: SLF001
            models["xgboost"] = t.XGBRegressor(
                n_estimators=t.XGB_TREES,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        elif not t.FAST_MODE:
            models["hgbt"] = t.HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=8,
                max_iter=500,
                random_state=42,
            )

        for name, model in models.items():
            m = t._eval_model(name, model, train, val, test)  # noqa: SLF001
            wide_rows.append(
                {
                    "horizon": hz,
                    "model": name,
                    "mae_overall": m.get("mae"),
                    "rmse_overall": m.get("rmse"),
                    "r2_overall": m.get("r2"),
                    "mae_pre_harmattan": m.get("mae_pre_harmattan"),
                    "rmse_pre_harmattan": m.get("rmse_pre_harmattan"),
                    "r2_pre_harmattan": m.get("r2_pre_harmattan"),
                    "mae_harmattan": m.get("mae_harmattan"),
                    "rmse_harmattan": m.get("rmse_harmattan"),
                    "r2_harmattan": m.get("r2_harmattan"),
                }
            )

    wide = pd.DataFrame(wide_rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path_wide = TABLES_DIR / "results_tabular_regime_metrics_wide.csv"
    wide.to_csv(path_wide, index=False)
    print(f"Wrote {path_wide}")


if __name__ == "__main__":
    main()
