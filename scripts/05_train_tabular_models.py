from __future__ import annotations

import json
import os
from pathlib import Path

import project_path  # noqa: F401 — side effect: repo src/ on sys.path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import LinearSVR

try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore
    _HAS_XGBOOST = False

from airpollution import constants as C
from airpollution.eval import SplitConfig, regression_metrics, time_split_by_target_time, top_decile_mask
from airpollution.utils import ensure_dir


ROOT = project_path.ROOT
FEATURE_DIR = ROOT / "data" / "features"
TABLES_DIR = ensure_dir(ROOT / "reports" / "tables")


HORIZONS = ["h24", "h168", "h336", "h672"]
FAST_MODE = os.environ.get("AIRP_FAST", "").strip().lower() in ("1", "true", "yes", "y")

# Make slow model sizes configurable for quick iteration / Overleaf tables.
RF_TREES = int(os.environ.get("AIRP_RF_TREES", "600"))
XGB_TREES = int(os.environ.get("AIRP_XGB_TREES", "2000"))
RF_MAX_DEPTH = os.environ.get("AIRP_RF_MAX_DEPTH", "").strip()
RF_MAX_DEPTH = None if RF_MAX_DEPTH == "" else int(RF_MAX_DEPTH)
RF_N_JOBS = int(os.environ.get("AIRP_RF_N_JOBS", "-1"))
_rf_mf = os.environ.get("AIRP_RF_MAX_FEATURES", "sqrt").strip()
try:
    RF_MAX_FEATURES = float(_rf_mf)  # allow e.g. "0.3"
except Exception:
    RF_MAX_FEATURES = _rf_mf  # "sqrt" / "log2" / "None"
if isinstance(RF_MAX_FEATURES, str) and RF_MAX_FEATURES.lower() == "none":
    RF_MAX_FEATURES = None


def _feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df["y"].to_numpy(dtype=float)
    X = df.drop(columns=["y"]).copy()
    # Defensive: replace infinities that can arise from bad parses/ops.
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def _build_preprocessor(X: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    # Use numeric columns only (after one-hot, most are numeric). Keep booleans as numeric.
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    # Drop timestamp columns if they sneak in as numeric (they shouldn't).
    drop_cols = [c for c in ["target_time"] if c in numeric_cols]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=True)),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
    )
    return pre, numeric_cols


def _eval_model(name: str, model, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    X_train, y_train = _feature_target_split(train)
    X_val, y_val = _feature_target_split(val)
    X_test, y_test = _feature_target_split(test)

    pre, numeric_cols = _build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    pred_test = pipe.predict(X_test)
    m = regression_metrics(y_test, pred_test)

    # Regime-slice metrics (Harmattan vs non-Harmattan) when label is available.
    # This mirrors the deep-model evaluation and supports "dry vs wet" comparisons
    # without requiring extra external meteorology.
    if "harmattan" in test.columns:
        harm = test["harmattan"].to_numpy(dtype=float)
        pre = harm < 0.5
        ha = harm >= 0.5
        if pre.sum() > 50 and ha.sum() > 50:
            m_pre = regression_metrics(y_test[pre], pred_test[pre])
            m_ha = regression_metrics(y_test[ha], pred_test[ha])
            for k, v in m_pre.items():
                m[f"{k}_pre_harmattan"] = v
            for k, v in m_ha.items():
                m[f"{k}_harmattan"] = v

    # Peak-slice metrics (top decile of y_true in test)
    mask = top_decile_mask(y_test)
    if mask.any():
        m_peak = regression_metrics(y_test[mask], pred_test[mask])
        m["mae_top_decile"] = m_peak["mae"]
        m["mse_top_decile"] = m_peak["mse"]
        m["rmse_top_decile"] = m_peak["rmse"]
    else:
        m["mae_top_decile"] = float("nan")
        m["mse_top_decile"] = float("nan")
        m["rmse_top_decile"] = float("nan")

    m["model"] = name
    return m


def _eval_baselines(df_test: pd.DataFrame) -> list[dict]:
    """Operational baselines using available lag features."""
    y_true = df_test["y"].to_numpy(dtype=float)
    harm = df_test["harmattan"].to_numpy(dtype=float) if "harmattan" in df_test.columns else None

    def _safe_metrics(yhat: np.ndarray) -> dict | None:
        mask = np.isfinite(y_true) & np.isfinite(yhat)
        if mask.sum() < 10:
            return None
        m = regression_metrics(y_true[mask], yhat[mask])
        if harm is not None:
            h = harm[mask]
            pre = h < 0.5
            ha = h >= 0.5
            if pre.sum() > 50 and ha.sum() > 50:
                m_pre = regression_metrics(y_true[mask][pre], yhat[mask][pre])
                m_ha = regression_metrics(y_true[mask][ha], yhat[mask][ha])
                for k, v in m_pre.items():
                    m[f"{k}_pre_harmattan"] = v
                for k, v in m_ha.items():
                    m[f"{k}_harmattan"] = v
        return m

    out = []

    # Naive / seasonal-naïve baselines (Hyndman-style simple benchmarks)
    # Baseline 1: naive level — predict using PM2.5 at forecast origin (y_t)
    if C.COL_PM25_CORR in df_test.columns:
        yhat = df_test[C.COL_PM25_CORR].to_numpy(dtype=float)
        m = _safe_metrics(yhat)
        if m:
            m["model"] = "naive_level"
            out.append(m)

    # Seasonal naïve, period 24 h: ŷ = y_{t-24}
    lag24 = f"lag_{C.COL_PM25_CORR}_24h"
    if lag24 in df_test.columns:
        yhat = df_test[lag24].to_numpy(dtype=float)
        m = _safe_metrics(yhat)
        if m:
            m["model"] = "seasonal_naive_24h"
            out.append(m)

    # Seasonal naïve, period 168 h (weekly): ŷ = y_{t-168}
    lag168 = f"lag_{C.COL_PM25_CORR}_168h"
    if lag168 in df_test.columns:
        yhat = df_test[lag168].to_numpy(dtype=float)
        m = _safe_metrics(yhat)
        if m:
            m["model"] = "seasonal_naive_168h"
            out.append(m)

    # Seasonal naïve, period 336 h (14d): ŷ = y_{t-336}
    lag336 = f"lag_{C.COL_PM25_CORR}_336h"
    if lag336 in df_test.columns:
        yhat = df_test[lag336].to_numpy(dtype=float)
        m = _safe_metrics(yhat)
        if m:
            m["model"] = "seasonal_naive_336h"
            out.append(m)

    # Seasonal naïve, period 672 h (28d): ŷ = y_{t-672}
    lag672 = f"lag_{C.COL_PM25_CORR}_672h"
    if lag672 in df_test.columns:
        yhat = df_test[lag672].to_numpy(dtype=float)
        m = _safe_metrics(yhat)
        if m:
            m["model"] = "seasonal_naive_672h"
            out.append(m)

    return out


def _ar_lag_fourier_feature_names(df: pd.DataFrame) -> list[str]:
    """
    Autoregressive + seasonality: PM$_{2.5}$ lags only, plus Fourier terms and
    hour / DOW / month (and site one-hots), as a clear benchmark next to full Ridge.
    """
    pm = C.COL_PM25_CORR
    out: list[str] = []
    for c in df.columns:
        if c in ("y", C.COL_LOCATION_NAME, C.COL_LOCAL_DT, "target_time"):
            continue
        if c.startswith("loc_"):
            out.append(c)
        elif c in ("hour", "dayofweek", "month", "harmattan"):
            out.append(c)
        elif c.startswith("sin_") or c.startswith("cos_"):
            out.append(c)
        elif c.startswith(f"lag_{pm}_"):
            out.append(c)
    return out


def _eval_ridge_ar_lag_fourier(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict | None:
    cols = _ar_lag_fourier_feature_names(train)
    cols = [c for c in cols if c in train.columns and c in test.columns]
    if not cols:
        return None
    X_train = train[cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[cols].replace([np.inf, -np.inf], np.nan)
    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)

    pre, _ = _build_preprocessor(X_train)
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0, random_state=42))])
    pipe.fit(X_train, y_train)
    pred_test = pipe.predict(X_test).astype(float)
    m = regression_metrics(y_test, pred_test)

    if "harmattan" in test.columns:
        harm = test["harmattan"].to_numpy(dtype=float)
        pre = harm < 0.5
        ha = harm >= 0.5
        if pre.sum() > 50 and ha.sum() > 50:
            m_pre = regression_metrics(y_test[pre], pred_test[pre])
            m_ha = regression_metrics(y_test[ha], pred_test[ha])
            for k, v in m_pre.items():
                m[f"{k}_pre_harmattan"] = v
            for k, v in m_ha.items():
                m[f"{k}_harmattan"] = v

    mask = top_decile_mask(y_test)
    if mask.any():
        m_peak = regression_metrics(y_test[mask], pred_test[mask])
        m["mae_top_decile"] = m_peak["mae"]
        m["mse_top_decile"] = m_peak["mse"]
        m["rmse_top_decile"] = m_peak["rmse"]
    else:
        m["mae_top_decile"] = float("nan")
        m["mse_top_decile"] = float("nan")
        m["rmse_top_decile"] = float("nan")

    m["model"] = "ridge_ar_lag_fourier"
    return m


def main() -> None:
    metrics_rows: list[dict] = []
    split_rows: list[dict] = []

    # Allow reproducible alternative splits without editing code.
    # Default matches the paper (val=14d, test=28d).
    cfg = SplitConfig(
        val_days=int(os.environ.get("AIRP_VAL_DAYS", "14")),
        test_days=int(os.environ.get("AIRP_TEST_DAYS", "28")),
    )

    for hz in HORIZONS:
        path_parquet = FEATURE_DIR / f"tabular_{hz}.parquet"
        path_csv = FEATURE_DIR / f"tabular_{hz}.csv.gz"

        ds = None
        if path_parquet.exists():
            try:
                mb = path_parquet.stat().st_size / (1024 * 1024)
                print(f"[{hz}] Loading {path_parquet.name} (~{mb:.1f} MB)...", flush=True)
                ds = pd.read_parquet(path_parquet)
            except Exception as e:
                print(f"[{hz}] Parquet load failed ({e!s}); falling back to {path_csv.name}...", flush=True)
                ds = None

        if ds is None:
            if not path_csv.exists():
                raise FileNotFoundError(
                    f"Missing feature dataset for {hz}. Expected {path_parquet.name} or {path_csv.name}. "
                    "Run scripts/04_make_tabular_dataset.py first."
                )
            mb = path_csv.stat().st_size / (1024 * 1024)
            print(f"[{hz}] Loading {path_csv.name} (~{mb:.1f} MB)...", flush=True)
            ds = pd.read_csv(path_csv)

        print(f"[{hz}] Loaded {ds.shape[0]:,} rows × {ds.shape[1]} columns.", flush=True)

        # Ensure target_time is parsed
        ds["target_time"] = pd.to_datetime(ds["target_time"])

        train, val, test, meta = time_split_by_target_time(ds, cfg=cfg)
        meta_row = {"horizon": hz, **{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in meta.items()}}
        split_rows.append(meta_row)

        # Baselines on test (naive + seasonal naïve)
        for m in _eval_baselines(test):
            metrics_rows.append({"horizon": hz, **m})

        # Regularised linear AR + Fourier (explicit benchmark requested for papers)
        m_ar = _eval_ridge_ar_lag_fourier(train, test)
        if m_ar:
            metrics_rows.append({"horizon": hz, **m_ar})

        # Models
        models = {"ridge": Ridge(alpha=1.0, random_state=42)}
        if not FAST_MODE:
            models["rf"] = RandomForestRegressor(
                n_estimators=RF_TREES,
                random_state=42,
                n_jobs=RF_N_JOBS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=3,
                max_features=RF_MAX_FEATURES,
            )
            # Liblinear: epsilon-insensitive + L2 penalty requires dual=True in recent scikit-learn
            # (dual=False raises ValueError). Prefer SVR(kernel="linear") if you need primal formulation.
            models["linear_svr"] = LinearSVR(
                random_state=42,
                C=1.0,
                epsilon=0.2,
                max_iter=20_000,
                tol=1e-3,
                dual=True,
            )
        if _HAS_XGBOOST:
            if not FAST_MODE:
                models["xgboost"] = XGBRegressor(
                    n_estimators=XGB_TREES,
                    learning_rate=0.03,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                )
        else:
            # Portable boosted-tree baseline when XGBoost can't load (e.g., missing libomp on macOS).
            if not FAST_MODE:
                models["hgbt"] = HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=500,
                    random_state=42,
                )

        # Drop non-numeric / leakage-prone columns from train/val/test before feeding to pipeline
        drop_cols = [c for c in [ds.columns[0], "target_time"] if c in ds.columns]  # defensive
        # Keep Location Name as it may be useful for reporting, but pipeline drops non-numeric anyway.
        for name, model in models.items():
            print(f"[{hz}] Training {name}...", flush=True)
            m = _eval_model(name, model, train, val, test)
            metrics_rows.append({"horizon": hz, **m})
            print(f"[{hz}] Done {name}: MAE={m['mae']:.2f}", flush=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["horizon", "mae", "rmse"]).reset_index(drop=True)
    metrics_df.to_csv(TABLES_DIR / "results_tabular_model_metrics.csv", index=False)

    pd.DataFrame(split_rows).to_csv(TABLES_DIR / "results_time_splits_by_horizon.csv", index=False)

    print("Wrote model metrics to reports/tables/results_tabular_model_metrics.csv")


if __name__ == "__main__":
    main()

