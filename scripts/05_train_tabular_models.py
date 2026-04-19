from __future__ import annotations

import json
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

    def _safe_metrics(yhat: np.ndarray) -> dict | None:
        mask = np.isfinite(y_true) & np.isfinite(yhat)
        if mask.sum() < 10:
            return None
        return regression_metrics(y_true[mask], yhat[mask])

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

    for hz in HORIZONS:
        path = FEATURE_DIR / f"tabular_{hz}.parquet"
        mb = path.stat().st_size / (1024 * 1024)
        print(f"[{hz}] Loading {path.name} (~{mb:.1f} MB)...", flush=True)
        ds = pd.read_parquet(path)
        print(f"[{hz}] Loaded {ds.shape[0]:,} rows × {ds.shape[1]} columns.", flush=True)

        # Ensure target_time is parsed
        ds["target_time"] = pd.to_datetime(ds["target_time"])

        train, val, test, meta = time_split_by_target_time(ds, cfg=SplitConfig(val_days=14, test_days=28))
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
        models = {
            "ridge": Ridge(alpha=1.0, random_state=42),
            "rf": RandomForestRegressor(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=3,
            ),
            # Liblinear: epsilon-insensitive + L2 penalty requires dual=True in recent scikit-learn
            # (dual=False raises ValueError). Prefer SVR(kernel="linear") if you need primal formulation.
            "linear_svr": LinearSVR(
                random_state=42,
                C=1.0,
                epsilon=0.2,
                max_iter=20_000,
                tol=1e-3,
                dual=True,
            ),
        }
        if _HAS_XGBOOST:
            models["xgboost"] = XGBRegressor(
                n_estimators=2000,
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

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["horizon", "mae", "rmse"]).reset_index(drop=True)
    metrics_df.to_csv(TABLES_DIR / "results_tabular_model_metrics.csv", index=False)

    pd.DataFrame(split_rows).to_csv(TABLES_DIR / "results_time_splits_by_horizon.csv", index=False)

    print("Wrote model metrics to reports/tables/results_tabular_model_metrics.csv")


if __name__ == "__main__":
    main()

