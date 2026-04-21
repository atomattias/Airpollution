"""
Create paper-style diagnostic scatter plots:
- Regression (y_true vs y_pred)
- Residuals (residual vs y_pred)
- Actual vs Predicted (time-ordered scatter)

This script re-fits tabular models on the standard leakage-safe split and
produces figures under reports/figures/.

Run after: scripts/04_make_tabular_dataset.py
"""

from __future__ import annotations

import os
from pathlib import Path

import project_path  # noqa: F401 — side effect: repo src/ on sys.path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR

from airpollution.eval import SplitConfig, time_split_by_target_time

try:
    from xgboost import XGBRegressor  # type: ignore

    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore
    _HAS_XGBOOST = False

ROOT = project_path.ROOT
FEATURE_DIR = ROOT / "data" / "features"
FIGS_DIR = ROOT / "reports" / "figures"


def _feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df["y"].to_numpy(dtype=float)
    X = df.drop(columns=["y"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def _build_preprocessor(X: pd.DataFrame):
    # Mirrors scripts/05_train_tabular_models.py
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
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


def _fit_predict(model, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_tr, y_tr = _feature_target_split(train)
    X_te, y_te = _feature_target_split(test)

    pre, _ = _build_preprocessor(X_tr)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr, y_tr)
    yhat = pipe.predict(X_te).astype(float)

    # Align target_time if available for the "Actual vs Predicted" panel.
    tt = pd.to_datetime(test["target_time"]).to_numpy() if "target_time" in test.columns else np.arange(len(y_te))
    return y_te.astype(float), yhat, tt


def _plot_triptych(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tt,
    title: str,
    out_path: Path,
) -> None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    tt = np.asarray(tt)[mask]

    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))

    # 1) Regression: actual vs predicted + 45-degree line
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=10, alpha=0.45)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], color="firebrick", lw=1.2)
    ax.set_title("Regression Plot")
    ax.set_xlabel("Actual PM$_{2.5}$")
    ax.set_ylabel("Predicted PM$_{2.5}$")
    ax.grid(alpha=0.2)

    # 2) Residuals vs predicted
    ax = axes[1]
    ax.scatter(y_pred, resid, s=10, alpha=0.45)
    ax.axhline(0.0, color="firebrick", lw=1.2)
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted PM$_{2.5}$")
    ax.set_ylabel("Residuals")
    ax.grid(alpha=0.2)

    # 3) Actual vs predicted over target_time (unordered scatter)
    ax = axes[2]
    # Sort by time if datetime-like
    try:
        order = np.argsort(tt)
    except Exception:
        order = np.arange(len(tt))
    ax.scatter(tt[order], y_true[order], s=10, alpha=0.45, label="Actual PM$_{2.5}$")
    ax.scatter(tt[order], y_pred[order], s=10, alpha=0.45, label="Predicted PM$_{2.5}$")
    ax.set_title("Actual vs Predicted PM$_{2.5}$")
    ax.set_xlabel("Target time")
    ax.set_ylabel("PM$_{2.5}$")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    hz = os.environ.get("AIRP_SCATTER_HZ", "h24").strip()
    path_csv = FEATURE_DIR / f"tabular_{hz}.csv.gz"

    # Prefer CSV to avoid pyarrow/parquet syscalls blocked in sandboxed environments.
    if path_csv.exists():
        ds = pd.read_csv(path_csv)
    else:
        raise FileNotFoundError(f"Missing feature dataset for {hz}. Expected {path_csv.name}.")

    ds["target_time"] = pd.to_datetime(ds["target_time"])
    cfg = SplitConfig(
        val_days=int(os.environ.get("AIRP_VAL_DAYS", "14")),
        test_days=int(os.environ.get("AIRP_TEST_DAYS", "28")),
    )
    train, _val, test, meta = time_split_by_target_time(ds, cfg=cfg)

    # NOTE: Some environments can crash with low-level floating point exceptions
    # in specific estimators (observed for LinearSVR on some macOS builds).
    # Default to robust tree models; you can explicitly request linear_svr if desired.
    requested = os.environ.get("AIRP_SCATTER_MODELS", "rf,xgboost").strip()
    requested_set = {m.strip() for m in requested.split(",") if m.strip()}

    models: dict[str, object] = {}
    if "rf" in requested_set:
        models["rf"] = RandomForestRegressor(
            n_estimators=int(os.environ.get("AIRP_RF_TREES", "200")),
            random_state=42,
            n_jobs=int(os.environ.get("AIRP_RF_N_JOBS", "1")),
            min_samples_leaf=3,
            max_features=os.environ.get("AIRP_RF_MAX_FEATURES", "sqrt"),
        )
    if "linear_svr" in requested_set:
        models["linear_svr"] = LinearSVR(
            random_state=42,
            C=1.0,
            epsilon=0.2,
            max_iter=20_000,
            tol=1e-3,
            dual=True,
        )
    if "xgboost" in requested_set and _HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            n_estimators=int(os.environ.get("AIRP_XGB_TREES", "500")),
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=int(os.environ.get("AIRP_XGB_N_JOBS", "1")),
        )

    for name, model in models.items():
        try:
            y_true, y_pred, tt = _fit_predict(model, train, test)
            title = f"{name} ({hz}) — test window from {meta['test_start']:%Y-%m-%d} to {meta['tmax']:%Y-%m-%d}"
            out = FIGS_DIR / f"results_scatter_{name}_{hz}.png"
            _plot_triptych(y_true=y_true, y_pred=y_pred, tt=tt, title=title, out_path=out)
            print(f"Wrote {out}")
        except Exception as e:
            # Keep the script usable even when one model fails in a given environment.
            print(f"Skipping {name} due to error: {e!s}")


if __name__ == "__main__":
    main()

