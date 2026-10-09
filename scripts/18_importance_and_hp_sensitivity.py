"""24-h grouped permutation importance and a small tabular hyperparameter check.

Does not overwrite headline model CSVs. Uses the validation block for both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

import project_path  # noqa: F401

from airpollution import constants as C
from airpollution.eval import SplitConfig, time_split_by_target_time
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.tabular import TabularFeatureSpec, make_supervised_tabular
from airpollution.utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
TABLES = ensure_dir(ROOT / "reports" / "tables")
SEED = 42
N_REPEATS = 8


def _feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df["y"].to_numpy(dtype=float)
    X = df.select_dtypes(include=["number", "bool"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    drop = [c for c in ("y", "target_time") if c in X.columns]
    X = X.drop(columns=drop, errors="ignore")
    return X, y


def _group_of(col: str) -> str:
    pm25 = C.COL_PM25_CORR
    if pm25 in col and (col.startswith("lag_") or col.startswith("roll_")):
        return "Lagged / rolling PM2.5"
    if any(k in col for k in (C.COL_PM1, C.COL_PM10, C.COL_PARTICLE_03)):
        return "Particulate indicators"
    if any(k in col for k in (C.COL_TEMP_CORR, C.COL_HUMIDITY_CORR)):
        return "Meteorological conditions"
    if C.COL_CO2_CORR in col:
        return "Combustion proxy (CO2)"
    if col in {"hour", "dayofweek", "month", "harmattan"} or col.startswith("sin_") or col.startswith("cos_"):
        return "Temporal / Fourier"
    if col.startswith("loc_"):
        return "Spatial (site)"
    return "Other environmental variables"


def _grouped_permutation_mae(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], n_repeats: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = mean_absolute_error(y, model.predict(X))
    rows = []
    X_np = {c: X[c].to_numpy() for c in X.columns}
    for gname, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            for c in cols:
                Xp[c] = X_np[c][idx]
            mae = mean_absolute_error(y, model.predict(Xp))
            deltas.append(mae - base)
        arr = np.asarray(deltas, dtype=float)
        rows.append(
            {
                "group": gname,
                "n_columns": len(cols),
                "delta_mae_mean": float(arr.mean()),
                "delta_mae_std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "delta_mae_q05": float(np.quantile(arr, 0.05)),
                "delta_mae_q95": float(np.quantile(arr, 0.95)),
                "baseline_val_mae": base,
            }
        )
    out = pd.DataFrame(rows).sort_values("delta_mae_mean", ascending=False)
    tot = out["delta_mae_mean"].clip(lower=0).sum()
    out["share_of_positive_delta_pct"] = np.where(tot > 0, 100.0 * out["delta_mae_mean"].clip(lower=0) / tot, np.nan)
    return out


def main() -> None:
    print("Building 24-h tabular features...", flush=True)
    df = preprocess(load_raw(), PreprocessOptions(use_local_time=True))
    ds = make_supervised_tabular(df, horizon_hours=24, spec=TabularFeatureSpec(), time_col=C.COL_LOCAL_DT)
    ds["target_time"] = pd.to_datetime(ds["target_time"])
    train, val, test, meta = time_split_by_target_time(ds, cfg=SplitConfig(val_days=14, test_days=28))
    print(f"split n_train={len(train)} n_val={len(val)} n_test={len(test)} test_start={meta['test_start']}", flush=True)

    X_tr, y_tr = _feature_matrix(train)
    X_va, y_va = _feature_matrix(val)
    imputer = SimpleImputer(strategy="median")
    X_tr_i = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_va_i = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns, index=X_va.index)

    groups: dict[str, list[str]] = {}
    for c in X_tr_i.columns:
        groups.setdefault(_group_of(c), []).append(c)
    print("groups:", {k: len(v) for k, v in groups.items()}, flush=True)

    print("Fitting RF (600 trees) on train...", flush=True)
    rf = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr_i, y_tr)

    print("Grouped permutation on validation...", flush=True)
    imp = _grouped_permutation_mae(rf, X_va_i, y_va, groups, N_REPEATS)
    imp_path = TABLES / "results_grouped_permutation_importance_h24.csv"
    imp.to_csv(imp_path, index=False)
    print(imp.to_string(index=False), flush=True)
    print(f"Wrote {imp_path}", flush=True)

    # Impurity importances summed by group (point estimate only, plus std across trees of group sum)
    fi = np.asarray([est.feature_importances_ for est in rf.estimators_])  # (trees, p)
    names = list(X_tr_i.columns)
    mdi_rows = []
    for gname, cols in groups.items():
        idx = [names.index(c) for c in cols]
        per_tree = fi[:, idx].sum(axis=1)
        mdi_rows.append(
            {
                "group": gname,
                "n_columns": len(cols),
                "mdi_mean": float(per_tree.mean()),
                "mdi_std_trees": float(per_tree.std(ddof=1)),
            }
        )
    mdi = pd.DataFrame(mdi_rows)
    mdi["mdi_pct"] = 100.0 * mdi["mdi_mean"] / mdi["mdi_mean"].sum()
    mdi = mdi.sort_values("mdi_pct", ascending=False)
    mdi_path = TABLES / "results_grouped_mdi_importance_h24.csv"
    mdi.to_csv(mdi_path, index=False)
    print(mdi.to_string(index=False), flush=True)
    print(f"Wrote {mdi_path}", flush=True)

    print("Hyperparameter sensitivity on 24-h validation MAE...", flush=True)
    hp_rows = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=alpha))])
        # Ridge AR+Fourier-like: all numeric (includes lags/fourier); this is a sensitivity not the restricted AR model
        pipe.fit(X_tr, y_tr)
        mae_va = mean_absolute_error(y_va, pipe.predict(X_va))
        hp_rows.append({"model": "Ridge (all numeric)", "setting": f"alpha={alpha}", "val_mae": mae_va})

    for n_est, leaf in ((200, 3), (600, 3), (600, 1), (600, 5)):
        m = RandomForestRegressor(
            n_estimators=n_est,
            min_samples_leaf=leaf,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )
        m.fit(X_tr_i, y_tr)
        mae_va = mean_absolute_error(y_va, m.predict(X_va_i))
        hp_rows.append({"model": "Random Forest", "setting": f"n_estimators={n_est}, min_samples_leaf={leaf}", "val_mae": mae_va})

    hp = pd.DataFrame(hp_rows)
    hp_path = TABLES / "results_hyperparameter_sensitivity_h24.csv"
    hp.to_csv(hp_path, index=False)
    print(hp.to_string(index=False), flush=True)
    print(f"Wrote {hp_path}", flush=True)


if __name__ == "__main__":
    main()
