from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class SplitConfig:
    val_days: int = 14
    test_days: int = 28


def time_split_by_target_time(
    df: pd.DataFrame,
    target_time_col: str = "target_time",
    cfg: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Blocked split using the *label timestamp* to prevent leakage."""
    c = cfg or SplitConfig()
    tmax = pd.to_datetime(df[target_time_col]).max()
    test_start = tmax - pd.Timedelta(days=c.test_days) + pd.Timedelta(hours=0)
    val_start = test_start - pd.Timedelta(days=c.val_days)

    tt = pd.to_datetime(df[target_time_col])
    train = df[tt < val_start].copy()
    val = df[(tt >= val_start) & (tt < test_start)].copy()
    test = df[tt >= test_start].copy()

    meta = {
        "tmax": tmax,
        "val_start": val_start,
        "test_start": test_start,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    return train, val, test, meta


def time_split_masks(
    target_time: pd.Series | np.ndarray,
    cfg: SplitConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Boolean masks aligned with sequence rows, using label time (leakage-safe)."""
    c = cfg or SplitConfig()
    tt = pd.to_datetime(pd.Series(target_time))
    tmax = tt.max()
    test_start = tmax - pd.Timedelta(days=c.test_days)
    val_start = test_start - pd.Timedelta(days=c.val_days)

    m_tr = (tt < val_start).to_numpy()
    m_va = ((tt >= val_start) & (tt < test_start)).to_numpy()
    m_te = (tt >= test_start).to_numpy()
    meta = {
        "tmax": tmax,
        "val_start": val_start,
        "test_start": test_start,
        "n_train": int(m_tr.sum()),
        "n_val": int(m_va.sum()),
        "n_test": int(m_te.sum()),
    }
    return m_tr, m_va, m_te, meta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def top_decile_mask(y_true: np.ndarray) -> np.ndarray:
    thr = np.nanquantile(y_true, 0.90)
    return y_true >= thr

