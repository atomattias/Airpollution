"""Export per-row test-set predictions for uncertainty quantification and paired tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def safe_model_filename_tag(model: str) -> str:
    return (
        str(model)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
    )


def write_test_predictions_csv(
    path: Path,
    *,
    target_time: pd.Series | np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pipeline: str,
    model: str,
    horizon: str,
    harmattan: np.ndarray | None = None,
    location: np.ndarray | pd.Series | None = None,
    split_meta: dict | None = None,
    keras_backend: str = "",
    seq_len: int | float | None = None,
    seed: int | float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(y_true))
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred must have the same length")
    tt = pd.to_datetime(pd.Series(np.asarray(target_time).ravel()).iloc[:n])

    if harmattan is None:
        harm = np.full(n, np.nan, dtype=float)
    else:
        harm = np.asarray(harmattan, dtype=float).ravel()[:n]

    if location is None:
        loc = np.full(n, np.nan, dtype=object)
    else:
        loc = np.asarray(location).ravel()[:n]

    sm = split_meta or {}
    val_s = sm.get("val_start", "")
    test_s = sm.get("test_start", "")
    tmax_s = sm.get("tmax", "")
    if hasattr(val_s, "isoformat"):
        val_s = val_s.isoformat()
    if hasattr(test_s, "isoformat"):
        test_s = test_s.isoformat()
    if hasattr(tmax_s, "isoformat"):
        tmax_s = tmax_s.isoformat()

    df = pd.DataFrame(
        {
            "target_time": tt,
            "y_true": np.asarray(y_true, dtype=float).ravel()[:n],
            "y_pred": np.asarray(y_pred, dtype=float).ravel()[:n],
            "harmattan": harm,
            "location": loc,
            "pipeline": pipeline,
            "model": model,
            "horizon": horizon,
            "split_val_start": val_s,
            "split_test_start": test_s,
            "split_tmax": tmax_s,
            "keras_backend": keras_backend or "",
            "seq_len": float(seq_len) if seq_len is not None else np.nan,
            "seed": float(seed) if seed is not None else np.nan,
        }
    )
    df.to_csv(path, index=False)
