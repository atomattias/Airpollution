"""Diagnostic 24-h residual LSTM. Does not overwrite headline models.

Same architecture as the paper direct LSTM: L=168, 64 units, Adam 1e-3, seed 42.

Target:  Δy = y_{t+24} − y_t
Reconstruct: ŷ_{t+24} = y_t + Δŷ

y_t is the last observed corrected PM2.5 in the lookback window (X[:, -1, 0]
before scaling). Predictions and MAE are reported on the reconstructed
concentration scale.

A same-run direct LSTM is trained as a backend-matched control. Headline
numbers remain: direct LSTM 23.19 (TensorFlow) and Ridge 11.96.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import project_path  # noqa: F401

from airpollution.eval import SplitConfig, time_split_masks
from airpollution.utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
SEQ_SRC = ROOT / "data" / "sequences" / "sequences_h24_L168.npz"
TABLES = ensure_dir(ROOT / "reports" / "tables")
PRED_DIR = ensure_dir(ROOT / "reports" / "predictions_h24_residual")
SEQ_LEN = 168
UNITS = 64
HORIZON = 24
SEED = 42
HEADLINE_DIRECT_MAE = 23.19
HEADLINE_RIDGE_MAE = 11.96


def _load_07():
    path = ROOT / "scripts" / "07_train_lstm_models.py"
    spec = importlib.util.spec_from_file_location("train_lstm_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def main() -> None:
    os.environ.setdefault("AIRP_SEED", str(SEED))
    if not SEQ_SRC.exists():
        raise FileNotFoundError(f"Missing {SEQ_SRC}. Run scripts/06_build_sequence_dataset.py first.")

    train07 = _load_07()
    keras, backend = train07._try_keras_backend()
    print(f"Keras backend: {backend}", flush=True)
    np.random.seed(SEED)
    try:
        import tensorflow as tf

        tf.random.set_seed(SEED)
    except Exception:
        pass

    data = np.load(SEQ_SRC, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.float32)
    y0 = X[:, -1, 0].astype(np.float32)
    if not np.isfinite(y0).all():
        raise ValueError("Origin PM2.5 (last lookback step, channel 0) has non-finite values.")
    tt = pd.to_datetime(data["target_time"])
    loc = data["location"] if "location" in data.files else None
    harm = data["harmattan_y"] if "harmattan_y" in data.files else None

    cfg = SplitConfig(val_days=14, test_days=28)
    m_tr, m_va, m_te, split_meta = time_split_masks(tt, cfg=cfg)
    X_tr, X_va, X_te = X[m_tr], X[m_va], X[m_te]
    y_tr, y_va, y_te = y[m_tr], y[m_va], y[m_te]
    y0_tr, y0_va, y0_te = y0[m_tr], y0[m_va], y0[m_te]
    dy_tr, dy_va, dy_te = y_tr - y0_tr, y_va - y0_va, y_te - y0_te

    persist_val = _mae(y_va, y0_va)
    persist_test = _mae(y_te, y0_te)
    print(
        f"n_train={len(y_tr)} n_val={len(y_va)} n_test={len(y_te)} "
        f"persist_val={persist_val:.3f} persist_test={persist_test:.3f}",
        flush=True,
    )

    X_tr_s, X_va_s, X_te_s = train07._scale_fit_train(X_tr, X_va, X_te)

    rows: list[dict] = []

    print("train residual LSTM (target Δy)", flush=True)
    model_res = train07.build_lstm(keras, X_tr_s.shape, lstm_units=UNITS)
    model_res = train07.train_one(keras, model_res, X_tr_s, dy_tr, X_va_s, dy_va)
    dhat_va = model_res.predict(X_va_s, verbose=0).reshape(-1)
    dhat_te = model_res.predict(X_te_s, verbose=0).reshape(-1)
    pred_res_va = y0_va + dhat_va
    pred_res_te = y0_te + dhat_te
    res_val = _mae(y_va, pred_res_va)
    res_test = _mae(y_te, pred_res_te)
    print(f"  residual val_mae={res_val:.3f} test_mae={res_test:.3f}", flush=True)

    pred_path = PRED_DIR / "deep_h24_lstm_residual.csv"
    train07.write_test_predictions_csv(
        pred_path,
        target_time=tt[m_te],
        y_true=y_te,
        y_pred=pred_res_te.astype(float),
        pipeline="deep_h24_residual",
        model="lstm_residual",
        horizon="h24",
        harmattan=harm[m_te] if harm is not None else None,
        location=loc[m_te] if loc is not None else None,
        split_meta=split_meta,
        keras_backend=backend,
        seq_len=SEQ_LEN,
        seed=SEED,
    )

    print("train same-run direct LSTM control (target y)", flush=True)
    model_dir = train07.build_lstm(keras, X_tr_s.shape, lstm_units=UNITS)
    model_dir = train07.train_one(keras, model_dir, X_tr_s, y_tr, X_va_s, y_va)
    pred_dir_va = model_dir.predict(X_va_s, verbose=0).reshape(-1)
    pred_dir_te = model_dir.predict(X_te_s, verbose=0).reshape(-1)
    dir_val = _mae(y_va, pred_dir_va)
    dir_test = _mae(y_te, pred_dir_te)
    print(f"  direct    val_mae={dir_val:.3f} test_mae={dir_test:.3f}", flush=True)

    train07.write_test_predictions_csv(
        PRED_DIR / "deep_h24_lstm_direct_control.csv",
        target_time=tt[m_te],
        y_true=y_te,
        y_pred=pred_dir_te.astype(float),
        pipeline="deep_h24_residual",
        model="lstm_direct_control",
        horizon="h24",
        harmattan=harm[m_te] if harm is not None else None,
        location=loc[m_te] if loc is not None else None,
        split_meta=split_meta,
        keras_backend=backend,
        seq_len=SEQ_LEN,
        seed=SEED,
    )

    rows.append(
        {
            "horizon": "24h",
            "model": "LSTM residual (Δy + y_t)",
            "framing": "diagnostic",
            "seq_len": SEQ_LEN,
            "lstm_units": UNITS,
            "n_train": int(split_meta["n_train"]),
            "n_val": int(split_meta["n_val"]),
            "n_test": int(split_meta["n_test"]),
            "val_mae": res_val,
            "test_mae": res_test,
            "delta_val_mae": _mae(dy_va, dhat_va),
            "delta_test_mae": _mae(dy_te, dhat_te),
            "bias_test": float((pred_res_te - y_te).mean()),
            "keras_backend": backend,
            "pred_file": pred_path.name,
            "headline_direct_lstm_mae": HEADLINE_DIRECT_MAE,
            "headline_ridge_mae": HEADLINE_RIDGE_MAE,
        }
    )
    rows.append(
        {
            "horizon": "24h",
            "model": "LSTM direct (same-run control)",
            "framing": "diagnostic",
            "seq_len": SEQ_LEN,
            "lstm_units": UNITS,
            "n_train": int(split_meta["n_train"]),
            "n_val": int(split_meta["n_val"]),
            "n_test": int(split_meta["n_test"]),
            "val_mae": dir_val,
            "test_mae": dir_test,
            "delta_val_mae": np.nan,
            "delta_test_mae": np.nan,
            "bias_test": float((pred_dir_te - y_te).mean()),
            "keras_backend": backend,
            "pred_file": "deep_h24_lstm_direct_control.csv",
            "headline_direct_lstm_mae": HEADLINE_DIRECT_MAE,
            "headline_ridge_mae": HEADLINE_RIDGE_MAE,
        }
    )
    rows.append(
        {
            "horizon": "24h",
            "model": "Persistence (y_t)",
            "framing": "diagnostic",
            "seq_len": SEQ_LEN,
            "lstm_units": 0,
            "n_train": int(split_meta["n_train"]),
            "n_val": int(split_meta["n_val"]),
            "n_test": int(split_meta["n_test"]),
            "val_mae": persist_val,
            "test_mae": persist_test,
            "delta_val_mae": persist_val,
            "delta_test_mae": persist_test,
            "bias_test": float((y0_te - y_te).mean()),
            "keras_backend": "",
            "pred_file": "",
            "headline_direct_lstm_mae": HEADLINE_DIRECT_MAE,
            "headline_ridge_mae": HEADLINE_RIDGE_MAE,
        }
    )

    out = pd.DataFrame(rows)
    path = TABLES / "results_lstm_h24_residual.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False), flush=True)
    print(f"Wrote {path}", flush=True)
    print(
        f"Compare: residual test {res_test:.2f} vs same-run direct {dir_test:.2f} "
        f"vs headline direct {HEADLINE_DIRECT_MAE:.2f} vs Ridge {HEADLINE_RIDGE_MAE:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    if sys.version_info[:2] >= (3, 14):
        sys.stderr.write(
            "Use Python 3.12 with JAX/TF, e.g.\n"
            "  /Users/mattiasgebrie/Desktop/Dev/Air\\ Pollution/.venv312/bin/python "
            "scripts/21_lstm_h24_residual.py\n"
        )
        sys.exit(1)
    main()
