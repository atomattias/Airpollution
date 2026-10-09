"""Pre-registered 24-h direct LSTM sensitivity: lookback × width.

Does not overwrite headline sequence files or deep_h24_lstm predictions.
Headline paper setting remains L=168, 64 units.

  LOOKBACKS = (168, 336, 672)
  WIDTHS = (32, 64, 128)
  learning rate 1e-3, batch 128, seed 42, lstm only
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

from airpollution import constants as C
from airpollution.eval import SplitConfig, time_split_masks
from airpollution.io import load_raw
from airpollution.preprocess import PreprocessOptions, preprocess
from airpollution.sequences import SequenceSpec, build_sequence_arrays
from airpollution.utils import ensure_dir

ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = ensure_dir(ROOT / "data" / "sequences_h24_sens")
TABLES = ensure_dir(ROOT / "reports" / "tables")
PRED_DIR = ensure_dir(ROOT / "reports" / "predictions_h24_sens")
LOOKBACKS = (168, 336, 672)
WIDTHS = (32, 64, 128)
HORIZON = 24
SEED = 42
PAPER_L = 168
PAPER_UNITS = 64


def _load_07():
    path = ROOT / "scripts" / "07_train_lstm_models.py"
    spec = importlib.util.spec_from_file_location("train_lstm_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_sequences() -> None:
    existing_l168 = ROOT / "data" / "sequences" / "sequences_h24_L168.npz"
    df = None
    for sl in LOOKBACKS:
        out = SEQ_DIR / f"sequences_h24_L{sl}.npz"
        if out.exists():
            print(f"exists {out}", flush=True)
            continue
        if sl == 168 and existing_l168.exists():
            out.write_bytes(existing_l168.read_bytes())
            print(f"copied {existing_l168} -> {out}", flush=True)
            continue
        if df is None:
            df_raw = load_raw()
            df = preprocess(df_raw, PreprocessOptions(use_local_time=True))
        spec = SequenceSpec(
            horizon_hours=HORIZON,
            seq_len=sl,
            time_col=C.COL_LOCAL_DT,
            add_location_onehot=True,
            add_time_features=True,
            add_fourier_daily=True,
            add_fourier_weekly=True,
        )
        X, y, target_time, meta, harm_y = build_sequence_arrays(df, spec)
        np.savez_compressed(
            out,
            X=X,
            y=y,
            harmattan_y=harm_y,
            target_time=target_time.astype("datetime64[ns]").to_numpy(),
            location=meta[C.COL_LOCATION_NAME].to_numpy(),
            horizon_hours=HORIZON,
            seq_len=sl,
        )
        print(f"Wrote {out} X={X.shape}", flush=True)


def main() -> None:
    os.environ.setdefault("AIRP_MODELS", "lstm")
    os.environ.setdefault("AIRP_SEED", str(SEED))
    _ensure_sequences()
    train07 = _load_07()
    keras, backend = train07._try_keras_backend()
    print(f"Keras backend: {backend}", flush=True)
    np.random.seed(SEED)
    cfg = SplitConfig(val_days=14, test_days=28)
    rows = []

    for sl in LOOKBACKS:
        npz_path = SEQ_DIR / f"sequences_h24_L{sl}.npz"
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y = data["y"].astype(np.float32)
        tt = pd.to_datetime(data["target_time"])
        loc = data["location"] if "location" in data.files else None
        m_tr, m_va, m_te, split_meta = time_split_masks(tt, cfg=cfg)
        X_tr, y_tr = X[m_tr], y[m_tr]
        X_va, y_va = X[m_va], y[m_va]
        X_te, y_te = X[m_te], y[m_te]
        X_tr, X_va, X_te = train07._scale_fit_train(X_tr, X_va, X_te)

        for units in WIDTHS:
            print(f"train L={sl} units={units} n_train={len(y_tr)} n_val={len(y_va)} n_test={len(y_te)}", flush=True)
            model = train07.build_lstm(keras, X_tr.shape, lstm_units=units)
            model = train07.train_one(keras, model, X_tr, y_tr, X_va, y_va)
            pred_va = model.predict(X_va, verbose=0).reshape(-1)
            pred_te = model.predict(X_te, verbose=0).reshape(-1)
            val_mae = float(mean_absolute_error(y_va, pred_va))
            test_mae = float(mean_absolute_error(y_te, pred_te))
            tag = f"lstm_L{sl}_u{units}"
            pred_path = PRED_DIR / f"deep_h24_{tag}.csv"
            train07.write_test_predictions_csv(
                pred_path,
                target_time=tt[m_te],
                y_true=y_te,
                y_pred=pred_te.astype(float),
                pipeline="deep_h24_sens",
                model=tag,
                horizon="h24",
                location=loc[m_te] if loc is not None else None,
                split_meta=split_meta,
                keras_backend=backend,
                seq_len=sl,
                seed=SEED,
            )
            rows.append(
                {
                    "horizon": "24h",
                    "model": "LSTM (per-horizon direct)",
                    "seq_len": sl,
                    "lstm_units": units,
                    "is_paper_setting": int(sl == PAPER_L and units == PAPER_UNITS),
                    "n_train": int(split_meta["n_train"]),
                    "n_val": int(split_meta["n_val"]),
                    "n_test": int(split_meta["n_test"]),
                    "val_mae": val_mae,
                    "test_mae": test_mae,
                    "keras_backend": backend,
                    "pred_file": str(pred_path.name),
                }
            )
            print(f"  val_mae={val_mae:.3f} test_mae={test_mae:.3f}", flush=True)

    out = pd.DataFrame(rows)
    path = TABLES / "results_lstm_h24_sensitivity.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False), flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    if sys.version_info[:2] >= (3, 14):
        sys.stderr.write(
            "Use Python 3.12 with JAX/TF, e.g.\n"
            "  /Users/mattiasgebrie/Desktop/Dev/Air\\ Pollution/.venv312/bin/python "
            "scripts/19_lstm_h24_sensitivity.py\n"
        )
        sys.exit(1)
    main()
