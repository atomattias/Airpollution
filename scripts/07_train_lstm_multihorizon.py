"""
Train a single multi-horizon model (joint outputs) for 24h/7d/14d/28d.

This is optional and complements `07_train_lstm_models.py`.
It expects the dataset produced by `06_build_sequence_dataset_multihorizon.py`.

Outputs:
  - reports/tables/results_deep_model_metrics_multihorizon.csv

Design goal:
  Write one row PER (model, horizon) so downstream LaTeX/merge logic can reuse existing tables.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import project_path  # noqa: F401 — repo src/ on sys.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from airpollution.eval import SplitConfig, regression_metrics, time_split_masks, top_decile_mask
from airpollution.utils import ensure_dir

ROOT = project_path.ROOT
SEQ_DIR = ROOT / "data" / "sequences"
TABLES_DIR = ensure_dir(ROOT / "reports" / "tables")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _scale_fit_train(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tr, T, F = X_tr.shape
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(n_tr, T, F)
    X_va_s = scaler.transform(X_va.reshape(-1, F)).reshape(X_va.shape[0], T, F)
    X_te_s = scaler.transform(X_te.reshape(-1, F)).reshape(X_te.shape[0], T, F)
    return X_tr_s, X_va_s, X_te_s


def _try_keras_backend() -> tuple[object, str]:
    """
    Keras 3 reads ``KERAS_BACKEND`` on first import. Try TF → JAX → Torch.
    """
    order = ("tensorflow", "jax", "torch")
    errors: list[str] = []
    for backend in order:
        os.environ["KERAS_BACKEND"] = backend
        for name in list(sys.modules):
            if name == "keras" or name.startswith("keras."):
                del sys.modules[name]
        try:
            if backend == "tensorflow":
                import tensorflow  # noqa: F401
            elif backend == "jax":
                import jax  # noqa: F401
            else:
                import torch  # noqa: F401
            keras = importlib.import_module("keras")
            return keras, backend
        except Exception as e:
            errors.append(f"{backend}: {e!s}")
            continue
    raise ImportError("No Keras backend available. Tried: " + "; ".join(errors))


def build_mh_lstm(keras, input_shape: tuple[int, int, int], out_dim: int, lstm_units: int = 64):
    layers = keras.layers
    inp = keras.Input(shape=input_shape[1:])
    x = layers.LSTM(lstm_units)(inp)
    x = layers.Dense(48, activation="relu")(x)
    out = layers.Dense(out_dim)(x)
    return keras.Model(inp, out)


def build_mh_lstm_attention(
    keras,
    input_shape: tuple[int, int, int],
    out_dim: int,
    lstm_units: int = 64,
    num_heads: int = 4,
):
    layers = keras.layers
    inp = keras.Input(shape=input_shape[1:])
    seq = layers.LSTM(lstm_units, return_sequences=True)(inp)
    seq = layers.LayerNormalization()(seq)
    key_dim = max(8, lstm_units // num_heads)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(seq, seq, seq)
    attn_pool = layers.GlobalAveragePooling1D()(attn)
    lstm_vec = layers.LSTM(lstm_units)(inp)
    gap_inp = layers.GlobalAveragePooling1D()(inp)
    cat = layers.Concatenate()([lstm_vec, attn_pool, gap_inp])
    x = layers.Dense(64, activation="relu")(cat)
    out = layers.Dense(out_dim)(x)
    return keras.Model(inp, out)


def train_one(
    keras,
    model,
    X_tr,
    y_tr,
    X_va,
    y_va,
    epochs: int = 100,
    batch_size: int = 128,
):
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=14, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
    ]
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=0,
    )
    return model


def main() -> None:
    # Allow alternative split lengths (useful for regime-spanning evaluation).
    cfg = SplitConfig(
        val_days=int(os.environ.get("AIRP_VAL_DAYS", "14")),
        test_days=int(os.environ.get("AIRP_TEST_DAYS", "28")),
    )

    # Find the multi-horizon npz written by script 06 (multi-horizon).
    cand = sorted(SEQ_DIR.glob("sequences_mh_*.npz"))
    if not cand:
        raise FileNotFoundError("No sequences_mh_*.npz found. Run scripts/06_build_sequence_dataset_multihorizon.py first.")
    npz_path = cand[-1]

    try:
        keras, backend_name = _try_keras_backend()
        print(f"Keras backend: {backend_name}", flush=True)
    except Exception as e:
        stub = {
            "status": "skipped",
            "reason": str(e),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "hint": "Install one backend: pip install tensorflow  OR  pip install jax jaxlib  OR  pip install torch. "
            "If pip still finds nothing, use Python 3.11/3.12.",
        }
        (TABLES_DIR / "results_deep_models_status_multihorizon.json").write_text(json.dumps(stub, indent=2))
        print(stub["reason"])
        return

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.float32)  # (N, K)
    tt = pd.to_datetime(data["target_time"])
    horizons = data["horizons_hours"].astype(int).tolist()
    harm = data["harmattan_y"] if "harmattan_y" in data.files else None
    if harm is not None:
        harm = np.asarray(harm).astype(np.float32)

    m_tr, m_va, m_te, split_meta = time_split_masks(tt, cfg=cfg)

    X_tr, y_tr = X[m_tr], y[m_tr]
    X_va, y_va = X[m_va], y[m_va]
    X_te, y_te = X[m_te], y[m_te]

    if len(X_tr) < 500 or len(X_va) < 100:
        raise RuntimeError("Insufficient samples after split; adjust AIRP_VAL_DAYS/AIRP_TEST_DAYS.")

    X_tr, X_va, X_te = _scale_fit_train(X_tr, X_va, X_te)
    input_shape = X_tr.shape
    out_dim = y_tr.shape[1]

    rows: list[dict] = []
    tag = npz_path.stem

    for name, builder in [
        ("mh_lstm", lambda: build_mh_lstm(keras, input_shape, out_dim)),
        ("mh_lstm_mha", lambda: build_mh_lstm_attention(keras, input_shape, out_dim)),
    ]:
        model = builder()
        model = train_one(keras, model, X_tr, y_tr, X_va, y_va)
        pred = model.predict(X_te, verbose=0)  # (N_test, K)

        # Emit per-horizon rows for compatibility with existing merge tables.
        for k, h in enumerate(horizons):
            yk = y_te[:, k].reshape(-1)
            pk = pred[:, k].reshape(-1)
            m = regression_metrics(yk, pk)
            meta_row = {
                "keras_backend": backend_name,
                "file": npz_path.name,
                "model": name,
                "horizon_hours": int(h),
                "seq_len": int(data["seq_len"]),
                "tmax": str(split_meta["tmax"]),
                "val_start": str(split_meta["val_start"]),
                "test_start": str(split_meta["test_start"]),
                "n_train_split": split_meta["n_train"],
                "n_val_split": split_meta["n_val"],
                "n_test_split": split_meta["n_test"],
                "note": f"multi-horizon:{tag}",
            }
            m.update(meta_row)

            md = top_decile_mask(yk)
            if md.any():
                mp = regression_metrics(yk[md], pk[md])
                m["mae_top_decile"] = mp["mae"]
                m["rmse_top_decile"] = mp["rmse"]
            else:
                m["mae_top_decile"] = np.nan
                m["rmse_top_decile"] = np.nan

            # Regime metrics can be computed only on the split key label time (max-horizon label time).
            # We therefore omit per-horizon regime slicing here (leave NaN for stable columns).
            for kk in (
                "mae_pre_harmattan",
                "rmse_pre_harmattan",
                "r2_pre_harmattan",
                "mae_harmattan",
                "rmse_harmattan",
                "r2_harmattan",
            ):
                m.setdefault(kk, np.nan)

            rows.append(m)

        print(f"Trained {tag}: {name}, test n={split_meta['n_test']}", flush=True)

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "results_deep_model_metrics_multihorizon.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

