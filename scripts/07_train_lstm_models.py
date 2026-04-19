"""
Train LSTM and LSTM + multi-head attention on pre-built sequence .npz files.

Uses Keras 3 with the first available backend (in order): TensorFlow, JAX, PyTorch.
Set e.g. `pip install tensorflow` or `pip install jax jaxlib` if the default is missing.

Run after: scripts/06_build_sequence_dataset.py
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
    Keras 3 reads ``KERAS_BACKEND`` on first import. Try TF → JAX → Torch so
    Python 3.14 / environments without TensorFlow can still run if JAX or Torch is installed.
    """
    order = ("tensorflow", "jax", "torch")
    errors: list[str] = []
    for backend in order:
        os.environ["KERAS_BACKEND"] = backend
        # Allow switching backend on retry (only matters if keras was imported before).
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
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    hint = (
        "Install ONE backend supported by your Python: "
        "pip install tensorflow  OR  pip install jax jaxlib  OR  pip install torch. "
        f"Your Python is {py}. TensorFlow / jaxlib / PyTorch often have no wheels for 3.14+ yet — "
        "use Python 3.11 or 3.12 in a separate venv for script 07 (see README)."
    )
    raise ImportError("No Keras backend available. Tried: " + "; ".join(errors) + ". " + hint)


def build_lstm(keras, input_shape: tuple[int, int, int], lstm_units: int = 64):
    layers = keras.layers
    inp = keras.Input(shape=input_shape[1:])
    x = layers.LSTM(lstm_units)(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def build_lstm_attention(keras, input_shape: tuple[int, int, int], lstm_units: int = 64, num_heads: int = 4):
    layers = keras.layers
    inp = keras.Input(shape=input_shape[1:])
    seq = layers.LSTM(lstm_units, return_sequences=True)(inp)
    seq = layers.LayerNormalization()(seq)
    key_dim = max(8, lstm_units // num_heads)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        seq, seq, seq
    )
    attn_pool = layers.GlobalAveragePooling1D()(attn)
    lstm_vec = layers.LSTM(lstm_units)(inp)
    gap_inp = layers.GlobalAveragePooling1D()(inp)
    cat = layers.Concatenate()([lstm_vec, attn_pool, gap_inp])
    x = layers.Dense(48, activation="relu")(cat)
    out = layers.Dense(1)(x)
    return keras.Model(inp, out)


def train_one(
    keras,
    model,
    X_tr,
    y_tr,
    X_va,
    y_va,
    epochs: int = 80,
    batch_size: int = 128,
):
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
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
    rows: list[dict] = []

    py_mm = sys.version_info[:2]
    if py_mm >= (3, 14):
        msg = (
            "Script 07 (LSTM) needs TensorFlow, jaxlib, or PyTorch. None of these publish "
            "pip wheels for Python 3.14 yet, so `pip install tensorflow` will fail.\n\n"
            "Fix: create a Python 3.12 virtualenv, install the project + one backend, then run this script:\n"
            "  python3.12 -m venv .venv312 && source .venv312/bin/activate\n"
            "  pip install -r requirements.txt && pip install -e . && pip install tensorflow\n"
            "  # tensorflow pins NumPy<2 on many platforms; install it last so deps resolve.\n"
            "  python scripts/07_train_lstm_models.py\n"
        )
        stub = {
            "status": "skipped",
            "reason": "no_deep_learning_wheels_for_python_3_14",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "hint": msg.strip(),
        }
        (TABLES_DIR / "results_deep_models_status.json").write_text(json.dumps(stub, indent=2))
        print(msg, flush=True)
        return

    try:
        keras, backend_name = _try_keras_backend()
        print(f"Keras backend: {backend_name}", flush=True)
    except Exception as e:
        stub = {
            "status": "skipped",
            "reason": str(e),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "hint": "Install one backend: pip install tensorflow  OR  pip install jax jaxlib  OR  pip install torch. "
            "If pip still finds nothing, your Python version may be too new — use Python 3.12.",
        }
        (TABLES_DIR / "results_deep_models_status.json").write_text(json.dumps(stub, indent=2))
        print(stub["reason"])
        return

    cfg = SplitConfig(val_days=14, test_days=28)

    for npz_path in sorted(SEQ_DIR.glob("sequences_h*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y = data["y"].astype(np.float32)
        tt = pd.to_datetime(data["target_time"])
        harm = data["harmattan_y"] if "harmattan_y" in data.files else None
        if harm is not None:
            harm = np.asarray(harm).astype(np.float32)

        m_tr, m_va, m_te, split_meta = time_split_masks(tt, cfg=cfg)

        X_tr, y_tr = X[m_tr], y[m_tr]
        X_va, y_va = X[m_va], y[m_va]
        X_te, y_te = X[m_te], y[m_te]

        if len(X_tr) < 500 or len(X_va) < 100:
            rows.append({"file": npz_path.name, "model": "—", "mae": np.nan, "note": "insufficient samples after split"})
            continue

        X_tr, X_va, X_te = _scale_fit_train(X_tr, X_va, X_te)
        input_shape = X_tr.shape

        tag = npz_path.stem
        for name, builder in [
            ("lstm", lambda: build_lstm(keras, input_shape)),
            ("lstm_mha", lambda: build_lstm_attention(keras, input_shape)),
        ]:
            model = builder()
            model = train_one(keras, model, X_tr, y_tr, X_va, y_va)
            pred = model.predict(X_te, verbose=0).reshape(-1)
            m = regression_metrics(y_te, pred)
            meta_row = {
                "keras_backend": backend_name,
                "file": npz_path.name,
                "model": name,
                "horizon_hours": int(data["horizon_hours"]),
                "seq_len": int(data["seq_len"]),
                "tmax": str(split_meta["tmax"]),
                "val_start": str(split_meta["val_start"]),
                "test_start": str(split_meta["test_start"]),
                "n_train_split": split_meta["n_train"],
                "n_val_split": split_meta["n_val"],
                "n_test_split": split_meta["n_test"],
            }
            m.update(meta_row)

            md = top_decile_mask(y_te)
            if md.any():
                mp = regression_metrics(y_te[md], pred[md])
                m["mae_top_decile"] = mp["mae"]
                m["rmse_top_decile"] = mp["rmse"]
            else:
                m["mae_top_decile"] = np.nan
                m["rmse_top_decile"] = np.nan

            if harm is not None:
                h_te = harm[m_te]
                pre = h_te < 0.5
                ha = h_te >= 0.5
                if pre.sum() > 50 and ha.sum() > 50:
                    m["mae_pre_harmattan"] = float(np.mean(np.abs(y_te[pre] - pred[pre])))
                    m["mae_harmattan"] = float(np.mean(np.abs(y_te[ha] - pred[ha])))

            rows.append(m)

        print(f"Trained {tag}: lstm + lstm_mha, test n={split_meta['n_test']}")

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "results_deep_model_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
