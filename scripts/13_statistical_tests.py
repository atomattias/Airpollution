"""
Paired block-bootstrap tests on mean absolute error differences.

For aligned test rows, defines per-row loss difference
  d_i = |y_i - ŷ_A,i| - |y_i - ŷ_B,i|
and bootstraps mean(d) with the same block scheme as script 12.

Pre-specified comparisons (edit list below if model keys change).
Outputs: reports/tables/results_model_significance_tests.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import project_path  # noqa: F401

from airpollution.predictions_export import safe_model_filename_tag

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "reports" / "predictions"
OUT_CSV = ROOT / "reports" / "tables" / "results_model_significance_tests.csv"


def _pred_path(pipeline: str, model: str, horizon: str) -> Path:
    tag = safe_model_filename_tag(model)
    if pipeline == "tabular":
        return PRED_DIR / f"tabular_{horizon}_{tag}.csv"
    if pipeline == "deep" and str(model).startswith("mh_"):
        return PRED_DIR / f"deep_mh_{horizon}_{tag}.csv"
    return PRED_DIR / f"deep_{horizon}_{tag}.csv"


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    block_len = max(1, min(int(block_len), n))
    if n <= block_len:
        return rng.choice(n, size=n, replace=True)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx: list[int] = []
    for s in starts:
        idx.extend(range(int(s), int(s) + block_len))
    return np.asarray(idx[:n], dtype=int)


def _align_two(
    path_a: Path,
    path_b: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path_a.is_file() or not path_b.is_file():
        raise FileNotFoundError(f"Missing {path_a} or {path_b}")
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    keys = ["target_time"]
    if "location" in a.columns and "location" in b.columns:
        if a["location"].notna().any() and b["location"].notna().any():
            keys.append("location")
    m = a.merge(b, on=keys, suffixes=("_a", "_b"), how="inner")
    if m.empty:
        raise ValueError(f"No aligned rows between {path_a.name} and {path_b.name}")
    y = m["y_true_a"].to_numpy(dtype=float)
    if not np.allclose(m["y_true_a"].to_numpy(float), m["y_true_b"].to_numpy(float), rtol=1e-5, atol=1e-4, equal_nan=True):
        raise ValueError("y_true mismatch after merge — check prediction exports.")
    ya = m["y_pred_a"].to_numpy(dtype=float)
    yb = m["y_pred_b"].to_numpy(dtype=float)
    return y, ya, yb


def main() -> None:
    if not PRED_DIR.is_dir():
        raise SystemExit(f"Missing {PRED_DIR}")

    n_boot = int(os.environ.get("AIRP_BOOTSTRAP_B", "1000"))
    block_len = int(os.environ.get("AIRP_BOOTSTRAP_BLOCK_LEN", "168"))
    seed = int(os.environ.get("AIRP_BOOTSTRAP_SEED", "42"))
    rng = np.random.default_rng(seed)

    # Pre-specified comparisons (pipeline, model_a, pipeline_b, model_b), every horizon.
    specs: list[tuple[str, str, str, str, str]] = [
        ("mh_lstm vs per-horizon lstm", "deep", "mh_lstm", "deep", "lstm"),
        ("mh_lstm_mha vs mh_lstm", "deep", "mh_lstm_mha", "deep", "mh_lstm"),
        ("mh_lstm_mha vs Ridge AR+Fourier", "deep", "mh_lstm_mha", "tabular", "ridge_ar_lag_fourier"),
    ]
    horizons_all = ["h24", "h168", "h336", "h672"]
    comparisons = [(lab, pa, ma, pb, mb, hz) for lab, pa, ma, pb, mb in specs for hz in horizons_all]

    rows: list[dict] = []
    for label, pa, ma, pb, mb, hz in comparisons:
        path_a = _pred_path(pa, ma, hz)
        path_b = _pred_path(pb, mb, hz)
        try:
            y, ya, yb = _align_two(path_a, path_b)
        except Exception as e:
            rows.append(
                {
                    "comparison": label,
                    "horizon": hz,
                    "model_a": f"{pa}:{ma}",
                    "model_b": f"{pb}:{mb}",
                    "n_aligned": 0,
                    "error": str(e),
                }
            )
            continue

        mask = np.isfinite(y) & np.isfinite(ya) & np.isfinite(yb)
        y, ya, yb = y[mask], ya[mask], yb[mask]
        n = len(y)
        d = np.abs(y - ya) - np.abs(y - yb)
        delta_point = float(np.mean(d))
        mae_a = float(np.mean(np.abs(y - ya)))
        mae_b = float(np.mean(np.abs(y - yb)))

        boots = np.full(n_boot, np.nan)
        for b in range(n_boot):
            idx = _block_bootstrap_indices(n, block_len, rng)
            boots[b] = float(np.mean(d[idx]))

        boots = boots[np.isfinite(boots)]
        if boots.size == 0:
            lo = hi = float("nan")
            ci_excludes_zero = False
        else:
            lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            # Interval for mean(|e_a|-|e_b|); excludes 0 => marginal evidence of separation
            ci_excludes_zero = bool(not (lo <= 0.0 <= hi))

        rows.append(
            {
                "comparison": label,
                "horizon": hz,
                "model_a": f"{pa}:{ma}",
                "model_b": f"{pb}:{mb}",
                "n_aligned": n,
                "mae_a": mae_a,
                "mae_b": mae_b,
                "delta_mae_point": delta_point,
                "delta_mean_bootstrap": float(np.nanmean(boots)) if boots.size else float("nan"),
                "delta_lower95": lo,
                "delta_upper95": hi,
                "ci_excludes_zero": ci_excludes_zero,
                "n_bootstrap": n_boot,
                "block_len": block_len,
            }
        )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
