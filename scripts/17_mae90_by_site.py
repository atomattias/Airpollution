"""Per-site MAE90 from exported test predictions (no retraining)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import project_path  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "reports" / "predictions"
OUT = ROOT / "reports" / "tables" / "results_mae90_by_site_all.csv"

SKIP_MODELS = {"ridge"}  # unregularised duplicate of ridge_ar when both exist


def main() -> None:
    rows = []
    for path in sorted(PRED.glob("*.csv")):
        df = pd.read_csv(path, usecols=lambda c: c in {"y_true", "y_pred", "location", "model", "horizon", "pipeline"})
        if not {"y_true", "y_pred", "location", "model", "horizon"}.issubset(df.columns):
            continue
        model = str(df["model"].iloc[0])
        if model in SKIP_MODELS:
            continue
        hz = str(df["horizon"].iloc[0])
        pipe = str(df["pipeline"].iloc[0]) if "pipeline" in df.columns else ""
        y = df["y_true"].to_numpy(float)
        q_pool = float(np.nanquantile(y, 0.90))
        for site, g in df.groupby(df["location"].astype(str), sort=True):
            yt = g["y_true"].to_numpy(float)
            pr = g["y_pred"].to_numpy(float)
            q_site = float(np.nanquantile(yt, 0.90))
            m_pool = yt >= q_pool
            m_site = yt >= q_site

            def mae(mask: np.ndarray) -> float:
                if int(mask.sum()) == 0:
                    return float("nan")
                return float(np.mean(np.abs(yt[mask] - pr[mask])))

            rows.append(
                {
                    "file": path.name,
                    "pipeline": pipe,
                    "horizon": hz,
                    "model": model,
                    "site": site,
                    "n": int(len(yt)),
                    "q_pool": round(q_pool, 2),
                    "n90_pool": int(m_pool.sum()),
                    "mae90_pool_thr": mae(m_pool),
                    "q_site": round(q_site, 2),
                    "n90_site": int(m_site.sum()),
                    "mae90_site_thr": mae(m_site),
                }
            )
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
