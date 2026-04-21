## Air Pollution (Tema) experiments

This workspace contains the CAS-style manuscript (**`Airpolution.tex`**, build with `latexmk` or Overleaf), the dataset (**`Tema Data.csv`**), and a reproducible training/evaluation pipeline.

### Project layout

| Path | Purpose |
|------|---------|
| **`Airpolution.tex`** / **`cas-refs.bib`** | Main paper and bibliography; figures resolve via `\graphicspath{{figs/}{figures/}{./}{reports/figures/}}` (see preamble). |
| **`Tema Data.csv`** | Raw hourly sensor CSV at repo root (input to the whole pipeline). |
| **`pyproject.toml`** | Editable install name **`airpollution-tema`**; Python package sources live under `src/`. |
| **`requirements.txt`** | Pinned Python deps (tabular + plotting; deep learning backends installed separately—see Notes). |
| **`src/airpollution/`** | Library code: **`constants`**, **`io`**, **`preprocess`**, **`features`**, **`tabular`**, **`sequences`**, **`eval`**, **`utils`**. |
| **`scripts/`** | Pipeline **`01`**–**`08`**; optional **`09`** (MAE-by-horizon figure), **`10_results_scatter_plots.py`** (24 h RF/SVR/XGBoost diagnostic panels), **`11_regime_metrics_export.py`** (Harmattan regime MAE/RMSE/R² → CSV). Optional multi-horizon deep learning: **`06_build_sequence_dataset_multihorizon.py`** + **`07_train_lstm_multihorizon.py`** (single model predicts all horizons). **`project_path.py`** prepends `src/` so `import airpollution` works without `pip install -e .`. |
| **`data/processed/`** | Parquet artifacts from tabular dataset build (written by **`04`**). |
| **`data/features/`** | Horizon-specific supervised feature tables (`tabular_h*.parquet`). |
| **`data/sequences/`** | **`sequences_h*_L*.npz`** tensors for **`07`** deep training. |
| **`reports/tables/`** | CSV metrics (QC, EDA summaries, **`results_merged_*.csv`**, **`results_tabular_regime_metrics_wide.csv`** from **`11`**, etc.). |
| **`reports/figures/`** | PNGs from EDA (**`02`**/**`03`**), **`09`** (MAE bars), **`10`** (`results_scatter_*_h24.png`). |
| **`figures/`** / **`figs/`** | Optional mirrors of key PNG/PDF assets for LaTeX, if you separate assets from **`reports/figures/`**. |

**Dependency direction (high level):** raw CSV → **`01`** (QC) → **`02`**/**`03`** (EDA) → **`04`**/**`05`** (tabular models) → **`06`**/**`07`** (sequences + LSTM) → **`08`** (merge metrics). **`09`**–**`11`** run after **`08`** (and **`11`** reuses tabular training from **`05`**).

### Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the pipeline (in order)

```bash
source .venv/bin/activate

# 1) Data audit + QC tables
python scripts/01_data_audit.py

# 2) EDA figures for Results 3.1 / 3.2
python scripts/02_eda_distribution.py
python scripts/03_eda_diurnal.py

# 3) Tabular ML baselines + RF/SVR/XGBoost
python scripts/04_make_tabular_dataset.py
python scripts/05_train_tabular_models.py

# 4) Sequence tensors + deep models (LSTM, LSTM + multi-head attention)
python scripts/06_build_sequence_dataset.py
python scripts/07_train_lstm_models.py

# Optional: multi-horizon deep learning (one model predicts 24h/7d/14d/28d jointly)
python scripts/06_build_sequence_dataset_multihorizon.py
python scripts/07_train_lstm_multihorizon.py

# 5) Merge tabular + deep metrics (LaTeX-friendly tables)
python scripts/08_merge_results.py

# Optional: MAE-by-horizon bar chart from merged wide metrics
python scripts/09_results_forecast_figures.py

# Optional: 24 h tabular scatter / residual / time panels (RF, linear SVR, XGBoost)
python scripts/10_results_scatter_plots.py

# Optional: regime-stratified tabular metrics (pre-Harmattan vs Harmattan) on an extended test window
# Default test length for regime export is 84 label-time days so both regimes appear (28-day test can be single-regime).
# Tunable: AIRP_REGIME_TEST_DAYS, AIRP_RF_TREES, AIRP_XGB_TREES (see script header).
python scripts/11_regime_metrics_export.py
```

Outputs are written to:
- `reports/figures/`: publication-ready plots (EDA, MAE-by-horizon, scatter diagnostics)
- `reports/tables/`: CSV tables for LaTeX import
- `data/sequences/`: compressed `.npz` sequence tensors per horizon
- `reports/tables/results_deep_model_metrics.csv`: deep-model metrics (after step 4)
- `reports/tables/results_deep_model_metrics_multihorizon.csv`: multi-horizon deep-model metrics (after optional multi-horizon run)
- `reports/tables/results_merged_long.csv`: combined long-format metrics (tabular + deep)
- `reports/tables/results_merged_wide_mae.csv`: MAE pivot by horizon (for quick copy into LaTeX)
- `reports/tables/results_tabular_regime_metrics_wide.csv`: overall + regime MAE/RMSE/R² per horizon/model (after **`11`**)

### Notes

- **Deep learning (`07`) and Python 3.14:** As of early 2026, **TensorFlow**, **jaxlib**, and **PyTorch** often have **no `pip` wheels for Python 3.14**, so `pip install tensorflow` fails with “no matching distribution”. Scripts **01–06, 05, 08** work on 3.14; for **LSTM training**, use **Python 3.11 or 3.12** in a separate environment:
  ```bash
  python3.12 -m venv .venv312
  source .venv312/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  pip install -e .
  pip install tensorflow
  python scripts/07_train_lstm_models.py
  ```
  Install **TensorFlow after** the other requirements so pip can align **NumPy** with TF (macOS x86 often only has TF 2.16, which needs NumPy 1.26, not 2.x).
  The script tries backends in order: **TensorFlow → JAX → PyTorch** when your Python version has wheels.
- **XGBoost on macOS** often requires OpenMP: `brew install libomp`.
- If XGBoost fails to load, `05_train_tabular_models.py` falls back to scikit-learn `HistGradientBoostingRegressor` where configured.

