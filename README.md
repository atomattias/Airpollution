## Air Pollution (Tema) experiments

This workspace contains the draft paper (`Airpolution.tex`) and the dataset (`Tema Data.csv`).

### Project layout

| Path | Purpose |
|------|---------|
| **`Tema Data.csv`** | Raw hourly sensor CSV at repo root (input to the whole pipeline). |
| **`pyproject.toml`** | Editable install name **`airpollution-tema`**; Python package sources live under `src/`. |
| **`requirements.txt`** | Pinned Python deps (tabular + plotting; deep learning backends installed separately—see Notes). |
| **`src/airpollution/`** | Library code: **`constants`**, **`io`**, **`preprocess`**, **`features`**, **`tabular`**, **`sequences`**, **`eval`**, **`utils`**. |
| **`scripts/`** | Runnable pipeline (**`01`** … **`08`**), **`project_path.py`** (prepends `src/` so `import airpollution` works without `pip install -e .`), **`09_results_forecast_figures.py`** (optional MAE-by-horizon plot from merged metrics). |
| **`data/processed/`** | Parquet artifacts from tabular dataset build (written by **`04`**). |
| **`data/features/`** | Horizon-specific supervised feature tables (`tabular_h*.parquet`). |
| **`data/sequences/`** | **`sequences_h*_L*.npz`** tensors for **`07`** deep training. |
| **`reports/tables/`** | CSV metrics (QC, EDA summaries, `results_*` for tables / Overleaf copy-paste). |
| **`reports/figures/`** | PNG figures from **`01`** / **`02`** / **`03`** / **`09`** (boxplots, diurnal, heatmap, MAE bars). |
| **`figures/`** / **`figs/`** | Optional mirrors of key PNG/PDF assets for LaTeX (`\graphicspath`), if you separate assets from **`reports/figures/`**. |

**Dependency direction (high level):** raw CSV → **`01`** (QC) → **`02`**/**`03`** (EDA) → **`04`**/**`05`** (tabular models) → **`06`**/**`07`** (sequences + LSTM) → **`08`** (merge metrics). **`09`** consumes **`reports/tables/results_merged_wide_mae.csv`** after **`08`**.

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

# 5) Merge tabular + deep metrics (LaTeX-friendly tables)
python scripts/08_merge_results.py

# Optional: MAE-by-horizon bar chart from merged wide metrics
python scripts/09_results_forecast_figures.py
```

Outputs are written to:
- `reports/figures/`: publication-ready plots
- `reports/tables/`: CSV tables for LaTeX import
- `data/sequences/`: compressed `.npz` sequence tensors per horizon
- `reports/tables/results_deep_model_metrics.csv`: deep-model metrics (after step 4)
- `reports/tables/results_merged_long.csv`: combined long-format metrics (tabular + deep)
- `reports/tables/results_merged_wide_mae.csv`: MAE pivot by horizon (for quick copy into LaTeX)

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

