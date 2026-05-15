# Revision TODO — PM₂.₅ multi-horizon forecasting paper

Checklist follows the recommended sequence: **evidence layer first** (predictions → time-aware uncertainty → pre-specified tests → anomaly defense), then **manuscript**, then **clarity / interpretability**, then **supplement**.

**Implemented in repo:** prediction CSVs from **`05`** / **`07`** / **`07_train_lstm_multihorizon`**, **`12_bootstrap_confidence_intervals.py`**, **`08_merge_results.py`** (merges CIs into **`results_merged_long.csv`**), **`13_statistical_tests.py`**, **`15_horizon_variability_analysis.py`**, **`09_results_forecast_figures.py`** (MAE error bars when CIs exist), **`Airpolution_updated.tex`** (Methods subsection + limitations/tone), **`README.md`**. Re-run training → **`12`** → **`08`** → **`13`** / **`15`** / **`09`**, then wire CI columns into LaTeX tables.

---

## Tier A — Minimum viable reviewer defense (do these first)

### A1. Prediction export infrastructure (blocking)

- [ ] Add `reports/predictions/` (or agreed path) and document it in README.
- [ ] **Tabular** (`scripts/05_train_tabular_models.py`): after each model’s test predictions, export one CSV per **model × horizon** with at least: `target_time`, `y_true`, `y_pred`, `harmattan` (or NaN if absent), `location` if available, `model`, `horizon`.
- [ ] **Per-horizon deep** (`scripts/07_train_lstm_models.py`): same schema; filenames e.g. `deep_h24_lstm_predictions.csv`.
- [ ] **Joint multi-horizon deep** (`scripts/07_train_lstm_multihorizon.py`): export **one CSV per horizon** from `(N, K)` outputs (`h24`, `h168`, `h336`, `h672`).
- [ ] Add optional metadata columns for reproducibility: `split_start`, `split_end`, `seed`, `seq_len`, backend name (where applicable).

### A2. Uncertainty — prefer time-aware resampling

- [ ] New script (e.g. `scripts/12_bootstrap_confidence_intervals.py`): load each prediction CSV; compute **MAE**, **RMSE**, **MAE⁹⁰** (or your exact tail definition) per resample.
- [ ] **Primary:** **block bootstrap** (or moving-block bootstrap) on test **time-ordered** indices to respect autocorrelation; document block length choice and sensitivity note if you run one alternative.
- [ ] **Secondary (optional):** simple paired row bootstrap only as a sensitivity check, not as the only headline.
- [ ] **B = 1000** resamples; output `reports/tables/results_confidence_intervals.csv` with mean, std, lower 95%, upper 95% per model × horizon × metric.
- [ ] Extend `scripts/08_merge_results.py` (or follow-on step) to merge CI columns into the main wide/long tables used by LaTeX (`mae_lower95`, `mae_upper95`, same for `rmse`, `mae90`).

### A3. Statistical comparison — small, pre-specified set

- [ ] New script (e.g. `scripts/13_statistical_tests.py`): **paired** loss differentials on **aligned** rows (same `target_time` × site × horizon).
- [ ] **Pre-specify** (adjust names to your pipeline keys) e.g.:
  - joint `mh_lstm` vs per-horizon `lstm`
  - `mh_lstm_mha` vs `mh_lstm`
  - `mh_lstm_mha` vs `ridge_ar_lag_fourier` (or your main tabular baseline)
- [ ] Use **block bootstrap** (same blocks as A2) for the distribution of **ΔMAE** (and optionally ΔMAE⁹⁰); report mean Δ, 95% CI, two-sided **p-value** (proportion of bootstrap samples with opposite sign / null-centered rule — document exactly).
- [ ] **Multiplicity:** state that multiple horizons × pairs were tested; use conservative wording or Bonferroni/FDR if you want stronger claims.
- [ ] Optional later: **Diebold–Mariano** on loss-differential series with **HAC / block** treatment if you add it — do not claim i.i.d. errors without adjustment.
- [ ] Output `reports/tables/results_model_significance_tests.csv`; add significance markers only where protocol + multiplicity are described.

### A4. 14-day “anomaly” — defend or contextualize

- [ ] New script (e.g. `scripts/15_horizon_variability_analysis.py`): on **test** labels (or full series, clearly stated), per horizon: mean, std, variance, IQR; optional lag-1 autocorrelation of **targets** or residuals vs naive persistence difficulty.
- [ ] Figures: distribution or boxplot by horizon; optional rolling variance — export to `reports/figures/`.
- [ ] **Results + Discussion:** one tight paragraph — why 14d might look easier **in this window** (variance, regime mix, evaluation span); **avoid overgeneralizing**.

### A5. Manuscript — limitations and R²

- [ ] **Discussion (early):** one paragraph — limited temporal coverage, single Harmattan transition where relevant, rankings may be **unstable**, findings = **feasibility / evidence**, not definitive global superiority.
- [ ] **Discussion:** **R² vs MAE** — episodic variance, noise, skew; why MAE remains operationally useful even when R² is weak or negative.
- [ ] **Tone pass:** replace loaded phrasing (“catastrophic”, “dramatic”) with precise quantitative language.

---

## Tier B — High value after Tier A

### B1. Figures and tables

- [ ] Update main results table(s) with **95% CI** columns (or footnote format) for MAE, RMSE, MAE⁹⁰.
- [ ] Update `scripts/09_results_forecast_figures.py` (or equivalent) — **error bars / CI** on MAE-by-horizon plot using merged CI file.
- [ ] **Architecture diagrams:** per-horizon LSTM, joint multi-horizon, joint + MHA — inputs, shared trunk, heads; caption with **tensor shapes** where possible.

### B2. Interpretability (pick one path first)

- [ ] **Path 1 (faster):** ablation table — drop meteorology, drop Fourier, drop long lags, drop Harmattan indicator (one column per ablation); same splits; MAE / MAE⁹⁰.
- [ ] **Path 2 (richer):** `scripts/14_feature_importance.py` — RF + XGBoost importances; optional **SHAP** (`TreeExplainer`) + summary plot; optional `feature_group_importance.csv` (lags / met / calendar / Fourier / Harmattan / location).

### B3. Regime evaluation for deep models

- [ ] Extend test window / protocol where needed (e.g. env `AIRP_TEST_DAYS=84` or repo equivalent) so deep test is not **accidentally** single-regime; document constraint (sequence length vs regime span).
- [ ] Optional script `16_deep_regime_metrics.py`: MAE pre-Harmattan vs Harmattan vs Δ for selected deep runs; align with tabular regime export story.

### B4. Reproducibility (short section)

- [ ] Pin **Python**, **TF / torch / jax** (whichever used), **CUDA** if GPU; CPU RAM; **git commit** or tag; **seeds** for numpy/TF/torch.
- [ ] Point to **public repo** (when available) and key config files.

---

## Tier C — Nice to have / supplement

- [ ] Rename or gloss **MAE⁹⁰** once (“tail-hour MAE”, “upper-decile MAE”) + one sentence on operational relevance.
- [ ] Supplementary PDF: extra residual histograms, QQ plots, horizon error histograms, optional attention snapshots, hyperparameter table, learning curves if logged.
- [ ] **Multi-seed** deep runs (2–3 seeds) on **one** flagship setting if compute allows; report mean ± std in supplement.
- [ ] **Rolling-origin** evaluation — only if time/compute permits or reviewer requests.

---

## Done criteria (Tier A)

- [ ] Every number in the main comparison table is backed by **exported predictions** and **block-bootstrap CIs**.
- [ ] Headline pairwise claims reference **pre-specified tests** + **multiplicity** language.
- [ ] 14d behavior is **explained** with a small variability figure and cautious wording.
- [ ] Discussion explicitly covers **limitations**, **R² vs MAE**, and **toned** claims.
