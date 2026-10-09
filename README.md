# Tema PM2.5 forecasting pipeline (code only)

Python package and numbered scripts for the Tema, Ghana hourly PM2.5
multi-horizon forecasting study. This snapshot contains **code only**.

Not included: manuscript, PDFs, reports, figures, work plans, or the
hourly observations (`Tema Data.csv` / `data/`).

## Layout

- `src/airpollution/` — shared library (I/O, features, evaluation, sequences)
- `scripts/` — numbered pipeline (`01`–`13`, `15`–`22`) plus `project_path.py`
- `pyproject.toml` / `requirements.txt` — install metadata and pinned deps

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Place the raw hourly file where `src/airpollution/io.py` expects it, then
run scripts in order from the repository root.

Source: https://github.com/atomattias/Airpollution (branch `zenodo-code`)
