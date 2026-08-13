# Early Prediction of Urban Air-Quality Episodes

Graduate-level, reproducible machine learning project that predicts whether an **elevated PM2.5 episode** will occur in the **next 24 hours** from historical air-quality and weather observations.

> **Honesty rule:** metrics in tables/figures/reports are produced only by running the pipeline on real data. Placeholder numbers are never invented.

---

## Problem statement

At time \(t\), given pollutants and meteorology observed up to \(t\), estimate:

\[
P\big(\max(\mathrm{PM2.5}_{t+1},\ldots,\mathrm{PM2.5}_{t+24}) \ge \tau\big)
\]

Default threshold \(\tau = 75\,\mu\mathrm{g}/m^3\) (configurable in `configs/config.yaml`).

**Hard constraint:** features must not use future information; evaluation uses **chronological** splits only.

---

## Architecture

```
raw UCI data → validate → preprocess → target + features
    → temporal split → baselines / XGB / LightGBM
    → calibration → ablation → SHAP → reports + optional API
```

| Layer | Path | Role |
|--------|------|------|
| Config | `configs/config.yaml` | thresholds, lags, splits, model knobs |
| Data | `src/data/` | download, load, validate, preprocess |
| Features | `src/features/` | lags/rolling/calendar, target, temporal splits |
| Models | `src/models/` | persistence, logistic, tree, XGB, LightGBM |
| Evaluation | `src/evaluation/` | metrics, calibration, plots |
| Explainability | `src/explainability/` | SHAP |
| API | `api/main.py` | `GET /health`, `POST /predict` |

---

## Dataset

**Beijing Multi-Site Air-Quality Data** (UCI Machine Learning Repository #501)

- URL: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- License: CC BY 4.0 (UCI redistribution)
- Coverage: hourly, 2013–2017, 12 Beijing monitoring sites
- Variables: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM
- **Not native:** relative humidity → derived from TEMP + DEWP (Magnus formula)
- Default experiment site: `Aotizhongxin` (change in config)

Citation: Zhang et al. (2017), *Cautionary Tales on Air-Quality Improvement in Beijing*, Proc. Royal Society A.

---

## Setup

```bash
cd urban-air-quality-prediction
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS note:** XGBoost and LightGBM need OpenMP:

```bash
brew install libomp
```

If OpenMP is missing, the training pipeline still runs using a documented
`HistGradientBoosting` fallback so you get real metrics; install `libomp` to
enable the native XGBoost / LightGBM paths.

### Download data

```bash
python scripts/download_data.py
```

Writes CSVs under `data/raw/beijing_multisite/` and a dataset card at `data/external/dataset_card.json`.

---

## Training & evaluation

```bash
# Full experiment (download + train + figures/tables)
python scripts/train_evaluate.py

# Faster smoke run (fewer tuning iterations)
python scripts/train_evaluate.py --quick

# Regenerate markdown report from artifacts
python scripts/generate_report.py
```

**Outputs (all from real runs):**

- `reports/tables/baseline_results.csv`
- `reports/tables/model_comparison.csv`
- `reports/tables/calibration_results.csv`
- `reports/tables/ablation_results.csv`
- `reports/tables/leakage_experiment.json`
- `reports/tables/test_predictions.csv`
- `reports/figures/*.png`
- `models/artifacts/best_model.joblib`
- `reports/technical_report.md` (after `generate_report.py`)

---

## Tests

```bash
pytest
```

Covers preprocessing, lags/rolling, targets, chronological splits, leakage guards, and metrics.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_eda.ipynb` | distributions, missingness, seasonality, events |
| `notebooks/02_feature_engineering.ipynb` | feature groups + dictionary |
| `notebooks/03_baseline_models.ipynb` | baseline table viewer |
| `notebooks/04_advanced_models.ipynb` | model comparison viewer |
| `notebooks/05_calibration.ipynb` | reliability diagrams |
| `notebooks/06_error_analysis.ipynb` | FP/FN analysis on test predictions |

---

## API (optional)

```bash
# after training so best_model.joblib exists
uvicorn api.main:app --reload --port 8000
```

- `GET /health`
- `POST /predict` with JSON `{"features": {"PM2.5": 80, ...}}` matching training columns

Docker:

```bash
docker build -t urban-aq .
docker run -p 8000:8000 urban-aq
```

---

## Results

Metrics below come from an actual chronological hold-out run on site
`Aotizhongxin` with threshold **150 µg/m³** (see `reports/tables/`).
Re-run locally after `brew install libomp` to include native XGBoost/LightGBM.

| Model | Split | PR-AUC | ROC-AUC | F1 | Brier |
|-------|-------|--------|---------|----|-------|
| persistence | test | 0.673 | 0.736 | 0.644 | 0.191 |
| logistic_regression | test | 0.885 | 0.911 | 0.784 | 0.124 |
| decision_tree | test | 0.863 | 0.886 | 0.761 | 0.128 |
| hist_gradient_boosting* | test | 0.902 | 0.918 | 0.795 | 0.111 |

\*Native XGBoost/LightGBM were unavailable in the generating environment (missing
`libomp`); sklearn `HistGradientBoosting` was used as the honest advanced-model
fallback. After installing OpenMP, re-run `python scripts/train_evaluate.py`.

Leakage demo (decision tree): leaked future-max feature → test PR-AUC **1.000**;
clean features → **0.839**.

Full tables/figures: `reports/tables/`, `reports/figures/`, `reports/technical_report.md`.

---

## Limitations

- Single-site primary experiments
- Fixed episode threshold (jurisdictions differ)
- Imputation of sensor gaps
- Predictive, not causal
- Derived RH

---

## Reproducibility checklist

1. Pin dependencies via `requirements.txt`
2. Set `project.random_seed` in config
3. Chronological split ratios in config
4. Tune only on validation / rolling-origin folds
5. Hold out the newest period as the final test set
6. Run `pytest` before trusting metrics
7. Regenerate the technical report from artifacts

---

## Project layout

See repository tree under `src/`, `scripts/`, `notebooks/`, `tests/`, `reports/`.
