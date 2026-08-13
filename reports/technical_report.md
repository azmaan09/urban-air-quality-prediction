# Technical Report: Early Prediction of Urban Air-Quality Episodes

> Generated from actual experiment artifacts. Do not edit metrics by hand.

## 1. Abstract

This project develops a reproducible machine learning system that predicts whether
an elevated PM2.5 episode (PM2.5 ≥ 150 µg/m³) will occur during the next 24 hours
using historical pollutant and meteorological observations from the Beijing Multi-Site
Air-Quality Data (UCI #501), primary site **Aotizhongxin**.
We emphasize chronological validation, leakage prevention, probability calibration,
ablation analysis, and SHAP explainability. On the chronologically held-out test window, the best model (hist_gradient_boosting) achieved PR-AUC=0.902, ROC-AUC=0.918, and Brier=0.111.

## 2. Introduction

Urban PM2.5 episodes drive acute health risk and operational decisions for public
alerts. Ranking quality (ROC/PR) is necessary but not sufficient: early-warning
systems also require well-calibrated probabilities so operators can set risk thresholds.

## 3. Problem Definition

- **Input at time t:** pollutants, weather, calendar context, and engineered history.
- **Output:** `P(max(PM2.5_{t+1…t+24}) ≥ 150)`.
- **Label:** `y_t = 1` if the future 24-hour maximum meets/exceeds the threshold.

## 4. Dataset

- **Name:** Beijing Multi-Site Air-Quality Data
- **Source / URL:** https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- **License:** CC BY 4.0 (via UCI redistribution)
- **Citation:** Zhang, S., Guo, B., Dong, A., He, J., Xu, B., & Chen, S. X. (2017). Cautionary Tales on Air-Quality Improvement in Beijing. Proceedings of the Royal Society A, 473(2205).
- **Primary site:** Aotizhongxin
- **All-sites rows:** 420768
- **Time range (all sites):** 2013-03-01 00:00:00+08:00 → 2017-02-28 23:00:00+08:00
- **Available variables:** PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM
- **Not native:** relative humidity (derived from TEMP+DEWP via Magnus formula)

Primary-site summary is stored in `data/external/dataset_card.json`.

## 5. Exploratory Data Analysis

EDA notebooks and figures live under `notebooks/01_eda.ipynb` and `reports/figures/eda_*.png`.
Key checks: missingness, PM2.5 distribution/heavy tails, seasonal and diurnal patterns,
correlations, and class balance of the next-24h episode label.

## 6. Feature Engineering

Configured in `configs/config.yaml`:

- PM2.5 lags: [1, 3, 6, 12, 24, 48, 72]
- Rolling windows: {'mean': [6, 12, 24], 'std': [24], 'min': [24], 'max': [24]}
- Pollutant lags / weather lags / calendar / missingness indicators
- Feature dictionary: `reports/tables/feature_dictionary.csv` (after notebook/pipeline)

## 7. Temporal Validation

Random splits are **forbidden**. Chronological split ratios:
train=0.7, valid=0.15, test=0.15.

```json
{
  "train": [
    "2013-03-04 00:00:00+08:00",
    "2015-12-18 20:00:00+08:00"
  ],
  "valid": [
    "2015-12-18 21:00:00+08:00",
    "2016-07-24 09:00:00+08:00"
  ],
  "test": [
    "2016-07-24 10:00:00+08:00",
    "2017-02-27 23:00:00+08:00"
  ],
  "n_train": 24477,
  "n_valid": 5245,
  "n_test": 5246
}
```

## 8. Baseline Models

| model               | split   |   average_precision |   roc_auc |       f1 |   precision |   recall |   brier_score |
|:--------------------|:--------|--------------------:|----------:|---------:|------------:|---------:|--------------:|
| persistence         | valid   |            0.566032 |  0.6989   | 0.569773 |    0.955157 | 0.405972 |      0.168086 |
| persistence         | test    |            0.672601 |  0.736374 | 0.644265 |    0.971735 | 0.481875 |      0.191387 |
| logistic_regression | valid   |            0.800262 |  0.871095 | 0.698283 |    0.644122 | 0.762389 |      0.138786 |
| logistic_regression | test    |            0.885337 |  0.910698 | 0.784431 |    0.74923  | 0.823103 |      0.123736 |
| decision_tree       | valid   |            0.780899 |  0.859332 | 0.70323  |    0.675644 | 0.733164 |      0.142075 |
| decision_tree       | test    |            0.862916 |  0.886499 | 0.760864 |    0.782431 | 0.740454 |      0.127865 |


## 9. XGBoost

See `model_comparison.csv` and figures `reports/figures/xgboost_*.png`.
Best hyperparameters (validation PR-AUC selection) are stored in
`models/artifacts/xgb_best_params.json`.

## 10. LightGBM

See `model_comparison.csv` and figures `reports/figures/lightgbm_*.png`.
Best hyperparameters: `models/artifacts/lgbm_best_params.json`.

## 11. Calibration

Calibrators are fit on **validation** probabilities only, then applied to test.

| model                  | split   |   average_precision |   roc_auc |       f1 |   precision |   recall |   brier_score |   threshold |   support |   prevalence |   tn |   fp |   fn |   tp |       ece | calibration   |
|:-----------------------|:--------|--------------------:|----------:|---------:|------------:|---------:|--------------:|------------:|----------:|-------------:|-----:|-----:|-----:|-----:|----------:|:--------------|
| hist_gradient_boosting | test    |            0.902008 |  0.917924 | 0.795076 |    0.794118 | 0.796037 |      0.11145  |         0.5 |      5246 |     0.394396 | 2750 |  427 |  422 | 1647 | 0.0424421 | none          |
| hist_gradient_boosting | test    |            0.902008 |  0.917924 | 0.795508 |    0.865341 | 0.736104 |      0.112958 |         0.5 |      5246 |     0.394396 | 2940 |  237 |  546 | 1523 | 0.0564834 | platt         |
| hist_gradient_boosting | test    |            0.890355 |  0.912676 | 0.770053 |    0.921833 | 0.661189 |      0.112754 |         0.5 |      5246 |     0.394396 | 3061 |  116 |  701 | 1368 | 0.0454737 | isotonic      |


Reliability diagrams: `reports/figures/reliability_*.png`.

Why calibration matters: an alert threshold of 0.3 should mean roughly 30% of flagged
hours truly experience an episode. Uncalibrated boosting scores often over-concentrate
near 0/1, distorting operational triage.

## 12. Ablation Study

| model                  | split   |   average_precision |   roc_auc |       f1 |   precision |   recall |   brier_score |
|:-----------------------|:--------|--------------------:|----------:|---------:|------------:|---------:|--------------:|
| A_all_features         | test    |            0.903911 |  0.92103  | 0.798058 |    0.80156  | 0.794587 |      0.110167 |
| B_no_rolling           | test    |            0.899881 |  0.917046 | 0.797274 |    0.802941 | 0.791687 |      0.111754 |
| C_no_pollutant_lags    | test    |            0.904836 |  0.923847 | 0.801377 |    0.815408 | 0.78782  |      0.108056 |
| D_no_weather           | test    |            0.889265 |  0.906302 | 0.77934  |    0.781995 | 0.776704 |      0.120718 |
| E_no_calendar          | test    |            0.901285 |  0.915831 | 0.796269 |    0.808978 | 0.783954 |      0.111214 |
| F_only_historical_pm25 | test    |            0.878251 |  0.891456 | 0.778968 |    0.784698 | 0.77332  |      0.124998 |
| G_only_weather         | test    |            0.794407 |  0.85134  | 0.721319 |    0.661427 | 0.793137 |      0.159193 |


## 13. Leakage Analysis

Educational experiment intentionally injects future information, then removes it.

- Leaked decision-tree test PR-AUC: **1.0000**
- Clean decision-tree test PR-AUC: **0.8386**
- Leakage source: Included y_future_max_pm25 (max PM2.5 over the prediction horizon) as an input feature.
- Prevention: Target-derived future aggregates must never enter X; enforced by feature_cols list + leakage tests.


The leaked model is **never** used as the production artifact.

## 14. Explainability

SHAP summary / waterfall plots (when available): `reports/figures/shap_*.png`.

## 15. Error Analysis

Test predictions: `reports/tables/test_predictions.csv`.
Explore false positives/negatives and high-confidence errors in
`notebooks/06_error_analysis.ipynb`.

## 16. Results

Full comparison:

| model                  | split   |   average_precision |   roc_auc |       f1 |   precision |   recall |   brier_score |
|:-----------------------|:--------|--------------------:|----------:|---------:|------------:|---------:|--------------:|
| persistence            | valid   |            0.566032 |  0.6989   | 0.569773 |    0.955157 | 0.405972 |      0.168086 |
| persistence            | test    |            0.672601 |  0.736374 | 0.644265 |    0.971735 | 0.481875 |      0.191387 |
| logistic_regression    | valid   |            0.800262 |  0.871095 | 0.698283 |    0.644122 | 0.762389 |      0.138786 |
| logistic_regression    | test    |            0.885337 |  0.910698 | 0.784431 |    0.74923  | 0.823103 |      0.123736 |
| decision_tree          | valid   |            0.780899 |  0.859332 | 0.70323  |    0.675644 | 0.733164 |      0.142075 |
| decision_tree          | test    |            0.862916 |  0.886499 | 0.760864 |    0.782431 | 0.740454 |      0.127865 |
| hist_gradient_boosting | valid   |            0.822519 |  0.878911 | 0.706304 |    0.666667 | 0.750953 |      0.131317 |
| hist_gradient_boosting | test    |            0.902008 |  0.917924 | 0.795076 |    0.794118 | 0.796037 |      0.11145  |


## 17. Limitations

- Single primary monitoring site (spatial transfer not evaluated).
- Fixed threshold (150 µg/m³); health-relevant thresholds vary by jurisdiction.
- Hourly missingness imputed; long outages may distort lags.
- No causal identification — predictive associations only.
- RH is derived, not measured.

## 18. Future Work

- Multi-site / spatial models and transfer across cities
- Cost-sensitive operating points and decision-curve analysis
- Online monitoring / drift detection
- Conformal risk control for alert rates

## 19. Conclusion

This repository provides a complete, leakage-aware pipeline for next-24h elevated
PM2.5 episode prediction with baselines, gradient boosting, calibration, ablation,
explainability, tests, and an optional FastAPI service. All reported metrics are
produced by `scripts/train_evaluate.py` from real data.
