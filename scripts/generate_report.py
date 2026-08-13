#!/usr/bin/env python3
"""Generate reports/technical_report.md from actual experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config


def _md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_No results available yet. Run `python scripts/train_evaluate.py`._\n"
    use = df if cols is None else df[cols]
    return use.to_markdown(index=False) + "\n"


def main() -> None:
    cfg = load_config()
    tab = ROOT / "reports" / "tables"
    card_path = ROOT / "data" / "external" / "dataset_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}

    baseline = pd.read_csv(tab / "baseline_results.csv") if (tab / "baseline_results.csv").exists() else pd.DataFrame()
    comparison = pd.read_csv(tab / "model_comparison.csv") if (tab / "model_comparison.csv").exists() else pd.DataFrame()
    calib = pd.read_csv(tab / "calibration_results.csv") if (tab / "calibration_results.csv").exists() else pd.DataFrame()
    ablation = pd.read_csv(tab / "ablation_results.csv") if (tab / "ablation_results.csv").exists() else pd.DataFrame()
    leak = json.loads((tab / "leakage_experiment.json").read_text()) if (tab / "leakage_experiment.json").exists() else {}
    split_info = json.loads((tab / "split_info.json").read_text()) if (tab / "split_info.json").exists() else {}

    metric_cols = [
        "model",
        "split",
        "average_precision",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "brier_score",
    ]

    # Abstract numbers from test set if available
    abstract_bits = ""
    if not comparison.empty:
        test = comparison[comparison["split"] == "test"].sort_values("average_precision", ascending=False)
        if len(test):
            top = test.iloc[0]
            abstract_bits = (
                f" On the chronologically held-out test window, the best model "
                f"({top['model']}) achieved PR-AUC={top['average_precision']:.3f}, "
                f"ROC-AUC={top['roc_auc']:.3f}, and Brier={top['brier_score']:.3f}."
            )

    thr = cfg["target"]["threshold_ug_m3"]
    site = cfg["dataset"]["primary_site"]

    leaked_line = ""
    if leak:
        leaked_line = (
            f"- Leaked decision-tree test PR-AUC: "
            f"**{leak['leaked_test_metrics']['average_precision']:.4f}**\n"
            f"- Clean decision-tree test PR-AUC: "
            f"**{leak['clean_test_metrics']['average_precision']:.4f}**\n"
            f"- Leakage source: {leak['leakage_source']}\n"
            f"- Prevention: {leak['prevention']}\n"
        )

    report = f"""# Technical Report: Early Prediction of Urban Air-Quality Episodes

> Generated from actual experiment artifacts. Do not edit metrics by hand.

## 1. Abstract

This project develops a reproducible machine learning system that predicts whether
an elevated PM2.5 episode (PM2.5 ≥ {thr:g} µg/m³) will occur during the next 24 hours
using historical pollutant and meteorological observations from the Beijing Multi-Site
Air-Quality Data (UCI #501), primary site **{site}**.
We emphasize chronological validation, leakage prevention, probability calibration,
ablation analysis, and SHAP explainability.{abstract_bits}

## 2. Introduction

Urban PM2.5 episodes drive acute health risk and operational decisions for public
alerts. Ranking quality (ROC/PR) is necessary but not sufficient: early-warning
systems also require well-calibrated probabilities so operators can set risk thresholds.

## 3. Problem Definition

- **Input at time t:** pollutants, weather, calendar context, and engineered history.
- **Output:** `P(max(PM2.5_{{t+1…t+24}}) ≥ {thr:g})`.
- **Label:** `y_t = 1` if the future 24-hour maximum meets/exceeds the threshold.

## 4. Dataset

- **Name:** {card.get("name", cfg["dataset"]["name"])}
- **Source / URL:** {card.get("url", cfg["dataset"]["url"])}
- **License:** {card.get("license", cfg["dataset"]["license"])}
- **Citation:** {card.get("citation", cfg["dataset"]["citation"]).strip()}
- **Primary site:** {site}
- **All-sites rows:** {card.get("all_sites", {}).get("n_rows", "see dataset_card.json")}
- **Time range (all sites):** {card.get("all_sites", {}).get("time_start")} → {card.get("all_sites", {}).get("time_end")}
- **Available variables:** PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM
- **Not native:** relative humidity (derived from TEMP+DEWP via Magnus formula)

Primary-site summary is stored in `data/external/dataset_card.json`.

## 5. Exploratory Data Analysis

EDA notebooks and figures live under `notebooks/01_eda.ipynb` and `reports/figures/eda_*.png`.
Key checks: missingness, PM2.5 distribution/heavy tails, seasonal and diurnal patterns,
correlations, and class balance of the next-24h episode label.

## 6. Feature Engineering

Configured in `configs/config.yaml`:

- PM2.5 lags: {cfg["features"]["pm25_lags"]}
- Rolling windows: {cfg["features"]["rolling_windows"]}
- Pollutant lags / weather lags / calendar / missingness indicators
- Feature dictionary: `reports/tables/feature_dictionary.csv` (after notebook/pipeline)

## 7. Temporal Validation

Random splits are **forbidden**. Chronological split ratios:
train={cfg["splits"]["train_ratio"]}, valid={cfg["splits"]["valid_ratio"]}, test={cfg["splits"]["test_ratio"]}.

```json
{json.dumps(split_info, indent=2)}
```

## 8. Baseline Models

{_md_table(baseline, metric_cols if not baseline.empty else None)}

## 9. XGBoost

See `model_comparison.csv` and figures `reports/figures/xgboost_*.png`.
Best hyperparameters (validation PR-AUC selection) are stored in
`models/artifacts/xgb_best_params.json`.

## 10. LightGBM

See `model_comparison.csv` and figures `reports/figures/lightgbm_*.png`.
Best hyperparameters: `models/artifacts/lgbm_best_params.json`.

## 11. Calibration

Calibrators are fit on **validation** probabilities only, then applied to test.

{_md_table(calib)}

Reliability diagrams: `reports/figures/reliability_*.png`.

Why calibration matters: an alert threshold of 0.3 should mean roughly 30% of flagged
hours truly experience an episode. Uncalibrated boosting scores often over-concentrate
near 0/1, distorting operational triage.

## 12. Ablation Study

{_md_table(ablation[ablation["split"]=="test"] if not ablation.empty else ablation, metric_cols if not ablation.empty else None)}

## 13. Leakage Analysis

Educational experiment intentionally injects future information, then removes it.

{leaked_line if leaked_line else "_Run the pipeline to populate leakage results._"}

The leaked model is **never** used as the production artifact.

## 14. Explainability

SHAP summary / waterfall plots (when available): `reports/figures/shap_*.png`.

## 15. Error Analysis

Test predictions: `reports/tables/test_predictions.csv`.
Explore false positives/negatives and high-confidence errors in
`notebooks/06_error_analysis.ipynb`.

## 16. Results

Full comparison:

{_md_table(comparison, metric_cols if not comparison.empty else None)}

## 17. Limitations

- Single primary monitoring site (spatial transfer not evaluated).
- Fixed threshold ({thr:g} µg/m³); health-relevant thresholds vary by jurisdiction.
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
"""

    out = ROOT / "reports" / "technical_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
