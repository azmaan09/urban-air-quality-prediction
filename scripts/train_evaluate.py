#!/usr/bin/env python3
"""
End-to-end training / evaluation pipeline.

Runs: preprocess → features → temporal split → baselines → XGB/LGBM
      → calibration → ablation → leakage demo → SHAP → persist tables/figures.

All metrics written to disk are computed from actual model predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_path
from src.data.loader import download_dataset, load_site
from src.data.preprocessing import preprocess_site
from src.data.validation import print_validation_report, validate_raw_frame
from src.evaluation.calibration import apply_calibrator, fit_calibrator
from src.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    metrics_to_row,
)
from src.evaluation.visualization import (
    save_confusion_matrix,
    save_feature_importance,
    save_reliability_diagram,
    save_roc_pr_curves,
)
from src.explainability.shap_analysis import (
    compute_shap_values,
    save_summary_plot,
    select_error_indices,
)
from src.features.engineering import build_feature_matrix, feature_groups
from src.features.splits import assert_no_temporal_leakage, create_temporal_split
from src.features.targets import create_target_from_config, target_prevalence
from src.models.availability import (
    LIGHTGBM_AVAILABLE,
    LIGHTGBM_ERROR,
    XGBOOST_AVAILABLE,
    XGBOOST_ERROR,
    booster_status,
)
from src.models.baselines import train_decision_tree, train_logistic, train_persistence
from src.models.hist_gbm import train_hist_gbm

if XGBOOST_AVAILABLE:
    from src.models.xgboost_model import train_xgboost
else:
    train_xgboost = None  # type: ignore[assignment]

if LIGHTGBM_AVAILABLE:
    from src.models.lightgbm_model import train_lightgbm
else:
    train_lightgbm = None  # type: ignore[assignment]


warnings.filterwarnings("ignore", category=UserWarning)


def _matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=float)


def _tune_xgb(X_tr, y_tr, X_va, y_va, seed: int, n_iter: int = 12):
    if not XGBOOST_AVAILABLE:
        raise ImportError(XGBOOST_ERROR or "xgboost unavailable")
    space = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "n_estimators": [200, 300, 400],
    }
    best_score, best_params, best_model = -1.0, None, None
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        model = train_xgboost(X_tr, y_tr, X_va, y_va, params=dict(params), random_state=seed)
        prob = model.predict_proba(X_va)[:, 1]
        score = classification_metrics(y_va, prob)["average_precision"]
        if score > best_score:
            best_score, best_params, best_model = score, dict(params), model
    return best_model, best_params, best_score


def _tune_lgbm(X_tr, y_tr, X_va, y_va, seed: int, n_iter: int = 12):
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(LIGHTGBM_ERROR or "lightgbm unavailable")
    space = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "n_estimators": [200, 300, 400],
    }
    best_score, best_params, best_model = -1.0, None, None
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        model = train_lightgbm(X_tr, y_tr, X_va, y_va, params=dict(params), random_state=seed)
        prob = model.predict_proba(X_va)[:, 1]
        score = classification_metrics(y_va, prob)["average_precision"]
        if score > best_score:
            best_score, best_params, best_model = score, dict(params), model
    return best_model, best_params, best_score


def _tune_hist_gbm(X_tr, y_tr, X_va, y_va, seed: int, n_iter: int = 12):
    space = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_iter": [200, 300, 400],
        "l2_regularization": [0.5, 1.0, 2.0, 5.0],
        "min_samples_leaf": [20, 50, 100],
    }
    best_score, best_params, best_model = -1.0, None, None
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        model = train_hist_gbm(X_tr, y_tr, X_va, y_va, params=dict(params), random_state=seed)
        prob = model.predict_proba(X_va)[:, 1]
        score = classification_metrics(y_va, prob)["average_precision"]
        if score > best_score:
            best_score, best_params, best_model = score, dict(params), model
    return best_model, best_params, best_score


def _train_boosted(X_tr, y_tr, X_va, y_va, seed: int, params=None):
    """Prefer XGBoost, else LightGBM, else HistGradientBoosting."""
    if XGBOOST_AVAILABLE:
        return train_xgboost(X_tr, y_tr, X_va, y_va, params=params, random_state=seed)
    if LIGHTGBM_AVAILABLE:
        return train_lightgbm(X_tr, y_tr, X_va, y_va, params=params, random_state=seed)
    return train_hist_gbm(X_tr, y_tr, X_va, y_va, params=params, random_state=seed)


def run_ablation(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_cols, seed: int) -> pd.DataFrame:
    groups = feature_groups(feature_cols)
    experiments = {
        "A_all_features": feature_cols,
        "B_no_rolling": [c for c in feature_cols if c not in groups["rolling"]],
        "C_no_pollutant_lags": [c for c in feature_cols if c not in groups["pollutant_lags"]],
        "D_no_weather": [c for c in feature_cols if c not in groups["weather"]],
        "E_no_calendar": [c for c in feature_cols if c not in groups["calendar"]],
        "F_only_historical_pm25": [
            c
            for c in feature_cols
            if c == "PM2.5" or c.startswith("PM25_lag_") or (c.startswith("PM25_rolling_"))
        ],
        "G_only_weather": groups["weather"],
    }
    rows = []
    for name, cols in experiments.items():
        if len(cols) < 2:
            continue
        idx = [feature_cols.index(c) for c in cols]
        model = _train_boosted(X_tr[:, idx], y_tr, X_va[:, idx], y_va, seed)
        for split, Xs, ys in [("valid", X_va, y_va), ("test", X_te, y_te)]:
            prob = model.predict_proba(Xs[:, idx])[:, 1]
            m = classification_metrics(ys, prob)
            rows.append(metrics_to_row(name, split, m))
    return pd.DataFrame(rows)


def run_leakage_experiment(df_feat: pd.DataFrame, feature_cols: list[str], split, seed: int):
    """
    Educational demo: intentionally add future max PM2.5 as a feature, measure
    inflated performance, then remove it.
    """
    leaked_cols = feature_cols + ["y_future_max_pm25"]
    X_all = df_feat[leaked_cols].to_numpy(dtype=float)
    y = df_feat["y_episode"].to_numpy(int)
    X_tr, y_tr = X_all[split.train_idx], y[split.train_idx]
    X_te, y_te = X_all[split.test_idx], y[split.test_idx]
    # Use a simple tree so the leak is obvious
    model = train_decision_tree(X_tr, y_tr, random_state=seed, max_depth=4)
    leaked_prob = model.predict_proba(X_te)[:, 1]
    leaked_metrics = classification_metrics(y_te, leaked_prob)

    X_clean = df_feat[feature_cols].to_numpy(dtype=float)
    model2 = train_decision_tree(
        X_clean[split.train_idx], y_tr, random_state=seed, max_depth=4
    )
    clean_prob = model2.predict_proba(X_clean[split.test_idx])[:, 1]
    clean_metrics = classification_metrics(y_te, clean_prob)

    return {
        "leakage_source": "Included y_future_max_pm25 (max PM2.5 over the prediction horizon) as an input feature.",
        "prevention": "Target-derived future aggregates must never enter X; enforced by feature_cols list + leakage tests.",
        "leaked_test_metrics": leaked_metrics,
        "clean_test_metrics": clean_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--n-tune", type=int, default=None, help="Override tuning iterations")
    parser.add_argument("--quick", action="store_true", help="Fewer tuning iters for smoke runs")
    args = parser.parse_args()

    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    np.random.seed(seed)

    fig_dir = project_path(cfg["paths"]["figures"])
    tab_dir = project_path(cfg["paths"]["tables"])
    art_dir = project_path(cfg["paths"]["artifacts"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_dataset()

    print("=== Load & validate ===")
    raw = load_site()
    report = validate_raw_frame(raw)
    print_validation_report(report)
    report.raise_if_invalid()

    print("=== Preprocess ===")
    clean = preprocess_site(raw, cfg)
    clean.to_parquet(project_path("data/processed/site_clean.parquet"), index=False)

    print("=== Target + features ===")
    labeled = create_target_from_config(clean, cfg)
    prev = target_prevalence(labeled["y_episode"])
    print("Target prevalence (pre-feature dropna):", prev)
    feat_df, feature_cols = build_feature_matrix(labeled, cfg)
    prev2 = target_prevalence(feat_df["y_episode"])
    print("Target prevalence (modeling table):", prev2)
    print(f"n_samples={len(feat_df)}, n_features={len(feature_cols)}")
    feat_df.to_parquet(project_path("data/processed/modeling_table.parquet"), index=False)
    (art_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))

    print("=== Temporal split ===")
    split = create_temporal_split(
        feat_df["timestamp"],
        train_ratio=cfg["splits"]["train_ratio"],
        valid_ratio=cfg["splits"]["valid_ratio"],
        test_ratio=cfg["splits"]["test_ratio"],
    )
    assert_no_temporal_leakage(
        feat_df.loc[split.train_idx, "timestamp"],
        feat_df.loc[split.valid_idx, "timestamp"],
        dual="valid",
    )
    assert_no_temporal_leakage(
        feat_df.loc[split.valid_idx, "timestamp"],
        feat_df.loc[split.test_idx, "timestamp"],
        dual="test",
    )
    split_info = {
        "train": split.train_time_range,
        "valid": split.valid_time_range,
        "test": split.test_time_range,
        "n_train": int(len(split.train_idx)),
        "n_valid": int(len(split.valid_idx)),
        "n_test": int(len(split.test_idx)),
    }
    print(json.dumps(split_info, indent=2))
    (tab_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    X = _matrix(feat_df, feature_cols)
    y = feat_df["y_episode"].to_numpy(dtype=int)
    X_tr, y_tr = X[split.train_idx], y[split.train_idx]
    X_va, y_va = X[split.valid_idx], y[split.valid_idx]
    X_te, y_te = X[split.test_idx], y[split.test_idx]

    thr = float(cfg["target"]["threshold_ug_m3"])
    pm_idx = feature_cols.index("PM2.5")
    decision_thr = float(cfg["evaluation"]["decision_threshold"])

    result_rows = []

    print("=== Baselines ===")
    persistence = train_persistence(X_tr, y_tr, pm25_index=pm_idx, threshold=thr)
    logistic = train_logistic(X_tr, y_tr, random_state=seed)
    tree = train_decision_tree(X_tr, y_tr, random_state=seed)

    baseline_models = {
        "persistence": persistence,
        "logistic_regression": logistic,
        "decision_tree": tree,
    }
    for name, model in baseline_models.items():
        for split_name, Xs, ys in [("valid", X_va, y_va), ("test", X_te, y_te)]:
            prob = model.predict_proba(Xs)[:, 1]
            m = classification_metrics(ys, prob, threshold=decision_thr)
            result_rows.append(metrics_to_row(name, split_name, m))
            print(f"{name:22s} {split_name:5s} PR-AUC={m['average_precision']:.4f} ROC-AUC={m['roc_auc']:.4f}")

    baseline_df = pd.DataFrame([r for r in result_rows if r["model"] in baseline_models])
    baseline_df.to_csv(tab_dir / "baseline_results.csv", index=False)

    print("=== Advanced models (tuned on validation) ===")
    print("Booster status:", json.dumps(booster_status(), indent=2)[:500])
    n_tune = args.n_tune if args.n_tune is not None else (4 if args.quick else int(cfg["models"]["tuning"]["n_iter"]))
    advanced = {}

    if XGBOOST_AVAILABLE:
        xgb_model, xgb_params, xgb_val = _tune_xgb(X_tr, y_tr, X_va, y_va, seed, n_iter=n_tune)
        print(f"Best XGB valid PR-AUC={xgb_val:.4f} params={xgb_params}")
        (art_dir / "xgb_best_params.json").write_text(json.dumps(xgb_params, indent=2))
        advanced["xgboost"] = xgb_model
    else:
        print(f"WARNING: XGBoost unavailable ({XGBOOST_ERROR}). On macOS: brew install libomp")

    if LIGHTGBM_AVAILABLE:
        lgb_model, lgb_params, lgb_val = _tune_lgbm(X_tr, y_tr, X_va, y_va, seed, n_iter=n_tune)
        print(f"Best LGBM valid PR-AUC={lgb_val:.4f} params={lgb_params}")
        (art_dir / "lgbm_best_params.json").write_text(json.dumps(lgb_params, indent=2))
        advanced["lightgbm"] = lgb_model
    else:
        print(f"WARNING: LightGBM unavailable ({LIGHTGBM_ERROR}). On macOS: brew install libomp")

    if not advanced:
        # Honest fallback so the pipeline still produces real metrics without OpenMP
        hgb_model, hgb_params, hgb_val = _tune_hist_gbm(X_tr, y_tr, X_va, y_va, seed, n_iter=n_tune)
        print(
            f"Using sklearn HistGradientBoosting fallback "
            f"(valid PR-AUC={hgb_val:.4f}) params={hgb_params}"
        )
        (art_dir / "hist_gbm_best_params.json").write_text(json.dumps(hgb_params, indent=2))
        advanced["hist_gradient_boosting"] = hgb_model

    for name, model in advanced.items():
        for split_name, Xs, ys in [("valid", X_va, y_va), ("test", X_te, y_te)]:
            prob = model.predict_proba(Xs)[:, 1]
            m = classification_metrics(ys, prob, threshold=decision_thr)
            result_rows.append(metrics_to_row(name, split_name, m))
            print(f"{name:22s} {split_name:5s} PR-AUC={m['average_precision']:.4f} ROC-AUC={m['roc_auc']:.4f}")

        # Figures on test
        prob_te = model.predict_proba(X_te)[:, 1]
        pred_te = (prob_te >= decision_thr).astype(int)
        save_roc_pr_curves(y_te, prob_te, name, fig_dir, stem=name)
        save_confusion_matrix(y_te, pred_te, f"Confusion — {name}", fig_dir / f"{name}_confusion.png")
        if hasattr(model, "feature_importances_"):
            save_feature_importance(
                feature_cols,
                model.feature_importances_,
                f"Feature importance — {name}",
                fig_dir / f"{name}_feature_importance.png",
            )

    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(tab_dir / "model_comparison.csv", index=False)

    # Pick strongest model by validation PR-AUC
    valid_scores = comparison[comparison["split"] == "valid"].sort_values(
        "average_precision", ascending=False
    )
    best_name = valid_scores.iloc[0]["model"]
    best_model = {**baseline_models, **advanced}[best_name]
    print(f"=== Best by valid PR-AUC: {best_name} ===")

    print("=== Calibration ===")
    raw_va = best_model.predict_proba(X_va)[:, 1]
    raw_te = best_model.predict_proba(X_te)[:, 1]
    calib_rows = []
    for method in cfg["calibration"]["methods"]:
        cal = fit_calibrator(y_va, raw_va, method=method)
        te_cal = apply_calibrator(cal, raw_te)
        m = classification_metrics(y_te, te_cal, threshold=decision_thr)
        m["ece"] = expected_calibration_error(y_te, te_cal)
        m["calibration"] = method
        calib_rows.append(metrics_to_row(best_name, "test", m))
        save_reliability_diagram(
            y_te,
            te_cal,
            f"Reliability — {best_name} / {method}",
            fig_dir / f"reliability_{best_name}_{method}.png",
        )
        print(f"calib={method:8s} Brier={m['brier_score']:.4f} ECE={m['ece']:.4f} PR-AUC={m['average_precision']:.4f}")
    pd.DataFrame(calib_rows).to_csv(tab_dir / "calibration_results.csv", index=False)

    # Persist best calibrated model (isotonic if available)
    best_cal = fit_calibrator(y_va, raw_va, method="isotonic")
    joblib.dump(
        {
            "model": best_model,
            "calibrator": best_cal,
            "feature_cols": feature_cols,
            "model_name": best_name,
            "threshold": thr,
            "decision_threshold": decision_thr,
            "config_target": cfg["target"],
        },
        art_dir / "best_model.joblib",
    )

    print("=== Ablation ===")
    ablation = run_ablation(X_tr, y_tr, X_va, y_va, X_te, y_te, feature_cols, seed)
    ablation.to_csv(tab_dir / "ablation_results.csv", index=False)
    print(ablation[ablation["split"] == "test"][["model", "average_precision", "roc_auc", "f1", "recall", "brier_score"]])

    print("=== Leakage experiment ===")
    leak = run_leakage_experiment(feat_df, feature_cols, split, seed)
    (tab_dir / "leakage_experiment.json").write_text(json.dumps(leak, indent=2, default=float))
    print(
        "Leaked PR-AUC={:.4f} | Clean PR-AUC={:.4f}".format(
            leak["leaked_test_metrics"]["average_precision"],
            leak["clean_test_metrics"]["average_precision"],
        )
    )

    print("=== SHAP (best advanced/tree model) ===")
    adv_valid = valid_scores[valid_scores["model"].isin(list(advanced.keys()))]
    if len(adv_valid) == 0:
        print("No advanced model available for SHAP.")
    else:
        tree_best_name = adv_valid.iloc[0]["model"]
        tree_best = advanced[tree_best_name]
        X_te_df = pd.DataFrame(X_te, columns=feature_cols)
        try:
            explainer, shap_vals, X_shap = compute_shap_values(tree_best, X_te_df, max_samples=800)
            save_summary_plot(shap_vals, X_shap, fig_dir / f"shap_summary_{tree_best_name}.png")
            probs_sample = tree_best.predict_proba(X_shap)[:, 1]
            top_i = int(np.argmax(probs_sample))
            from src.explainability.shap_analysis import save_waterfall_for_index

            save_waterfall_for_index(
                explainer,
                shap_vals,
                X_shap,
                top_i,
                fig_dir / f"shap_waterfall_high_risk_{tree_best_name}.png",
                title="High-risk prediction",
            )
            err = select_error_indices(y_te, tree_best.predict_proba(X_te)[:, 1])
            (tab_dir / "error_index_counts.json").write_text(
                json.dumps({k: int(len(v)) for k, v in err.items()}, indent=2)
            )
        except Exception as e:
            print(f"SHAP skipped due to: {e}")

    # Save test predictions for error-analysis notebook
    pred_frame = feat_df.iloc[split.test_idx][["timestamp", "PM2.5", "y_episode"]].copy()
    pred_frame["y_prob"] = best_model.predict_proba(X_te)[:, 1]
    pred_frame["y_pred"] = (pred_frame["y_prob"] >= decision_thr).astype(int)
    pred_frame.to_csv(tab_dir / "test_predictions.csv", index=False)

    print("=== Done. Tables/figures written under reports/ ===")


if __name__ == "__main__":
    main()
