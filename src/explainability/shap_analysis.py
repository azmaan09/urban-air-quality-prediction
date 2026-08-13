"""SHAP-based explainability for tree models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_shap_values(model: Any, X: pd.DataFrame | np.ndarray, max_samples: int = 1000):
    """
    Compute SHAP values with TreeExplainer when possible.

    Returns (explainer, shap_values, X_used)
    """
    import shap

    if isinstance(X, pd.DataFrame):
        X_use = X.sample(n=min(max_samples, len(X)), random_state=42)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=min(max_samples, len(X)), replace=False)
        X_use = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_use)
    # LightGBM / XGBoost binary: sometimes list of two arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values, X_use


def save_summary_plot(
    shap_values,
    X: pd.DataFrame,
    out_path: Path,
    title: str = "SHAP summary",
) -> Path:
    import shap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=25)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def save_waterfall_for_index(
    explainer,
    shap_values,
    X: pd.DataFrame,
    index: int,
    out_path: Path,
    title: str,
) -> Path:
    import shap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer Explanation API when available
    try:
        exp = shap.Explanation(
            values=shap_values[index],
            base_values=explainer.expected_value if np.isscalar(explainer.expected_value)
            else explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value,
            data=X.iloc[index].values,
            feature_names=list(X.columns),
        )
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=20)
    except Exception:
        # Fallback: force bar of local contributions
        vals = shap_values[index]
        order = np.argsort(np.abs(vals))[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(np.array(X.columns)[order][::-1], vals[order][::-1])
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def select_error_indices(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "false_positives": np.where((y_pred == 1) & (y_true == 0))[0],
        "false_negatives": np.where((y_pred == 0) & (y_true == 1))[0],
        "high_risk_correct": np.where((y_pred == 1) & (y_true == 1) & (y_prob >= 0.8))[0],
        "high_confidence_errors": np.where(
            ((y_pred != y_true) & (((y_prob >= 0.8) & (y_pred == 1)) | ((y_prob <= 0.2) & (y_pred == 0))))
        )[0],
    }
