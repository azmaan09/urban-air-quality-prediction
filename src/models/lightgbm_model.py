"""LightGBM binary classifier for episode prediction."""

from __future__ import annotations

from typing import Any


def _import_lgbm():
    try:
        from lightgbm import LGBMClassifier
    except OSError as exc:  # common on macOS when libomp is missing
        raise ImportError(
            "LightGBM native library failed to load. On macOS install OpenMP:\n"
            "  brew install libomp\n"
            f"Original error: {exc}"
        ) from exc
    return LGBMClassifier


def build_lightgbm(params: dict[str, Any] | None = None, random_state: int = 42):
    LGBMClassifier = _import_lgbm()
    defaults = dict(
        n_estimators=300,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        objective="binary",
        random_state=random_state,
        n_jobs=4,
    )
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, params=None, random_state=42):
    model = build_lightgbm(params=params, random_state=random_state)
    fit_kwargs = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = "average_precision"
    model.fit(X_train, y_train, **fit_kwargs)
    return model
