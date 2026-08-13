"""XGBoost binary classifier for episode prediction."""

from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier


def build_xgboost(params: dict[str, Any] | None = None, random_state: int = 42) -> XGBClassifier:
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=4,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def train_xgboost(X_train, y_train, X_valid=None, y_valid=None, params=None, random_state=42):
    model = build_xgboost(params=params, random_state=random_state)
    # Compute scale_pos_weight from training labels
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    model.set_params(scale_pos_weight=neg / pos)

    fit_kwargs = {}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model
