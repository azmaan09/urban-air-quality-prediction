"""sklearn HistGradientBoosting fallback when XGBoost/LightGBM lack OpenMP."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier


def train_hist_gbm(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    defaults = dict(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        random_state=random_state,
        class_weight="balanced",
    )
    if params:
        defaults.update(params)
    model = HistGradientBoostingClassifier(**defaults)
    model.fit(X_train, y_train)
    return model
