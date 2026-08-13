"""Baseline models: persistence, logistic regression, decision tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass
class FittedModel:
    name: str
    model: Any
    feature_cols: list[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # Persistence returns a 1-d probability-like score already
        return np.asarray(self.model.predict(X), dtype=float)


class PersistenceBaseline:
    """
    Predict P(episode) ≈ 1 if current PM2.5 is already elevated, else 0.

    Uses the column index of PM2.5 in the feature matrix, or a dedicated vector.
    This is a deliberately simple operational baseline.
    """

    def __init__(self, pm25_index: int, threshold: float = 75.0):
        self.pm25_index = pm25_index
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        pm = X[:, self.pm25_index]
        p = (pm >= self.threshold).astype(float)
        # Soften to avoid 0/1 extremes for Brier comparisons
        p = 0.05 + 0.90 * p
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_persistence(X_train, y_train, pm25_index: int, threshold: float) -> PersistenceBaseline:
    return PersistenceBaseline(pm25_index, threshold).fit(X_train, y_train)


def train_logistic(X_train, y_train, random_state: int = 42) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_decision_tree(
    X_train,
    y_train,
    random_state: int = 42,
    max_depth: int = 6,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        min_samples_leaf=50,
    )
    clf.fit(X_train, y_train)
    return clf
