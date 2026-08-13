"""Logistic regression model wrapper."""

from __future__ import annotations

from src.models.baselines import train_logistic


def build_logistic(random_state: int = 42):
    """Return an unfitted factory-compatible trainer."""
    return lambda X, y: train_logistic(X, y, random_state=random_state)
