"""Decision tree model wrapper."""

from __future__ import annotations

from src.models.baselines import train_decision_tree


def build_decision_tree(random_state: int = 42, max_depth: int = 6):
    return lambda X, y: train_decision_tree(X, y, random_state=random_state, max_depth=max_depth)
