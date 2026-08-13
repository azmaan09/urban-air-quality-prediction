"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_metrics, expected_calibration_error


def test_perfect_predictions_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["average_precision"] == 1.0
    assert m["f1"] == 1.0


def test_ece_zero_for_perfectly_calibrated_binary():
    # Deterministic bins: probs match outcomes
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(y, p, n_bins=2)
    assert ece == 0.0
