"""Probability calibration utilities (Platt / isotonic)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibrationMethod = Literal["none", "platt", "isotonic"]


class IdentityCalibrator:
    """No-op calibrator that returns raw scores/probabilities."""

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return np.asarray(y_prob, dtype=float)


class PlattCalibrator:
    """Sigmoid (Platt) scaling fit on validation probabilities."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        x = np.asarray(y_prob, dtype=float).reshape(-1, 1)
        self.model.fit(x, np.asarray(y_true).astype(int))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        x = np.asarray(y_prob, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator."""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        self.model.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true).astype(int))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(y_prob, dtype=float))


def fit_calibrator(
    y_valid_true: np.ndarray,
    y_valid_prob: np.ndarray,
    method: CalibrationMethod = "platt",
):
    """Fit a calibrator on validation probabilities only (never test)."""
    if method == "none":
        return IdentityCalibrator().fit(y_valid_prob, y_valid_true)
    if method == "platt":
        return PlattCalibrator().fit(y_valid_prob, y_valid_true)
    if method == "isotonic":
        return IsotonicCalibrator().fit(y_valid_prob, y_valid_true)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(calibrator, y_prob: np.ndarray) -> np.ndarray:
    return np.asarray(calibrator.transform(y_prob), dtype=float)
