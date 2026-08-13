"""Chronological and rolling-origin splits — never random shuffle for time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    train_time_range: tuple[str, str]
    valid_time_range: tuple[str, str]
    test_time_range: tuple[str, str]


def create_temporal_split(
    timestamps: pd.Series,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TemporalSplit:
    """
    Split indices by time order.

    Oldest → train | middle → valid | newest → test
    """
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train/valid/test ratios must sum to 1.")

    n = len(timestamps)
    if n < 100:
        raise ValueError(f"Not enough samples to split: n={n}")

    order = np.argsort(timestamps.to_numpy())
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # remainder → test to absorb rounding
    n_test = n - n_train - n_valid

    train_idx = order[:n_train]
    valid_idx = order[n_train : n_train + n_valid]
    test_idx = order[n_train + n_valid :]

    ts = timestamps.to_numpy()

    def _range(idx: np.ndarray) -> tuple[str, str]:
        return str(ts[idx].min()), str(ts[idx].max())

    # Overlap guards
    if timestamps.iloc[train_idx].max() >= timestamps.iloc[valid_idx].min():
        # Equal boundary can happen if duplicate timestamps; enforce strict inequality
        if timestamps.iloc[train_idx].max() > timestamps.iloc[valid_idx].min():
            raise RuntimeError("Train/valid temporal overlap detected.")
    if timestamps.iloc[valid_idx].max() > timestamps.iloc[test_idx].min():
        raise RuntimeError("Valid/test temporal overlap detected.")

    return TemporalSplit(
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        train_time_range=_range(train_idx),
        valid_time_range=_range(valid_idx),
        test_time_range=_range(test_idx),
    )


def create_rolling_origin_splits(
    timestamps: pd.Series,
    n_splits: int = 5,
    min_train_hours: int = 8760,
    valid_hours: int = 2160,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window (rolling-origin) splits for hyperparameter selection.

    Yields (train_idx, valid_idx) pairs. Does not touch the final hold-out test
    region — caller should pass only train+valid timestamps.
    """
    order = np.argsort(timestamps.to_numpy())
    ts_sorted = timestamps.to_numpy()[order]
    n = len(order)

    # Convert "hours" to row counts assuming hourly data after alignment
    min_train = min(min_train_hours, n // 2)
    valid_size = min(valid_hours, max(24, n // (n_splits + 2)))

    # Place validation folds from the end of the provided series backward
    available = n - min_train
    if available < valid_size:
        raise ValueError("Not enough data for rolling-origin validation.")

    step = max(valid_size, available // n_splits)
    folds = 0
    start_valid = min_train
    while start_valid + valid_size <= n and folds < n_splits:
        train_idx = order[:start_valid]
        valid_idx = order[start_valid : start_valid + valid_size]
        yield train_idx, valid_idx
        folds += 1
        start_valid += step


def assert_no_temporal_leakage(
    train_times: pd.Series,
    eval_times: pd.Series,
    dual: str = "eval",
) -> None:
    """Fail hard if evaluation timestamps are not strictly after training."""
    if train_times.max() > eval_times.min():
        raise AssertionError(
            f"Temporal leakage: train max ({train_times.max()}) > "
            f"{dual} min ({eval_times.min()})."
        )


def assert_feature_no_future_dependency(
    feature_col: str,
    forbidden_substrings: tuple[str, ...] = ("future", "lead_", "t+", "y_future"),
) -> None:
    """Name-level guard against obviously leaked feature columns."""
    lower = feature_col.lower()
    for bad in forbidden_substrings:
        if bad in lower:
            raise AssertionError(f"Feature '{feature_col}' looks like future leakage.")
