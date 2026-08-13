"""Automated leakage detection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import build_feature_matrix
from src.features.splits import assert_feature_no_future_dependency, assert_no_temporal_leakage
from src.features.targets import create_episode_target


def _toy_aq_frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2016-01-01", periods=n, freq="h", tz="Asia/Shanghai")
    pm = rng.uniform(10, 120, size=n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "PM2.5": pm,
            "PM10": pm * 1.2,
            "SO2": rng.uniform(1, 30, n),
            "NO2": rng.uniform(10, 80, n),
            "CO": rng.uniform(200, 2000, n),
            "O3": rng.uniform(10, 100, n),
            "TEMP": rng.uniform(-5, 30, n),
            "PRES": rng.uniform(990, 1030, n),
            "DEWP": rng.uniform(-15, 20, n),
            "RAIN": rng.uniform(0, 5, n),
            "WSPM": rng.uniform(0, 6, n),
            "wd_deg": rng.uniform(0, 360, n),
            "RH": rng.uniform(20, 90, n),
        }
    )
    # Minimal missingness indicators expected by feature builder optionally
    for c in ["PM2.5", "TEMP"]:
        df[f"{c}_was_missing"] = 0
    return df


def test_feature_names_have_no_future_tokens():
    df = create_episode_target(_toy_aq_frame(), threshold=75.0, horizon_hours=24)
    feat_df, cols = build_feature_matrix(df)
    for c in cols:
        assert_feature_no_future_dependency(c)
        assert c != "y_future_max_pm25"
        assert c != "y_episode"


def test_target_not_in_features():
    df = create_episode_target(_toy_aq_frame(), threshold=75.0, horizon_hours=24)
    _, cols = build_feature_matrix(df)
    assert "y_episode" not in cols
    assert "y_future_max_pm25" not in cols


def test_temporal_assert_catches_overlap():
    a = pd.Series(pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC"))
    b = pd.Series(pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC"))
    with pytest.raises(AssertionError):
        assert_no_temporal_leakage(a, b)


def test_lag_feature_correlation_with_shifted_series():
    """PM25_lag_1 should equal PM2.5 shifted by 1 (past), not future lead."""
    df = create_episode_target(_toy_aq_frame(300), threshold=75.0, horizon_hours=24)
    feat_df, _ = build_feature_matrix(df)
    expected = feat_df["PM2.5"].shift(1)
    # After dropna in build_feature_matrix, lag_1 should match previous PM2.5
    # Recompute on the returned frame contiguous index:
    # Compare lag to shift within feat_df
    assert np.allclose(
        feat_df["PM25_lag_1"].to_numpy()[1:],
        feat_df["PM2.5"].to_numpy()[:-1],
        equal_nan=True,
    )
