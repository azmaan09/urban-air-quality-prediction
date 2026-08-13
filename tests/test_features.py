"""Tests for lag / rolling feature engineering (leakage-safe)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import add_lag_features, add_rolling_features


def test_lag_uses_past_values_only():
    df = pd.DataFrame({"PM2.5": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = add_lag_features(df, "PM2.5", lags=[1, 2], prefix="PM25")
    assert np.isnan(out.loc[0, "PM25_lag_1"])
    assert out.loc[1, "PM25_lag_1"] == 1.0
    assert out.loc[2, "PM25_lag_2"] == 1.0
    # Ensure lag does not equal a future value incorrectly
    assert out.loc[1, "PM25_lag_1"] != 3.0


def test_lag_rejects_non_positive():
    df = pd.DataFrame({"PM2.5": [1.0, 2.0]})
    try:
        add_lag_features(df, "PM2.5", lags=[0], prefix="PM25")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rolling_mean_window():
    df = pd.DataFrame({"PM2.5": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = add_rolling_features(
        df,
        "PM2.5",
        windows_mean=[3],
        windows_std=[],
        windows_min=[],
        windows_max=[],
        prefix="PM25",
    )
    # At index 2: mean(1,2,3)=2
    assert abs(out.loc[2, "PM25_rolling_mean_3"] - 2.0) < 1e-9
