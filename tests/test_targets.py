"""Tests for target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.targets import create_episode_target


def test_target_uses_future_only():
    # Need enough trailing hours so early indices have a full 24h horizon
    n = 80
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    pm = np.zeros(n)
    pm[40] = 100  # elevated only at t=40
    df = pd.DataFrame({"timestamp": ts, "PM2.5": pm})
    out = create_episode_target(df, threshold=75.0, horizon_hours=24)

    # Indices whose next-24h window includes t=40 are 16..39
    assert out.loc[15, "y_episode"] == 0
    assert out.loc[16, "y_episode"] == 1
    assert out.loc[39, "y_episode"] == 1
    # At the spike itself, future window is 41..64 → no elevation
    assert out.loc[40, "y_episode"] == 0
    # Last 24 rows incomplete → NaN
    assert out["y_episode"].isna().sum() == 24


def test_target_prevalence_with_constant_low():
    n = 100
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "PM2.5": np.ones(n) * 10})
    out = create_episode_target(df, threshold=75.0, horizon_hours=24)
    y = out["y_episode"].dropna()
    assert (y == 0).all()
