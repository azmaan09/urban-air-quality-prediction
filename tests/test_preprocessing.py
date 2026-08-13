"""Tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import align_hourly, relative_humidity_magnus, treat_missing


def test_rh_between_0_and_100():
    temp = pd.Series([20.0, 0.0, 30.0])
    dewp = pd.Series([10.0, -5.0, 30.0])
    rh = relative_humidity_magnus(temp, dewp)
    assert (rh >= 0).all() and (rh <= 100).all()
    # When temp == dew point, RH ≈ 100
    assert abs(rh.iloc[2] - 100) < 1.0


def test_align_hourly_inserts_gaps():
    ts = pd.to_datetime(["2015-01-01 00:00", "2015-01-01 01:00", "2015-01-01 03:00"]).tz_localize(
        "UTC"
    )
    df = pd.DataFrame({"timestamp": ts, "PM2.5": [1.0, 2.0, 4.0], "station": ["A"] * 3})
    out = align_hourly(df)
    assert len(out) == 4  # hour 02 inserted
    assert out["PM2.5"].isna().sum() == 1


def test_treat_missing_creates_indicator():
    df = pd.DataFrame({"PM2.5": [1.0, np.nan, 3.0, np.nan, 5.0]})
    out = treat_missing(df, ["PM2.5"], interpolate_limit_hours=2)
    assert "PM2.5_was_missing" in out.columns
    assert out["PM2.5_was_missing"].tolist() == [0, 1, 0, 1, 0]
    assert out["PM2.5"].isna().sum() == 0
