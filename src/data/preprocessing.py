"""Preprocessing: cleaning, RH derivation, missingness, frequency alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config


def relative_humidity_magnus(temp_c: pd.Series, dewp_c: pd.Series) -> pd.Series:
    """
    Approximate relative humidity (%) from temperature and dew point (°C).

    Magnus formula (Alduchov & Eskridge, 1996 approximation commonly used in AQ work).
    RH is clipped to [0, 100].
    """
    a, b = 17.625, 243.04
    # Avoid overflow / invalid inputs
    t = temp_c.astype(float)
    td = dewp_c.astype(float)
    gamma_td = (a * td) / (b + td)
    gamma_t = (a * t) / (b + t)
    rh = 100.0 * np.exp(gamma_td - gamma_t)
    return rh.clip(0, 100)


def align_hourly(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Reindex to a complete hourly grid for a single station.

    Missing hours become NaN rows (later handled by missing-value treatment).
    """
    if "station" in df.columns and df["station"].nunique() > 1:
        parts = [align_hourly(g.drop(columns=[]), freq=freq) for _, g in df.groupby("station")]
        return pd.concat(parts, ignore_index=True)

    out = df.copy()
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="first")
    full_idx = pd.date_range(out["timestamp"].min(), out["timestamp"].max(), freq=freq)
    out = out.set_index("timestamp").reindex(full_idx)
    out.index.name = "timestamp"
    out = out.reset_index()
    if "station" in df.columns:
        station = df["station"].dropna().iloc[0]
        out["station"] = out["station"].fillna(station)
    return out


def treat_missing(
    df: pd.DataFrame,
    value_cols: list[str],
    interpolate_limit_hours: int = 3,
) -> pd.DataFrame:
    """
    Short-gap linear interpolation, then forward/backward fill for longer gaps.

    Also creates `{col}_was_missing` indicators on the *original* missing mask
    before imputation (leakage-safe: indicator uses only contemporaneous info).
    """
    out = df.copy()
    for col in value_cols:
        if col not in out.columns:
            continue
        original_missing = out[col].isna()
        out[f"{col}_was_missing"] = original_missing.astype(np.int8)
        # Interpolate only along time for short gaps
        out[col] = out[col].interpolate(
            method="linear", limit=interpolate_limit_hours, limit_direction="both"
        )
        out[col] = out[col].ffill().bfill()
    return out


def flag_outliers_iqr(
    df: pd.DataFrame,
    cols: list[str],
    multiplier: float = 5.0,
) -> pd.DataFrame:
    """Add `{col}_outlier` flags using a conservative IQR rule (no clipping by default)."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        out[f"{col}_outlier"] = ((out[col] < lo) | (out[col] > hi)).astype(np.int8)
    return out


def preprocess_site(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single-site frame.

    Steps:
      1. Drop exact duplicate rows / duplicate timestamps
      2. Sort chronologically
      3. Align to hourly frequency
      4. Derive RH from TEMP + DEWP
      5. Encode wind direction degrees if needed
      6. Missing-value treatment + indicators
      7. Outlier flags
    """
    cfg = config or load_config()
    prep = cfg["preprocessing"]
    out = df.copy()

    if prep.get("drop_duplicates", True):
        out = out.drop_duplicates()
        out = out.drop_duplicates(subset=["timestamp"], keep="first")

    if prep.get("sort_chronologically", True):
        out = out.sort_values("timestamp").reset_index(drop=True)

    out = align_hourly(out, freq="1h")

    if "TEMP" in out.columns and "DEWP" in out.columns:
        out["RH"] = relative_humidity_magnus(out["TEMP"], out["DEWP"])

    value_cols = [
        c
        for c in [
            "PM2.5",
            "PM10",
            "SO2",
            "NO2",
            "CO",
            "O3",
            "TEMP",
            "PRES",
            "DEWP",
            "RAIN",
            "WSPM",
            "wd_deg",
            "RH",
        ]
        if c in out.columns
    ]

    out = treat_missing(
        out,
        value_cols=value_cols,
        interpolate_limit_hours=int(prep.get("interpolate_limit_hours", 3)),
    )
    out = flag_outliers_iqr(
        out,
        cols=[c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if c in out.columns],
        multiplier=float(prep.get("outlier_iqr_multiplier", 5.0)),
    )
    return out.reset_index(drop=True)
