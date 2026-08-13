"""Leakage-safe feature engineering for urban air-quality episode prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config

SEASON_MAP = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def _require_sorted(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if not df[ts_col].is_monotonic_increasing:
        return df.sort_values(ts_col).reset_index(drop=True)
    return df.reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    """Create lag_k features using only past values (shift positive)."""
    out = df.copy()
    prefix = prefix or col.replace(".", "").replace(" ", "_")
    for lag in lags:
        if lag <= 0:
            raise ValueError("Lags must be positive to avoid leakage.")
        out[f"{prefix}_lag_{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows_mean: list[int],
    windows_std: list[int],
    windows_min: list[int],
    windows_max: list[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Rolling stats over the *past* including the current time (closed on the right).

    Implementation detail: use shift(1) before rolling if you need strictly past-only
    excluding t. Here we allow value at t in rolling windows because t is observed
    at prediction time. Future values are never included.
    """
    out = df.copy()
    prefix = prefix or col.replace(".", "").replace(" ", "_")
    for w in windows_mean:
        out[f"{prefix}_rolling_mean_{w}"] = out[col].rolling(w, min_periods=max(1, w // 2)).mean()
    for w in windows_std:
        out[f"{prefix}_rolling_std_{w}"] = out[col].rolling(w, min_periods=max(2, w // 2)).std()
    for w in windows_min:
        out[f"{prefix}_rolling_min_{w}"] = out[col].rolling(w, min_periods=max(1, w // 2)).min()
    for w in windows_max:
        out[f"{prefix}_rolling_max_{w}"] = out[col].rolling(w, min_periods=max(1, w // 2)).max()
    return out


def add_calendar_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    ts = out[ts_col]
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["weekend"] = (out["day_of_week"] >= 5).astype(np.int8)
    out["season"] = out["month"].map(SEASON_MAP)
    # Cyclic encodings (model-friendly)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def add_wind_cyclic(df: pd.DataFrame, deg_col: str = "wd_deg") -> pd.DataFrame:
    out = df.copy()
    if deg_col not in out.columns:
        return out
    rad = np.deg2rad(out[deg_col].astype(float))
    out["wd_sin"] = np.sin(rad)
    out["wd_cos"] = np.cos(rad)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the full supervised learning table.

    Returns
    -------
    frame : DataFrame with features + y_episode (+ metadata cols)
    feature_cols : list of columns safe to use as model inputs
    """
    cfg = config or load_config()
    fcfg = cfg["features"]
    out = _require_sorted(df)

    # --- PM2.5 lags & rolling ---
    out = add_lag_features(out, "PM2.5", list(fcfg["pm25_lags"]), prefix="PM25")
    rw = fcfg["rolling_windows"]
    out = add_rolling_features(
        out,
        "PM2.5",
        windows_mean=list(rw.get("mean", [])),
        windows_std=list(rw.get("std", [])),
        windows_min=list(rw.get("min", [])),
        windows_max=list(rw.get("max", [])),
        prefix="PM25",
    )

    # --- Other pollutants ---
    for pol in ["PM10", "SO2", "NO2", "CO", "O3"]:
        if pol not in out.columns:
            continue
        prefix = pol.replace(".", "")
        out = add_lag_features(out, pol, list(fcfg["pollutant_lags"]), prefix=prefix)
        for w in fcfg.get("pollutant_rolling_mean", []):
            out[f"{prefix}_rolling_mean_{w}"] = (
                out[pol].rolling(w, min_periods=max(1, w // 2)).mean()
            )

    # --- Weather ---
    for wcol, prefix in [
        ("TEMP", "TEMP"),
        ("PRES", "PRES"),
        ("RH", "RH"),
        ("DEWP", "DEWP"),
        ("RAIN", "RAIN"),
        ("WSPM", "WSPM"),
    ]:
        if wcol not in out.columns:
            continue
        out = add_lag_features(out, wcol, list(fcfg["weather_lags"]), prefix=prefix)

    out = add_wind_cyclic(out)
    if fcfg.get("include_calendar", True):
        out = add_calendar_features(out)

    # Season one-hot (drop_first to reduce collinearity for linear models)
    if "season" in out.columns:
        dummies = pd.get_dummies(out["season"], prefix="season", drop_first=True, dtype=np.int8)
        out = pd.concat([out, dummies], axis=1)

    # --- Assemble feature column list ---
    exclude = {
        "timestamp",
        "y_episode",
        "y_future_max_pm25",
        "No",
        "year",
        "month",
        "day",
        "hour",
        "station",
        "wd",
        "season",
        # raw contemporaneous pollutants are allowed at t, but we prefer engineered
        # forms; keep raw PM2.5 + weather at t as they are known at prediction time
    }
    # Keep raw current observations that are known at inference time
    keep_raw = [
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
            "RH",
            "wd_deg",
            "wd_sin",
            "wd_cos",
            "day_of_week",
            "weekend",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "dow_sin",
            "dow_cos",
        ]
        if c in out.columns
    ]

    engineered = [
        c
        for c in out.columns
        if (
            ("_lag_" in c)
            or ("_rolling_" in c)
            or c.endswith("_was_missing")
            or c.startswith("season_")
        )
    ]
    feature_cols = sorted(set(keep_raw + engineered))

    # Drop rows without a full target horizon or with NaNs from long lags
    if "y_episode" in out.columns and dropna_target:
        out = out.dropna(subset=["y_episode"]).copy()
        out["y_episode"] = out["y_episode"].astype(int)

    # After lag creation, drop rows where required features are still NaN
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    out.attrs["feature_cols"] = feature_cols
    return out, feature_cols


def feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Partition features for ablation studies."""
    groups = {
        "pm25_lags": [c for c in feature_cols if c.startswith("PM25_lag_")],
        "rolling": [c for c in feature_cols if "_rolling_" in c],
        "pollutant_lags": [
            c
            for c in feature_cols
            if any(c.startswith(p) and "_lag_" in c for p in ["PM10", "SO2", "NO2", "CO", "O3"])
        ],
        "weather": [
            c
            for c in feature_cols
            if any(
                tok in c
                for tok in ["TEMP", "PRES", "RH", "DEWP", "RAIN", "WSPM", "wd_"]
            )
        ],
        "calendar": [
            c
            for c in feature_cols
            if c
            in {
                "day_of_week",
                "weekend",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "dow_sin",
                "dow_cos",
            }
            or c.startswith("season_")
        ],
        "missingness": [c for c in feature_cols if c.endswith("_was_missing")],
        "raw_pollutants": [
            c for c in feature_cols if c in {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
        ],
    }
    return groups
