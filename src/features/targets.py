"""Target construction for next-horizon elevated PM2.5 episodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config


def create_episode_target(
    df: pd.DataFrame,
    pollutant: str = "PM2.5",
    threshold: float = 75.0,
    horizon_hours: int = 24,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Create binary target: will an elevated episode occur in the next H hours?

    Definition
    ----------
    y_t = 1{ max(pollutant_{t+1}, ..., pollutant_{t+H}) >= threshold }

    Important
    ---------
    - Uses strictly *future* observations (t+1 … t+H), never the current value alone
      as the event definition (current level may still appear as a *feature*).
    - The final H rows cannot form a complete horizon → target is NaN and must be dropped
      before training.
    """
    if pollutant not in df.columns:
        raise KeyError(f"Pollutant column '{pollutant}' not in frame.")

    out = df.copy()
    if not out[timestamp_col].is_monotonic_increasing:
        out = out.sort_values(timestamp_col).reset_index(drop=True)

    # Shift(-1) then rolling max over H looks forward over the next H hours
    future = out[pollutant].shift(-1)
    # rolling(H) on a reversed series is an alternative; use fixed forward window via reverse
    forward_max = (
        future.iloc[::-1]
        .rolling(window=horizon_hours, min_periods=horizon_hours)
        .max()
        .iloc[::-1]
    )
    out["y_episode"] = (forward_max >= threshold).astype("float")
    # Rows without full future window stay NaN
    out.loc[forward_max.isna(), "y_episode"] = np.nan
    out["y_future_max_pm25"] = forward_max
    out.attrs["target_threshold"] = threshold
    out.attrs["target_horizon_hours"] = horizon_hours
    return out


def create_target_from_config(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Wrapper reading threshold/horizon from config.yaml."""
    cfg = config or load_config()
    t = cfg["target"]
    return create_episode_target(
        df,
        pollutant=t["pollutant"],
        threshold=float(t["threshold_ug_m3"]),
        horizon_hours=int(t["horizon_hours"]),
    )


def target_prevalence(y: pd.Series) -> dict[str, float]:
    """Return class balance diagnostics."""
    y_clean = y.dropna()
    pos = float((y_clean == 1).sum())
    n = float(len(y_clean))
    return {
        "n": n,
        "n_positive": pos,
        "n_negative": n - pos,
        "prevalence": pos / n if n else float("nan"),
    }
