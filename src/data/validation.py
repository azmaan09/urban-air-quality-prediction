"""Data quality validation for air-quality time series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
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
]

# Physically plausible bounds (hourly). Used as soft checks, not hard filters.
PLAUSIBLE_RANGES = {
    "PM2.5": (0, 1000),
    "PM10": (0, 1500),
    "SO2": (0, 1000),
    "NO2": (0, 500),
    "CO": (0, 10000),   # µg/m³ in this dataset (not ppm)
    "O3": (0, 500),
    "TEMP": (-40, 50),
    "PRES": (900, 1100),
    "DEWP": (-50, 40),
    "RAIN": (0, 200),
    "WSPM": (0, 50),
}


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("Validation failed:\n- " + "\n- ".join(self.errors))


def validate_raw_frame(df: pd.DataFrame, expected_freq: str = "1h") -> ValidationReport:
    """
    Validate a loaded site/all-sites frame before preprocessing.

    Hard errors block the pipeline. Warnings are informational (missingness,
    outliers, irregular gaps).
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            errors.append("`timestamp` must be datetime64.")
        else:
            if df["timestamp"].isna().any():
                errors.append("`timestamp` contains nulls.")
            if not df["timestamp"].is_monotonic_increasing and "station" not in df.columns:
                warnings.append("Timestamps are not sorted ascending.")
            # Duplicate timestamps (per station if present)
            if "station" in df.columns:
                dup = df.duplicated(subset=["station", "timestamp"]).sum()
            else:
                dup = df.duplicated(subset=["timestamp"]).sum()
            stats["n_duplicate_timestamps"] = int(dup)
            if dup > 0:
                warnings.append(f"Found {dup} duplicate timestamp rows.")

            # Gap analysis (single station frames)
            if "station" not in df.columns or df["station"].nunique() == 1:
                deltas = df["timestamp"].sort_values().diff().dropna()
                expected = pd.Timedelta(expected_freq)
                irregular = (deltas != expected).sum()
                stats["n_irregular_gaps"] = int(irregular)
                stats["max_gap"] = str(deltas.max()) if len(deltas) else None
                if irregular > 0:
                    warnings.append(
                        f"Found {irregular} gaps that are not exactly {expected_freq}."
                    )

    # Missingness
    present = [c for c in REQUIRED_COLUMNS if c in df.columns and c != "timestamp"]
    miss_pct = df[present].isna().mean() * 100
    stats["missing_pct"] = miss_pct.round(2).to_dict()
    high_miss = miss_pct[miss_pct > 20]
    if len(high_miss):
        warnings.append(
            "Columns with >20% missing: "
            + ", ".join(f"{k}={v:.1f}%" for k, v in high_miss.items())
        )

    # Soft physical-range checks
    range_violations: dict[str, int] = {}
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        s = df[col]
        n_bad = int(((s < lo) | (s > hi)).fillna(False).sum())
        if n_bad:
            range_violations[col] = n_bad
            warnings.append(
                f"{col}: {n_bad} values outside plausible range [{lo}, {hi}]."
            )
    stats["range_violations"] = range_violations

    # PM2.5 vs PM10 consistency (PM2.5 should usually be ≤ PM10)
    if "PM2.5" in df.columns and "PM10" in df.columns:
        both = df[["PM2.5", "PM10"]].dropna()
        n_inconsistent = int((both["PM2.5"] > both["PM10"] * 1.05).sum())
        stats["pm25_gt_pm10"] = n_inconsistent
        if n_inconsistent > 0:
            warnings.append(
                f"PM2.5 exceeds PM10 in {n_inconsistent} rows (possible sensor noise)."
            )

    stats["n_rows"] = int(len(df))
    ok = len(errors) == 0
    return ValidationReport(ok=ok, errors=errors, warnings=warnings, stats=stats)


def print_validation_report(report: ValidationReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"[validation] {status}")
    for e in report.errors:
        print(f"  ERROR: {e}")
    for w in report.warnings:
        print(f"  WARN:  {w}")
    print(f"  stats: n_rows={report.stats.get('n_rows')}")
