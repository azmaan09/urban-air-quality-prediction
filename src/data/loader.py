"""Dataset download and loading for Beijing Multi-Site Air-Quality Data (UCI #501)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.config import load_config, project_path

# Wind direction string → degrees (center of 16-point compass)
WD_TO_DEG = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def _download_site_csvs_as_zip(raw_dir: Path) -> bytes | None:
    """
    Fallback: fetch per-site CSVs from the UCI openml / mirror layout and zip them.

    Returns zip bytes, or None if unavailable.
    """
    # Public GitHub mirror of the 12 PRSA station files (commonly used in coursework)
    base = (
        "https://raw.githubusercontent.com/AlexandruPascov/beijing_air_quality/"
        "master/PRSA_Data_20130301-20170228/"
    )
    stations = [
        "Aotizhongxin",
        "Changping",
        "Dingling",
        "Dongsi",
        "Guanyuan",
        "Gucheng",
        "Huairou",
        "Nongzhanguan",
        "Shunyi",
        "Tiantan",
        "Wanliu",
        "Wanshouxigong",
    ]
    buf = io.BytesIO()
    ok = 0
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for station in stations:
            fname = f"PRSA_Data_{station}_20130301-20170228.csv"
            url = base + fname
            try:
                print(f"[loader] Fallback CSV: {url}")
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                zf.writestr(fname, r.content)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[loader] Fallback failed for {station}: {exc}")
    if ok == 0:
        return None
    print(f"[loader] Packaged {ok} station CSV(s) into zip via fallback mirror")
    return buf.getvalue()


def download_dataset(
    force: bool = False,
    config_path: str | Path | None = None,
) -> Path:
    """
    Download the UCI Beijing Multi-Site Air-Quality zip into data/raw/.

    Returns the path to the extracted directory containing PRSA_Data_*.csv files.
    """
    cfg = load_config(config_path)
    raw_dir = project_path(cfg["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = raw_dir / "beijing_multisite"
    marker = extract_dir / ".download_complete"

    if marker.exists() and not force:
        print(f"[loader] Dataset already present at {extract_dir}")
        return extract_dir

    # Multiple mirrors: UCI primary, then community mirrors if UCI/proxy fails.
    urls = [
        cfg["dataset"]["download_url"],
        "https://github.com/jbrownlee/Datasets/releases/download/Beijing/PRSA_data_2010.1.1-2014.12.31.csv",  # single-site fallback marker
        # Official-style mirror used widely in research repos:
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv",
    ]
    # Prefer the full multi-site archive from alternative hosts first if configured.
    extra = cfg["dataset"].get("mirror_urls") or []
    urls = list(extra) + urls

    zip_path = raw_dir / "beijing_multisite.zip"
    content: bytes | None = None
    last_err: Exception | None = None
    for url in urls:
        if not url.lower().endswith(".zip"):
            continue
        try:
            print(f"[loader] Trying download: {url}")
            resp = requests.get(
                url,
                timeout=180,
                headers={"User-Agent": "urban-aq-research/1.0"},
            )
            resp.raise_for_status()
            content = resp.content
            print(f"[loader] Downloaded {len(content) / 1e6:.1f} MB from {url}")
            break
        except Exception as exc:  # noqa: BLE001 — try next mirror
            print(f"[loader] Failed ({type(exc).__name__}): {exc}")
            last_err = exc

    if content is None:
        # Last resort: download individual site CSVs from a known public mirror tree
        content = _download_site_csvs_as_zip(raw_dir)
        if content is None and last_err is not None:
            raise RuntimeError(
                "Could not download Beijing multi-site dataset from any mirror."
            ) from last_err

    zip_path.write_bytes(content)
    print(f"[loader] Saved archive: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    if extract_dir.exists():
        # Clean partial extracts on force re-download
        for p in extract_dir.rglob("*"):
            if p.is_file():
                p.unlink()

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(extract_dir)

    # Some UCI zips nest an extra folder; normalize by finding CSVs
    csvs = list(extract_dir.rglob("PRSA_Data_*.csv"))
    if not csvs:
        # Alternate naming in some mirrors
        csvs = list(extract_dir.rglob("*.csv"))
    if not csvs:
        raise RuntimeError(
            f"Download succeeded but no CSV files found under {extract_dir}. "
            "Check the archive structure."
        )

    marker.write_text("ok\n", encoding="utf-8")
    print(f"[loader] Extracted {len(csvs)} CSV file(s) under {extract_dir}")
    return extract_dir


def list_site_files(raw_root: Path | None = None) -> list[Path]:
    """Return sorted list of per-site PRSA CSV paths."""
    if raw_root is None:
        cfg = load_config()
        raw_root = project_path(cfg["dataset"]["raw_dir"]) / "beijing_multisite"
    files = sorted(raw_root.rglob("PRSA_Data_*.csv"))
    if not files:
        files = sorted(p for p in raw_root.rglob("*.csv") if p.is_file())
    return files


def _parse_timestamp(df: pd.DataFrame) -> pd.Series:
    """Build timezone-aware timestamps from year/month/day/hour columns."""
    ts = pd.to_datetime(
        dict(
            year=df["year"].astype(int),
            month=df["month"].astype(int),
            day=df["day"].astype(int),
            hour=df["hour"].astype(int),
        ),
        errors="raise",
    )
    # Observations are local China Standard Time (no DST)
    return ts.dt.tz_localize("Asia/Shanghai")


def load_site(
    site: str | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load a single monitoring site into a tidy DataFrame.

    Returns columns including `timestamp`, pollutants, meteorology, and `station`.
    """
    cfg = load_config(config_path)
    site = site or cfg["dataset"]["primary_site"]
    files = list_site_files()
    if not files:
        raise FileNotFoundError(
            "No raw CSV files found. Run scripts/download_data.py first."
        )

    match = [p for p in files if site.lower() in p.name.lower()]
    if not match:
        available = [p.stem for p in files]
        raise ValueError(f"Site '{site}' not found. Available: {available}")

    path = match[0]
    df = pd.read_csv(path)
    df["timestamp"] = _parse_timestamp(df)
    if "wd" in df.columns:
        df["wd_deg"] = df["wd"].map(WD_TO_DEG)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["source_file"] = str(path)
    df.attrs["site"] = site
    return df


def load_all_sites(
    sites: Iterable[str] | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and concatenate multiple sites (or all available)."""
    files = list_site_files()
    if not files:
        raise FileNotFoundError(
            "No raw CSV files found. Run scripts/download_data.py first."
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        # Filenames look like PRSA_Data_Aotizhongxin_20130301-20170228.csv
        stem = path.stem
        site_name = stem.replace("PRSA_Data_", "").split("_2013")[0]
        if sites is not None and site_name not in sites:
            continue
        df = pd.read_csv(path)
        df["timestamp"] = _parse_timestamp(df)
        if "wd" in df.columns:
            df["wd_deg"] = df["wd"].map(WD_TO_DEG)
        if "station" not in df.columns:
            df["station"] = site_name
        frames.append(df)

    if not frames:
        raise ValueError("No sites matched the requested filter.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["station", "timestamp"]).reset_index(drop=True)
    return out


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a machine-readable summary for documentation / reports."""
    pollutant_cols = [c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"] if c in df.columns]
    met_cols = [c for c in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd", "wd_deg"] if c in df.columns]
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "stations": sorted(df["station"].unique().tolist()) if "station" in df.columns else [],
        "pollutant_columns": pollutant_cols,
        "meteorology_columns": met_cols,
        "missing_counts": df[pollutant_cols + met_cols].isna().sum().to_dict(),
        "missing_pct": (df[pollutant_cols + met_cols].isna().mean() * 100).round(2).to_dict(),
    }
