#!/usr/bin/env python3
"""Download the Beijing Multi-Site Air-Quality dataset into data/raw/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/download_data.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import dataset_summary, download_dataset, load_all_sites, load_site
from src.data.validation import print_validation_report, validate_raw_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & sanity-check AQ dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument(
        "--site",
        default=None,
        help="Optional single site to summarize (default: config primary_site)",
    )
    args = parser.parse_args()

    extract_dir = download_dataset(force=args.force)
    print(f"Extract dir: {extract_dir}")

    df = load_site(site=args.site)
    report = validate_raw_frame(df)
    print_validation_report(report)
    summary = dataset_summary(df)
    print(json.dumps(summary, indent=2))

    # Also write a lightweight dataset card for the report
    out = ROOT / "data" / "external" / "dataset_card.json"
    all_df = load_all_sites()
    card = {
        "name": "Beijing Multi-Site Air-Quality Data",
        "uci_id": 501,
        "url": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        "primary_site_summary": summary,
        "all_sites": {
            "n_rows": int(len(all_df)),
            "stations": sorted(all_df["station"].unique().tolist()),
            "time_start": str(all_df["timestamp"].min()),
            "time_end": str(all_df["timestamp"].max()),
        },
        "available_variables": {
            "pollutants": ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"],
            "meteorology": ["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"],
            "derived_later": ["RH (from TEMP+DEWP via Magnus formula)"],
            "not_native": ["relative humidity (derived)", "explicit timezone offset column"],
        },
        "license": "CC BY 4.0 (via UCI redistribution)",
        "citation": (
            "Zhang, S., Guo, B., Dong, A., He, J., Xu, B., & Chen, S. X. (2017). "
            "Cautionary Tales on Air-Quality Improvement in Beijing. "
            "Proceedings of the Royal Society A, 473(2205)."
        ),
    }
    out.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Wrote dataset card → {out}")


if __name__ == "__main__":
    main()
