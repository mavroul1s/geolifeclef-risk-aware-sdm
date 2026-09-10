#!/usr/bin/env python
"""Safely list/download selected GeoLifeCLEF 2025 files; secrets are never displayed."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

COMPETITION = "geolifeclef-2025"
DEFAULT_PATTERNS = ("presence", "absence", "observation", "label", "metadata", "landsat", "clim", "bioclim", "soil", "elevation", "land.?cover", "human.?footprint")


def auth_source() -> str | None:
    if os.environ.get("KAGGLE_API_TOKEN"): return "KAGGLE_API_TOKEN"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"): return "legacy environment variables"
    if (Path.home() / ".kaggle" / "kaggle.json").is_file(): return "user-local kaggle.json"
    return None


def get_api():
    source = auth_source()
    if not source:
        raise RuntimeError("No Kaggle authentication found. Configure KAGGLE_API_TOKEN, the legacy pair, or ~/.kaggle/kaggle.json, then accept the competition rules in your browser.")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate(); return api, source
    except Exception as error:
        raise RuntimeError("Kaggle authentication failed or competition rules are not accepted. Sign in, accept the GeoLifeCLEF 2025 rules, then retry.") from error


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("data/raw")); parser.add_argument("--list-only", action="store_true"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args(); api, source = get_api(); print(f"Authentication available via {source}. Listing {COMPETITION} files.")
    try: files = api.competition_list_files(COMPETITION)
    except Exception as error: raise RuntimeError("Cannot access this competition. Confirm the slug and accept its rules while signed in.") from error
    names = [getattr(item, "name", str(item)) for item in files]; patterns = args.include or list(DEFAULT_PATTERNS); selected = [name for name in names if any(re.search(pattern, name, re.I) for pattern in patterns)]
    print("Selected files:"); print("\n".join(f"  - {name}" for name in selected) or "  (none; use --include after reviewing the list)")
    if args.list_only:
        print("All files:"); print("\n".join(f"  - {name}" for name in names)); return 0
    if not selected: raise RuntimeError("No files matched. Run --list-only and pass explicit --include patterns.")
    if args.dry_run: print("Dry run: no files downloaded."); return 0
    args.output.mkdir(parents=True, exist_ok=True)
    for name in selected:
        print(f"Downloading {name}"); api.competition_download_file(COMPETITION, name, path=str(args.output), quiet=False)
    print("Download complete. Run scripts/audit_data.py next."); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except RuntimeError as error: print(f"ERROR: {error}", file=sys.stderr); raise SystemExit(2)

