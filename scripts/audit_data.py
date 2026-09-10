#!/usr/bin/env python
"""Build a lightweight data card from filenames and NPY/NPZ headers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HINTS = {"landsat": ("landsat",), "climate": ("clim", "bioclim"), "static": ("soil", "elevation", "land", "footprint"), "labels": ("label", "presence", "absence", "observation")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("data/raw")); parser.add_argument("--report-dir", type=Path, default=Path("data/reports")); args = parser.parse_args()
    if not args.data_root.is_dir(): raise SystemExit(f"Data directory not found: {args.data_root}. Run download_data.py first.")
    files = []
    for path in sorted(item for item in args.data_root.rglob("*") if item.is_file()):
        name = str(path.relative_to(args.data_root)); row = {"path": name, "bytes": path.stat().st_size, "modalities": [kind for kind, hints in HINTS.items() if any(hint in name.lower() for hint in hints)]}
        if path.suffix == ".npy": row["shape"] = list(np.load(path, mmap_mode="r").shape)
        if path.suffix == ".npz":
            with np.load(path, mmap_mode="r", allow_pickle=False) as archive: row["arrays"] = {key: list(archive[key].shape) for key in archive.files}
        files.append(row)
    report = {"data_root": str(args.data_root), "file_count": len(files), "files": files, "discovered_modalities": sorted({modality for row in files for modality in row["modalities"]}), "note": "Inventory only; inspect raw schema before building canonical NPZ splits."}
    args.report_dir.mkdir(parents=True, exist_ok=True); (args.report_dir / "data_card.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# GeoLifeCLEF 2025 data card (initial audit)", "", f"Files: {len(files)}", f"Discovered modalities: {', '.join(report['discovered_modalities']) or 'none'}", "", "| File | Bytes | Inferred modalities | Array headers |", "| --- | ---: | --- | --- |"]
    lines.extend(f"| {row['path']} | {row['bytes']} | {', '.join(row['modalities']) or '-'} | {row.get('shape', row.get('arrays', '-'))} |" for row in files)
    (args.report_dir / "data_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print(f"Wrote {args.report_dir / 'data_card.md'}")


if __name__ == "__main__": main()

